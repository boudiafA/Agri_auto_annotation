#!/usr/bin/env python3
"""
Enhanced SAM Batch Processing - Dynamic Dual GPU with Advanced Optimizations
Features:
- Dynamic work distribution (no pre-splitting - images go to available GPUs)
- Dual GPU processing (GPU 0 and GPU 1)
- Resume functionality (skip already processed images)
- Adaptive points_per_batch to handle CUDA OOM errors
- GPU-specific memory optimization
- OPTIMIZED: Parallel image pre-loading with ThreadPoolExecutor
- OPTIMIZED: Selective multi-scale preprocessing for large images
- OPTIMIZED: Smart size prediction and variant selection
- Dataset completion tracking with .done files
- Intelligent VRAM-based points_per_batch estimation
"""

import os
import time
import glob
import argparse
import json
from pathlib import Path
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import threading
import queue
import math

# GPU settings - will be set per process
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import cv2
import torch
import numpy as np
from tqdm import tqdm

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    print("Error: segment_anything not found. Install with: pip install segment-anything")
    sys.exit(1)

try:
    from pycocotools import mask as mask_util
    HAS_PYCOCOTOOLS = True
except ImportError:
    print("Warning: pycocotools not found. Will use basic RLE encoding.")
    HAS_PYCOCOTOOLS = False


class OptimizedImagePreprocessor:
    """Advanced preprocessor with parallel processing and smart multi-scale handling"""
    
    def __init__(self):
        self.size_variants = [1536, 2048, 2560]  # Available size options
        
    def predict_optimal_size(self, original_shape):
        """Predict optimal processing size based on image characteristics"""
        h, w = original_shape[:2]
        total_pixels = h * w
        
        # Heuristic based on image size and typical GPU performance
        if total_pixels > 12000000:  # > 12MP - very large
            return 1536
        elif total_pixels > 6000000:  # > 6MP - large
            return 2048  
        else:
            return 2560  # smaller images can handle larger processing sizes
    
    def should_create_variants(self, original_shape):
        """Decide whether to create multiple size variants"""
        h, w = original_shape[:2]
        max_dim = max(h, w)
        
        # Only create variants for very large images where size flexibility matters
        # This balances memory usage vs. OOM robustness
        return max_dim > 3000  # Only for images larger than 3000px
    
    def smart_resize(self, image, target_size):
        """Intelligent resizing with high-quality interpolation"""
        h, w = image.shape[:2]
        
        if max(h, w) <= target_size:
            return image
            
        scale = target_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        # Use high-quality interpolation for better results
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    def preprocess_single_image(self, image_path):
        """Enhanced preprocessing with selective multi-scale generation"""
        try:
            # Load original image
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Convert BGR to RGB (done once on CPU - faster than GPU)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_shape = image.shape
            
            # Decide on preprocessing strategy
            if self.should_create_variants(original_shape):
                # Create multiple variants for large images
                variants = {}
                for target_size in self.size_variants:
                    resized_image = self.smart_resize(image, target_size)
                    variants[target_size] = {
                        'image_data': resized_image,
                        'actual_size': resized_image.shape[:2],
                        'scale_factor': min(target_size / original_shape[0], target_size / original_shape[1])
                    }
                
                return {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'original_shape': original_shape,
                    'variants': variants,
                    'optimal_size_hint': self.predict_optimal_size(original_shape),
                    'has_variants': True
                }
            else:
                # Single size for smaller images (saves memory)
                optimal_size = self.predict_optimal_size(original_shape)
                processed_image = self.smart_resize(image, optimal_size)
                
                return {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'original_shape': original_shape,
                    'image_data': processed_image,  # Single processed image
                    'actual_size': processed_image.shape[:2],
                    'optimal_size_hint': optimal_size,
                    'has_variants': False
                }
                
        except Exception as e:
            print(f"Error preprocessing {os.path.basename(image_path)}: {e}")
            return None


class AdaptiveSAMProcessor:
    def __init__(self, model_type: str = "vit_h", checkpoint_path: str = None,
                 gpu_id: int = 0, optimize_memory: bool = True, output_dir: str = None):
        """
        Enhanced SAM processor with adaptive points_per_batch for single GPU
        """
        # Set GPU-specific environment variables
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        self.model_type = model_type
        self.gpu_id = gpu_id
        self.optimize_memory = optimize_memory
        self.output_dir = output_dir # This is dataset_output_dir
        self.device = f"cuda:0"  # Always 0 since we set CUDA_VISIBLE_DEVICES
        
        # Adaptive batch size parameters
        self.initial_points_per_batch = 32  # Default fallback
        self.current_points_per_batch = self.initial_points_per_batch
        self.min_points_per_batch = 1  # Minimum safe value
        self.max_points_per_batch = 64  # Maximum to try (conservative, will be updated by VRAM estimation)
        self.oom_count = 0
        self.successful_batches = 0
        
        # Create output directory for SAM jsons
        if self.output_dir:
            self.sam_output_dir = os.path.join(self.output_dir, "sam")
            os.makedirs(self.sam_output_dir, exist_ok=True)
        else:
            # This case should ideally not happen if output_dir is always provided
            self.sam_output_dir = None 
        
        print(f"Initializing Adaptive SAM Processor for GPU {gpu_id}")
        print(f"Device: {self.device} (Physical GPU {gpu_id})")
        print(f"Model: {model_type}")
        
        # Load model
        self.sam = self._load_stable_model(checkpoint_path)
        
        # Initialize mask generator with adaptive settings
        self.mask_generator = None # Will be initialized after VRAM estimation
        
        # Setup optimizations
        self._setup_stable_optimizations()
        
        print(f"GPU {gpu_id} model loaded successfully!")
    
    def estimate_optimal_points_per_batch(self, test_image_path: str = None, test_image_data: np.ndarray = None):
        """
        Estimate optimal points_per_batch based on actual VRAM usage with a test image
        """
        print(f"GPU {self.gpu_id}: Estimating optimal points_per_batch based on VRAM usage...")
        
        try:
            # Clear GPU cache first
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Get initial VRAM usage (model only)
            initial_vram = torch.cuda.memory_allocated(0)
            total_vram = torch.cuda.get_device_properties(0).total_memory
            
            print(f"GPU {self.gpu_id}: Model VRAM usage: {initial_vram / (1024**3):.2f}GB")
            print(f"GPU {self.gpu_id}: Total VRAM: {total_vram / (1024**3):.2f}GB")
            
            # Load test image
            test_image = None
            if test_image_data is not None:
                test_image = test_image_data.copy()
            elif test_image_path and os.path.exists(test_image_path):
                test_image = cv2.imread(test_image_path)
                if test_image is not None:
                    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
            
            if test_image is None:
                print(f"GPU {self.gpu_id}: No test image available, using conservative default")
                self.current_points_per_batch = self._get_conservative_batch_size()
                self.initial_points_per_batch = self.current_points_per_batch
                self.max_points_per_batch = max(self.max_points_per_batch, self.current_points_per_batch * 2) # Update max based on conservative estimate
                self._initialize_mask_generator()
                return self.current_points_per_batch
            
            # Resize test image to typical processing size
            h, w = test_image.shape[:2]
            if max(h, w) > 2048:
                scale = 2048 / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                test_image = cv2.resize(test_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            print(f"GPU {self.gpu_id}: Using test image of size {test_image.shape[:2]}")
            
            # Try different points_per_batch values to find optimal
            test_values = [32, 64, 96, 128, 192, 256, 384, 512] # More aggressive test values
            optimal_batch_size = 32 # Fallback
            
            for test_batch_size in test_values:
                try:
                    print(f"GPU {self.gpu_id}: Testing points_per_batch={test_batch_size}...")
                    
                    # Create temporary mask generator
                    temp_generator = SamAutomaticMaskGenerator(
                        model=self.sam,
                        points_per_side=32,
                        pred_iou_thresh=0.86,
                        stability_score_thresh=0.92,
                        crop_n_layers=1,
                        crop_n_points_downscale_factor=2,
                        min_mask_region_area=100,
                        points_per_batch=test_batch_size
                    )
                    
                    # Clear cache before test
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    start_vram = torch.cuda.memory_allocated(0)
                    
                    # Process test image
                    with torch.no_grad():
                        masks = temp_generator.generate(test_image)
                    
                    # Check peak VRAM usage
                    peak_vram = torch.cuda.max_memory_allocated(0)
                    vram_used = peak_vram - initial_vram
                    vram_remaining = total_vram - peak_vram
                    vram_usage_percent = (peak_vram / total_vram) * 100
                    
                    print(f"GPU {self.gpu_id}: points_per_batch={test_batch_size} -> "
                          f"VRAM: {vram_used/(1024**3):.2f}GB used, "
                          f"{vram_remaining/(1024**3):.2f}GB free, "
                          f"{vram_usage_percent:.1f}% total, "
                          f"masks: {len(masks)}")
                    
                    # If we're using less than 80% of VRAM, this batch size is safe
                    if vram_usage_percent < 80:
                        optimal_batch_size = test_batch_size
                        # Reset max memory stats for next test
                        torch.cuda.reset_peak_memory_stats(0)
                    else:
                        print(f"GPU {self.gpu_id}: Approaching VRAM limit, stopping at points_per_batch={optimal_batch_size}")
                        break
                        
                    # Clean up
                    del temp_generator, masks
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(0)
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"GPU {self.gpu_id}: OOM at points_per_batch={test_batch_size}, using {optimal_batch_size}")
                    torch.cuda.empty_cache()
                    break
                except Exception as e:
                    print(f"GPU {self.gpu_id}: Error testing points_per_batch={test_batch_size}: {e}")
                    torch.cuda.empty_cache()
                    break
            
            # Set the optimal batch size
            self.current_points_per_batch = optimal_batch_size
            self.initial_points_per_batch = optimal_batch_size
            self.max_points_per_batch = max(self.max_points_per_batch, optimal_batch_size * 2) # Update max based on estimation
            
            print(f"GPU {self.gpu_id}: Optimal points_per_batch estimated: {optimal_batch_size}")
            
            # Initialize mask generator with optimal settings
            self._initialize_mask_generator()
            
            return optimal_batch_size
            
        except Exception as e:
            print(f"GPU {self.gpu_id}: Error in VRAM estimation: {e}")
            self.current_points_per_batch = self._get_conservative_batch_size()
            self.initial_points_per_batch = self.current_points_per_batch
            self.max_points_per_batch = max(self.max_points_per_batch, self.current_points_per_batch * 2)
            self._initialize_mask_generator()
            return self.current_points_per_batch
    
    def _get_conservative_batch_size(self):
        """Get conservative points_per_batch based on GPU memory and model type"""
        try:
            # Get GPU memory info
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"GPU {self.gpu_id}: Detected {gpu_memory_gb:.1f}GB VRAM")
                
                # Conservative starting values based on model type and GPU memory
                if self.model_type == "vit_h":
                    if gpu_memory_gb >= 24:
                        return 128  # High-end GPU
                    elif gpu_memory_gb >= 16:
                        return 96   # Mid-range GPU
                    elif gpu_memory_gb >= 12:
                        return 64   # Entry-level GPU
                    else:
                        return 32   # Low memory GPU
                elif self.model_type == "vit_l":
                    if gpu_memory_gb >= 16:
                        return 128
                    elif gpu_memory_gb >= 12:
                        return 96
                    else:
                        return 64
                else:  # vit_b
                    if gpu_memory_gb >= 12:
                        return 128
                    else:
                        return 96
            else:
                return 64  # Fallback for CPU
        except:
            return 64  # Safe fallback
    
    def _initialize_mask_generator(self):
        """Initialize or reinitialize the mask generator with current batch size"""
        if self.mask_generator is not None: # Clean up old generator if exists
            del self.mask_generator
            torch.cuda.empty_cache()
            
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            points_per_batch=self.current_points_per_batch  # Key adaptive parameter
        )
        print(f"GPU {self.gpu_id}: Mask generator initialized with points_per_batch={self.current_points_per_batch}")
    
    def _handle_oom_error(self):
        """Handle CUDA OOM by reducing points_per_batch and reinitializing"""
        self.oom_count += 1
        old_batch_size = self.current_points_per_batch
        
        # Reduce batch size (more aggressive reduction for repeated OOMs)
        if self.oom_count == 1:
            self.current_points_per_batch = max(self.min_points_per_batch, 
                                              int(self.current_points_per_batch * 0.7))
        elif self.oom_count == 2:
            self.current_points_per_batch = max(self.min_points_per_batch, 
                                              int(self.current_points_per_batch * 0.5))
        else:
            self.current_points_per_batch = self.min_points_per_batch
        
        print(f"GPU {self.gpu_id}: CUDA OOM detected (#{self.oom_count}). "
              f"Reducing points_per_batch: {old_batch_size} -> {self.current_points_per_batch}")
        
        # Clear GPU memory
        torch.cuda.empty_cache()
        
        # Reinitialize mask generator
        self._initialize_mask_generator()
        
        return self.current_points_per_batch >= self.min_points_per_batch
    
    def _try_increase_batch_size(self):
        """Occasionally try to increase batch size if we've been successful"""
        if (self.successful_batches > 0 and 
            self.successful_batches % 20 == 0 and  # Every 20 successful images
            self.current_points_per_batch < self.max_points_per_batch and
            self.oom_count == 0):  # Only if no recent OOMs
            
            old_batch_size = self.current_points_per_batch
            # More conservative increase: +10% or +min_step, capped by max_points_per_batch
            increase_step = max(1, int(self.current_points_per_batch * 0.1)) 
            self.current_points_per_batch = min(self.max_points_per_batch, self.current_points_per_batch + increase_step)
            
            if old_batch_size != self.current_points_per_batch:
                print(f"GPU {self.gpu_id}: Attempting to increase points_per_batch: {old_batch_size} -> {self.current_points_per_batch}")
                # Reinitialize mask generator
                self._initialize_mask_generator()
    
    def _choose_optimal_image_size(self, prepared_data):
        """Choose optimal image size based on current GPU state and available variants"""
        if not prepared_data.get('has_variants', False):
            # Single image - return as is
            return prepared_data['image_data'], prepared_data['actual_size']
        
        # Multiple variants available - choose based on current points_per_batch
        variants = prepared_data['variants']
        
        # Choose size based on current GPU memory pressure (higher points_per_batch means less memory per image)
        if self.current_points_per_batch >= 128:
            target_size = 1536  # Use smallest for high batch sizes
        elif self.current_points_per_batch >= 64:
            target_size = 2048  # Medium
        else:
            target_size = 2560  # Largest for low batch sizes
        
        # Fallback to closest available size if exact size not available
        if target_size not in variants:
            available_sizes = sorted(variants.keys())
            target_size = min(available_sizes, key=lambda x: abs(x - target_size))
        
        variant = variants[target_size]
        return variant['image_data'], variant['actual_size']
    
    def _load_stable_model(self, checkpoint_path: str):
        """Load SAM model with stable optimizations"""
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            script_dir = os.path.dirname(os.path.abspath(__file__)) # Get script directory
            checkpoint_paths = {
                "vit_h": os.path.join(script_dir, "sam_vit_h_4b8939.pth"),
                "vit_l": os.path.join(script_dir, "sam_vit_l_0b3195.pth"),
                "vit_b": os.path.join(script_dir, "sam_vit_b_01ec64.pth")
            }
            default_checkpoint = checkpoint_paths.get(self.model_type)
            if default_checkpoint and os.path.exists(default_checkpoint):
                checkpoint_path = default_checkpoint
                print(f"GPU {self.gpu_id}: Using default checkpoint: {checkpoint_path}")
            else:
                # Fallback to current working directory if script_dir one not found
                cwd_checkpoint_name = os.path.basename(default_checkpoint) if default_checkpoint else f"sam_{self.model_type}.pth"
                cwd_checkpoint = os.path.join(os.getcwd(), cwd_checkpoint_name)
                if os.path.exists(cwd_checkpoint):
                    checkpoint_path = cwd_checkpoint
                    print(f"GPU {self.gpu_id}: Using checkpoint from CWD: {checkpoint_path}")
                else:
                    error_msg = f"Checkpoint not found for model type {self.model_type}."
                    if default_checkpoint:
                         error_msg += f" Tried: {default_checkpoint} and {cwd_checkpoint}"
                    else:
                         error_msg += f" Tried CWD: {cwd_checkpoint}"
                    raise FileNotFoundError(error_msg)

        
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        sam.eval()
        
        if self.optimize_memory and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print(f"GPU {self.gpu_id}: ✓ Flash attention enabled")
            except Exception as e:
                print(f"GPU {self.gpu_id}: Flash attention not available or failed to enable: {e}")
        
        return sam
    
    def _setup_stable_optimizations(self):
        """Setup stable GPU optimizations"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False # Usually False for speed, True for reproducibility
            torch.cuda.empty_cache()
            
            # Limiting memory fraction can sometimes be counterproductive with dynamic batching.
            # Consider removing or making it optional if issues arise.
            # if self.optimize_memory:
            #     try:
            #         torch.cuda.set_per_process_memory_fraction(0.9) # Allow slightly more
            #         print(f"✓ GPU {self.gpu_id}: Memory limited to 90%")
            #     except Exception as e:
            #         print(f"GPU {self.gpu_id}: Could not set memory fraction: {e}")
            #         pass
    
    def process_optimized_image(self, prepared_data: dict, max_retries: int = 3):
        """Process optimized image data with smart variant selection"""
        image_name = prepared_data['image_name']
        
        for attempt in range(max_retries + 1):
            try:
                # Choose optimal image size based on current GPU state
                image, actual_size = self._choose_optimal_image_size(prepared_data)
                
                start_time = time.time()
                
                # Clear cache before processing
                if self.optimize_memory:
                    torch.cuda.empty_cache()
                
                # Process with current settings (image is already optimally preprocessed!)
                masks = self.mask_generator.generate(image)
                
                # Success! Update counters
                self.successful_batches += 1
                self.oom_count = 0  # Reset OOM counter on success
                
                # Occasionally try to increase batch size
                self._try_increase_batch_size()
                
                # Clean up GPU memory
                if self.optimize_memory:
                    torch.cuda.empty_cache()
                
                process_time = time.time() - start_time
                
                # Convert annotations
                annotations = self.masks_to_annotations_fast(masks, image.shape)
                
                # Save JSON
                json_path = None
                if self.sam_output_dir:
                    json_path = self._save_annotations_robust(prepared_data['image_path'], annotations)
                
                return {
                    'path': image_name,
                    'size': (image.shape[1], image.shape[0]),
                    'masks': len(masks),
                    'time': process_time,
                    'fps': 1.0 / process_time if process_time > 0 else 0,
                    'json_path': json_path,
                    'success': True,
                    'gpu_id': self.gpu_id,
                    'points_per_batch': self.current_points_per_batch,
                    'attempt': attempt + 1,
                    'actual_size': actual_size,
                    'has_variants': prepared_data.get('has_variants', False)
                }
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"GPU {self.gpu_id}: CUDA OOM on attempt {attempt + 1} for {image_name}")
                
                if attempt < max_retries:
                    # Try to handle OOM and retry
                    if self._handle_oom_error():
                        print(f"GPU {self.gpu_id}: Retrying with points_per_batch={self.current_points_per_batch}")
                        continue
                    else:
                        print(f"GPU {self.gpu_id}: Cannot reduce batch size further for {image_name}")
                        break
                else:
                    print(f"GPU {self.gpu_id}: Max retries reached for {image_name}")
                    break
                    
            except Exception as e:
                print(f"GPU {self.gpu_id}: Error processing {image_name}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # If we get here, all attempts failed
        return {
            'path': image_name,
            'error': f'Failed after {max_retries + 1} attempts (OOM or other error)',
            'success': False,
            'gpu_id': self.gpu_id,
            'final_points_per_batch': self.current_points_per_batch
        }
    
    def load_image_efficiently(self, path):
        """Load image with error handling and optimization"""
        try:
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Dynamic resizing based on current batch size
            h, w = image.shape[:2]
            
            # Adjust max size based on points_per_batch (lower batch size = can handle larger images)
            if self.current_points_per_batch >= 128:
                max_size = 1536
            elif self.current_points_per_batch >= 64:
                max_size = 2048
            else:
                max_size = 2560  # Smaller batch size can handle larger images
            
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            return image, True
        except Exception as e:
            print(f"GPU {self.gpu_id}: Failed to load {path}: {e}")
            return None, False
    
    def process_single_image(self, image_path: str, max_retries: int = 3):
        """Process single image with adaptive OOM handling"""
        for attempt in range(max_retries + 1):
            try:
                # Load image
                image, success = self.load_image_efficiently(image_path)
                if not success:
                    return {'path': os.path.basename(image_path), 'success': False, 
                           'error': 'Failed to load', 'gpu_id': self.gpu_id}
                
                start_time = time.time()
                
                # Clear cache before processing
                if self.optimize_memory:
                    torch.cuda.empty_cache()
                
                # Process with current settings
                masks = self.mask_generator.generate(image)
                
                # Success! Update counters
                self.successful_batches += 1
                self.oom_count = 0  # Reset OOM counter on success
                
                # Occasionally try to increase batch size
                self._try_increase_batch_size()
                
                # Clean up GPU memory
                if self.optimize_memory:
                    torch.cuda.empty_cache()
                
                process_time = time.time() - start_time
                
                # Convert annotations
                annotations = self.masks_to_annotations_fast(masks, image.shape)
                
                # Save JSON
                json_path = None
                if self.sam_output_dir:
                    json_path = self._save_annotations_robust(image_path, annotations)
                
                return {
                    'path': os.path.basename(image_path),
                    'size': (image.shape[1], image.shape[0]),
                    'masks': len(masks),
                    'time': process_time,
                    'fps': 1.0 / process_time if process_time > 0 else 0,
                    'json_path': json_path,
                    'success': True,
                    'gpu_id': self.gpu_id,
                    'points_per_batch': self.current_points_per_batch,
                    'attempt': attempt + 1
                }
                
            except torch.cuda.OutOfMemoryError as e:
                print(f"GPU {self.gpu_id}: CUDA OOM on attempt {attempt + 1} for {os.path.basename(image_path)}")
                
                if attempt < max_retries:
                    # Try to handle OOM and retry
                    if self._handle_oom_error():
                        print(f"GPU {self.gpu_id}: Retrying with points_per_batch={self.current_points_per_batch}")
                        continue
                    else:
                        print(f"GPU {self.gpu_id}: Cannot reduce batch size further for {os.path.basename(image_path)}")
                        break
                else:
                    print(f"GPU {self.gpu_id}: Max retries reached for {os.path.basename(image_path)}")
                    break
                    
            except Exception as e:
                print(f"GPU {self.gpu_id}: Error processing {os.path.basename(image_path)}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # If we get here, all attempts failed
        return {
            'path': os.path.basename(image_path),
            'error': f'Failed after {max_retries + 1} attempts (OOM or other error)',
            'success': False,
            'gpu_id': self.gpu_id,
            'final_points_per_batch': self.current_points_per_batch
        }
    
    def masks_to_annotations_fast(self, masks, image_shape):
        """Fast and stable mask to annotation conversion with robust RLE"""
        annotations = []
        h, w = image_shape[:2]
        
        for i, mask_data in enumerate(masks):
            try:
                segmentation_mask = mask_data['segmentation']
                bbox = mask_data['bbox']
                area = mask_data['area']
                
                if HAS_PYCOCOTOOLS:
                    mask_uint8 = segmentation_mask.astype(np.uint8)
                    mask_fortran = np.asfortranarray(mask_uint8)
                    rle = mask_util.encode(mask_fortran)
                    if isinstance(rle['counts'], bytes):
                        counts = rle['counts'].decode('utf-8')
                    else:
                        # Ensure counts is a list of integers for JSON serializability
                        counts = [int(c) for c in rle['counts']] if isinstance(rle['counts'], list) else rle['counts']
                else:
                    counts = self._robust_numpy_rle_encode(segmentation_mask)
                
                annotation = {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "segmentation": {
                        "size": [int(h), int(w)], 
                        "counts": counts
                    },
                    "area": int(area),
                    "predicted_iou": float(mask_data.get('predicted_iou', 0.0)),
                    "stability_score": float(mask_data.get('stability_score', 0.0))
                }
                
                annotations.append(annotation)
                
            except Exception as e:
                print(f"GPU {self.gpu_id}: Warning: Failed to process mask {i}: {e}")
                continue
        
        return annotations
    
    def _robust_numpy_rle_encode(self, mask):
        """Robust RLE encoding that produces valid JSON-serializable output"""
        try:
            mask = mask.astype(np.uint8)
            mask_flat = mask.flatten()
            diff_indices = np.where(np.diff(np.concatenate(([0], mask_flat, [0]))))[0]
            run_lengths = np.diff(diff_indices)
            
            rle_counts = []
            if len(diff_indices) > 0:
                start_val = mask_flat[0] if len(mask_flat) > 0 else 0
                if start_val == 1:
                    rle_counts.append(0) # Add leading zero if mask starts with 1
                for length in run_lengths:
                    rle_counts.append(int(length))
            else: # Handle cases of all zeros or all ones
                if len(mask_flat) > 0 and mask_flat[0] == 1: # All ones
                    rle_counts = [0, len(mask_flat)]
                else: # All zeros
                    rle_counts = [len(mask_flat)] # Or [] if COCO expects empty for no ones
            
            return rle_counts
            
        except Exception as e:
            print(f"GPU {self.gpu_id}: Error in RLE encoding: {e}")
            return [mask.size] # Fallback: RLE for all zeros
    
    def _save_annotations_robust(self, image_path, annotations):
        """Robust JSON saving with validation"""
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(self.sam_output_dir, f"{image_name}.json")
            
            output_data = {
                "image_name": image_name,
                "annotations": annotations,
                "num_masks": len(annotations),
                "processed_by_gpu": self.gpu_id,
                "points_per_batch_used": self.current_points_per_batch
            }
            
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            try:
                with open(json_path, 'r') as f:
                    json.load(f) # Validate JSON
                return json_path
            except json.JSONDecodeError as e:
                print(f"✗ GPU {self.gpu_id}: Invalid JSON written for {image_name}: {e}")
                os.remove(json_path) # Remove invalid JSON
                return None
            
        except Exception as e:
            print(f"GPU {self.gpu_id}: Error saving JSON for {os.path.basename(image_path)}: {e}")
            return None


def optimized_image_loader_worker(image_paths, prepared_queue, max_buffer_size=10, 
                                 num_preprocessor_threads=4):
    """
    OPTIMIZED: Parallel image loader using ThreadPoolExecutor for preprocessing
    """
    import sys
    print(f"Optimized Image Loader: STARTING with {num_preprocessor_threads} parallel threads", flush=True)
    print(f"Processing {len(image_paths)} images (buffer size: {max_buffer_size})", flush=True)
    sys.stdout.flush()
    
    preprocessor = OptimizedImagePreprocessor()
    loaded_count = 0
    variant_count = 0  # Track images with multiple variants
    
    def process_single_image_for_pool(image_path):
        """Process single image with error handling for ThreadPoolExecutor"""
        try:
            return preprocessor.preprocess_single_image(image_path)
        except Exception as e:
            print(f"Error in preprocessor thread for {os.path.basename(image_path)}: {e}")
            return None # Ensure it returns None on failure
    
    try:
        # Use ThreadPoolExecutor for parallel preprocessing
        with ThreadPoolExecutor(max_workers=num_preprocessor_threads) as executor:
            # Submit all preprocessing jobs
            future_to_path = {
                executor.submit(process_single_image_for_pool, path): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in future_to_path: # This iterates in completion order for Python 3.8+
                try:
                    prepared_data = future.result() # This will raise exception if future had one
                    if prepared_data is not None:
                        # Track variant statistics
                        if prepared_data.get('has_variants', False):
                            variant_count += 1
                        
                        # Put in queue (blocks if queue is full - provides backpressure)
                        prepared_queue.put(prepared_data) # This can block
                        loaded_count += 1
                        
                            
                except Exception as e: # Catch exceptions from future.result()
                    image_path = future_to_path[future]
                    print(f"Failed to process {os.path.basename(image_path)} during pre-loading: {e}", flush=True)
                    sys.stdout.flush()
                    continue
        
        print(f"Optimized Image Loader: FINISHED", flush=True)
        print(f"Successfully pre-processed and queued: {loaded_count} images", flush=True)
        print(f"Images with multi-scale variants: {variant_count}", flush=True)
        print(f"Memory-optimized single-scale: {loaded_count - variant_count}", flush=True)
        sys.stdout.flush()
        return loaded_count
        
    except Exception as e:
        print(f"Optimized Image Loader: CRITICAL ERROR: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Signal end of data to consumers if using a sentinel
        # For JoinableQueue, task_done() is handled by consumers
        pass


def optimized_gpu_worker(gpu_id, prepared_queue, results_queue, progress_queue, model_type, checkpoint_path, 
                        optimize_memory, output_dir, dataset_name, total_images, first_image_sample=None):
    """
    OPTIMIZED: GPU worker that uses optimized preprocessing data
    """
    import sys
    processor = None # Initialize to ensure it's defined in finally block
    try:
        # Initialize adaptive processor
        processor = AdaptiveSAMProcessor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id,
            optimize_memory=optimize_memory,
            output_dir=output_dir # This is dataset_output_dir
        )
        
        print(f"GPU {gpu_id}: Ready for optimized processing of dataset '{dataset_name}'", flush=True)
        sys.stdout.flush()
        
        # Enhanced warmup with VRAM-based estimation
        print(f"GPU {gpu_id}: Starting enhanced warmup with VRAM-based points_per_batch estimation...", flush=True)
        sys.stdout.flush()
        
        warmup_done = False
        warmup_attempts = 0
        max_warmup_attempts = 10 # Increased attempts for robustness
        
        while not warmup_done and warmup_attempts < max_warmup_attempts:
            try:
                warmup_attempts += 1
                print(f"GPU {gpu_id}: Warmup attempt {warmup_attempts}/{max_warmup_attempts}", flush=True)
                sys.stdout.flush()
                
                # Try to get a sample image for VRAM estimation
                test_image_data_for_vram_estimation = None
                if first_image_sample is not None:
                    test_image_data_for_vram_estimation = first_image_sample
                else:
                    try:
                        # Non-blocking get for warmup sample from queue
                        warmup_data_from_queue = prepared_queue.get(block=False) 
                        if warmup_data_from_queue.get('has_variants', False):
                            test_variant = list(warmup_data_from_queue['variants'].values())[0]
                            test_image_data_for_vram_estimation = test_variant['image_data']
                        else:
                            test_image_data_for_vram_estimation = warmup_data_from_queue['image_data']
                        # IMPORTANT: Put it back if it was taken for estimation only
                        prepared_queue.put(warmup_data_from_queue) 
                    except queue.Empty: # queue.Empty if block=False and queue is empty
                        pass # No sample available from queue, proceed without it for VRAM estimation
                
                # Estimate optimal points_per_batch using VRAM measurement
                optimal_batch_size = processor.estimate_optimal_points_per_batch(test_image_data=test_image_data_for_vram_estimation)
                
                print(f"GPU {gpu_id}: VRAM-based estimation completed. Optimal points_per_batch: {optimal_batch_size}", flush=True)
                sys.stdout.flush()
                
                # Test with a real image if available (can be different from VRAM estimation image)
                # This is more like a "dry run"
                try:
                    # Try to get an actual item from the queue for a dry run
                    # This ensures the processor is fully warmed up with actual data flow
                    dry_run_data = prepared_queue.get(timeout=5) # Wait a bit for an item
                    
                    # Use a copy for dry run if it has variants to avoid modifying queue item
                    if dry_run_data.get('has_variants', False):
                        test_variant_dry_run = list(dry_run_data['variants'].values())[0]
                        dry_run_image_data = test_variant_dry_run['image_data']
                    else:
                        dry_run_image_data = dry_run_data['image_data']

                    dry_run_test_data = {
                        'image_data': dry_run_image_data, 
                        'image_name': 'warmup_dry_run', 
                        'image_path': 'warmup_dry_run_path', # Dummy path
                        'has_variants': False, # Simplified for dry run
                        'actual_size': dry_run_image_data.shape[:2],
                        'original_shape': dry_run_image_data.shape # Add original_shape for consistency
                    }
                    warmup_result = processor.process_optimized_image(dry_run_test_data)
                    
                    # Put the actual item back into the queue for processing
                    prepared_queue.put(dry_run_data)

                    if warmup_result.get('success'):
                        print(f"GPU {gpu_id}: Warmup dry run successful: {warmup_result['masks']} masks in {warmup_result['time']:.2f}s "
                              f"(final points_per_batch: {warmup_result.get('points_per_batch', 'unknown')})", flush=True)
                        warmup_done = True
                    else:
                        print(f"GPU {gpu_id}: Warmup dry run failed, but proceeding with VRAM estimation values.", flush=True)
                        warmup_done = True # Still mark as done to avoid loop
                except queue.Empty:
                     print(f"GPU {gpu_id}: No image available from queue for dry run, proceeding with VRAM estimation", flush=True)
                     warmup_done = True

                sys.stdout.flush()
                
            except Exception as e:
                print(f"GPU {gpu_id}: Warmup attempt {warmup_attempts} failed: {e}", flush=True)
                if warmup_attempts >= max_warmup_attempts:
                    print(f"GPU {gpu_id}: All warmup attempts failed, proceeding without full warmup (VRAM estimation might be less accurate)", flush=True)
                    # Fallback to conservative if estimation failed badly
                    if processor.mask_generator is None:
                         processor.current_points_per_batch = processor._get_conservative_batch_size()
                         processor.initial_points_per_batch = processor.current_points_per_batch
                         processor._initialize_mask_generator()
                    warmup_done = True
                sys.stdout.flush()
        
        processed_count = 0
        consecutive_timeouts = 0
        # Max timeouts before worker exits: 5 timeouts * 10s each = 50s of inactivity
        # This might need adjustment based on how long the loader takes to fill the queue initially.
        max_consecutive_timeouts = 10 # Increased for longer tolerance
        
        variant_used_count = 0
        
        # Process prepared images from queue until empty or too many timeouts
        while True: # Loop will be broken by timeouts or sentinel
            try:
                # Get prepared image from queue with timeout
                prepared_data = prepared_queue.get(timeout=10)  # Increased timeout
                
                if prepared_data is None: # Sentinel value indicating no more data
                    print(f"GPU {gpu_id}: Received sentinel. Shutting down.", flush=True)
                    prepared_queue.task_done() # Acknowledge sentinel
                    break 
                    
                consecutive_timeouts = 0  # Reset timeout counter
                
                try:
                    # Use optimized processing method
                    result = processor.process_optimized_image(prepared_data)
                    results_queue.put(result) # This should be a manager.Queue, non-blocking
                    processed_count += 1
                    
                    # Track variant usage
                    if result.get('has_variants', False):
                        variant_used_count += 1
                    
                    # Send progress update with queue size information
                    try:
                        # qsize() can be unreliable with mp.Queue
                        current_q_size = prepared_queue.qsize() if hasattr(prepared_queue, 'qsize') else -1 
                    except NotImplementedError: # Some queue types (e.g., JoinableQueue on macOS) don't implement qsize
                        current_q_size = -1 # Indicate unknown size
                    
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count, # This is count for this GPU
                        'image_name': prepared_data['image_name'],
                        'success': result.get('success', False),
                        'fps': result.get('fps', 0),
                        'masks': result.get('masks', 0),
                        'batch_size': result.get('points_per_batch', 0),
                        'queue_size': current_q_size,
                        'has_variants': result.get('has_variants', False)
                    })
                    
                    # Mark task as done
                    prepared_queue.task_done()
                    
                except Exception as e:
                    print(f"GPU {gpu_id}: Error processing {prepared_data['image_name']}: {e}", flush=True)
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
                    # Put error result
                    results_queue.put({
                        'path': prepared_data['image_name'],
                        'error': str(e),
                        'success': False,
                        'gpu_id': gpu_id
                    })
                    
                    # Send progress update for failed image
                    try:
                        current_q_size_fail = prepared_queue.qsize() if hasattr(prepared_queue, 'qsize') else -1
                    except NotImplementedError:
                        current_q_size_fail = -1
                        
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count, # Count for this GPU
                        'image_name': prepared_data['image_name'],
                        'success': False,
                        'error': str(e),
                        'queue_size': current_q_size_fail
                    })
                    
                    prepared_queue.task_done() # Still mark as done
                    
            except queue.Empty:  # queue.Empty from prepared_queue.get(timeout=10)
                consecutive_timeouts += 1
                print(f"GPU {gpu_id}: Timeout {consecutive_timeouts}/{max_consecutive_timeouts} waiting for images.", flush=True)
                if consecutive_timeouts >= max_consecutive_timeouts:
                    print(f"GPU {gpu_id}: Max timeouts reached. Assuming no more images. Shutting down.", flush=True)
                    break # Exit loop after max timeouts
                sys.stdout.flush()
            except Exception as e_outer: # Catch other unexpected errors in the loop
                print(f"GPU {gpu_id}: Outer loop critical error: {e_outer}", flush=True)
                import traceback
                traceback.print_exc()
                break # Exit loop on critical error
        
        print(f"GPU {gpu_id}: Exited processing loop. Processed {processed_count} images.", flush=True)
        print(f"GPU {gpu_id}: Used multi-scale variants for {variant_used_count} images", flush=True)
        sys.stdout.flush()
        
        # Print final stats for this GPU if processor was initialized
        if processor:
            final_batch_size = processor.current_points_per_batch
            initial_batch_size = processor.initial_points_per_batch
            oom_count = processor.oom_count
            
            print(f"GPU {gpu_id}: Completed optimized processing", flush=True)
            print(f"GPU {gpu_id}: Images processed by this GPU: {processed_count}", flush=True)
            print(f"GPU {gpu_id}: Final points_per_batch: {final_batch_size} (started with {initial_batch_size})", flush=True)
            if oom_count > 0:
                print(f"GPU {gpu_id}: Handled {oom_count} CUDA OOM errors during processing", flush=True)
        sys.stdout.flush()
        
        return processed_count
        
    except Exception as e:
        print(f"GPU {gpu_id}: Worker failed with error: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Ensure GPU resources are released if SAM model was loaded
        if processor and hasattr(processor, 'sam'):
            del processor.sam # type: ignore
            del processor
        torch.cuda.empty_cache()
        print(f"GPU {gpu_id}: Worker finalized and cleaned up.", flush=True)


def simple_gpu_worker(gpu_id, work_queue, results_queue, progress_queue, model_type, checkpoint_path, 
                     optimize_memory, output_dir, dataset_name, total_images, first_image_path=None):
    """
    Simple worker function that processes images from a work queue (on-demand loading)
    """
    import sys
    processor = None
    try:
        # Initialize adaptive processor
        processor = AdaptiveSAMProcessor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id,
            optimize_memory=optimize_memory,
            output_dir=output_dir # This is dataset_output_dir
        )
        
        print(f"GPU {gpu_id}: Ready for simple processing of dataset '{dataset_name}'", flush=True)
        sys.stdout.flush()
        
        # Enhanced warmup with VRAM-based estimation
        if first_image_path and os.path.exists(first_image_path):
            print(f"GPU {gpu_id}: Starting VRAM-based points_per_batch estimation with test image: {first_image_path}", flush=True)
            optimal_batch_size = processor.estimate_optimal_points_per_batch(test_image_path=first_image_path)
            print(f"GPU {gpu_id}: Optimal points_per_batch: {optimal_batch_size}", flush=True)
        else:
            print(f"GPU {gpu_id}: No test image available or path invalid, using conservative estimation for points_per_batch.", flush=True)
            # Initialize with conservative and then call generator init
            processor.current_points_per_batch = processor._get_conservative_batch_size()
            processor.initial_points_per_batch = processor.current_points_per_batch
            processor.max_points_per_batch = max(processor.max_points_per_batch, processor.current_points_per_batch * 2)
            processor._initialize_mask_generator() # Ensure generator is initialized
        
        processed_count = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 10 # Increased for longer tolerance

        # Process images from queue until empty or too many timeouts
        while True:
            try:
                # Get image path from queue with timeout
                image_path = work_queue.get(timeout=10)  # Increased timeout
                if image_path is None: # Sentinel value
                    print(f"GPU {gpu_id}: Received sentinel. Shutting down (simple_gpu_worker).", flush=True)
                    work_queue.task_done()
                    break

                consecutive_timeouts = 0  # Reset timeout counter
                
                try:
                    result = processor.process_single_image(image_path)
                    results_queue.put(result)
                    processed_count += 1
                    
                    # Send progress update
                    try:
                        current_q_size = work_queue.qsize() if hasattr(work_queue, 'qsize') else -1
                    except NotImplementedError:
                        current_q_size = -1
                    
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count,
                        'image_name': os.path.basename(image_path),
                        'success': result.get('success', False),
                        'fps': result.get('fps', 0),
                        'masks': result.get('masks', 0),
                        'batch_size': result.get('points_per_batch', 0),
                        'queue_size': current_q_size
                    })
                    
                    # Mark task as done
                    work_queue.task_done()
                    
                except Exception as e:
                    print(f"GPU {gpu_id}: Error processing {os.path.basename(image_path)}: {e}", flush=True)
                    sys.stdout.flush()
                    import traceback
                    traceback.print_exc()
                    # Put error result
                    results_queue.put({
                        'path': os.path.basename(image_path),
                        'error': str(e),
                        'success': False,
                        'gpu_id': gpu_id
                    })
                    
                    # Send progress update for failed image
                    try:
                        current_q_size_fail = work_queue.qsize() if hasattr(work_queue, 'qsize') else -1
                    except NotImplementedError:
                        current_q_size_fail = -1
                        
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count, # This is count for this GPU
                        'image_name': os.path.basename(image_path),
                        'success': False,
                        'error': str(e),
                        'queue_size': current_q_size_fail
                    })
                    
                    work_queue.task_done() # Still mark as done
                    
            except queue.Empty:  # queue.Empty from work_queue.get(timeout=10)
                consecutive_timeouts += 1
                print(f"GPU {gpu_id}: Timeout {consecutive_timeouts}/{max_consecutive_timeouts} waiting for images (simple worker).", flush=True)
                if consecutive_timeouts >= max_consecutive_timeouts:
                    print(f"GPU {gpu_id}: Max timeouts reached. Assuming no more images (simple worker). Shutting down.", flush=True)
                    break # Exit loop after max timeouts
                sys.stdout.flush()
            except Exception as e_outer: # Catch other unexpected errors in the loop
                print(f"GPU {gpu_id}: Outer loop critical error (simple worker): {e_outer}", flush=True)
                import traceback
                traceback.print_exc()
                break # Exit loop on critical error
        
        print(f"GPU {gpu_id}: Exited processing loop (simple worker). Processed {processed_count} images.", flush=True)
        sys.stdout.flush()
        
        # Print final stats for this GPU
        if processor:
            final_batch_size = processor.current_points_per_batch
            initial_batch_size = processor.initial_points_per_batch
            oom_count = processor.oom_count
            
            print(f"GPU {gpu_id}: Completed simple processing", flush=True)
            print(f"GPU {gpu_id}: Images processed by this GPU: {processed_count}", flush=True)
            print(f"GPU {gpu_id}: Final points_per_batch: {final_batch_size} (started with {initial_batch_size})", flush=True)
            if oom_count > 0:
                print(f"GPU {gpu_id}: Handled {oom_count} CUDA OOM errors during processing", flush=True)
        sys.stdout.flush()
        
        return processed_count
        
    except Exception as e:
        print(f"GPU {gpu_id}: Worker (simple) failed with error: {e}", flush=True)
        sys.stdout.flush()
        import traceback
        traceback.print_exc()
        return 0
    finally:
        if processor and hasattr(processor, 'sam'):
            del processor.sam # type: ignore
            del processor
        torch.cuda.empty_cache()
        print(f"GPU {gpu_id}: Worker (simple) finalized and cleaned up.", flush=True)


def get_image_paths(input_path: str):
    """Get image paths efficiently from a given directory, case-insensitive."""
    if not os.path.isdir(input_path):
        # This should be caught earlier, but good to have a check
        print(f"Warning: Input path is not a directory: {input_path}")
        return [] 
    
    valid_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
    
    # Using Pathlib for potentially better cross-platform compatibility and cleaner syntax
    p = Path(input_path)
    image_paths = [
        str(f) for f in p.iterdir() 
        if f.is_file() and f.suffix.lower().lstrip('.') in valid_extensions
    ]
    
    return sorted(image_paths)


def check_dataset_completion(dataset_output_dir, dataset_name):
    """Check if dataset is already completed (has .done file)"""
    done_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.done")
    return os.path.exists(done_file_path)


def mark_dataset_completed(dataset_output_dir, dataset_name, stats):
    """Mark dataset as completed by creating .done file with stats"""
    try:
        os.makedirs(dataset_output_dir, exist_ok=True) # Ensure dataset_output_dir exists
        done_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.done")
        
        completion_info = {
            "dataset_name": dataset_name,
            "completion_time": time.time(),
            "completion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images_processed": stats.get('total_processed', 0),
            "total_images_failed": stats.get('total_failed', 0),
            "processing_mode": stats.get('processing_mode', 'unknown'),
            "total_masks_generated": stats.get('total_masks', 0),
            "total_processing_time_seconds": stats.get('total_time', 0),
            "average_fps": stats.get('average_fps', 0)
        }
        
        with open(done_file_path, 'w') as f:
            json.dump(completion_info, f, indent=2)
        
        print(f"✓ Dataset '{dataset_name}' marked as completed: {done_file_path}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to mark dataset '{dataset_name}' as completed: {e}")
        return False


def check_already_processed(image_paths, sam_output_dir):
    """Check which images have already been processed and return list of remaining images"""
    if not os.path.exists(sam_output_dir):
        os.makedirs(sam_output_dir, exist_ok=True) # Create if it doesn't exist
        return image_paths, []
    
    remaining_images = []
    already_processed = []
    
    for image_path in image_paths:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(sam_output_dir, f"{image_name}.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Basic check for valid structure
                    if 'annotations' in data and 'num_masks' in data: 
                        already_processed.append(image_path)
                        continue
            except (json.JSONDecodeError, IOError):
                # If JSON is corrupt or unreadable, treat as not processed
                print(f"Warning: Corrupt or unreadable JSON found, will reprocess: {json_path}")
                pass # Fall through to add to remaining_images
        
        remaining_images.append(image_path)
    
    return remaining_images, already_processed


def print_optimized_gpu_summary(all_results, overall_time, output_dir, dataset_name, num_gpus_active):
    """Print comprehensive summary for optimized GPU processing (single or dual)"""
    successful_results = [r for r in all_results if r.get('success', False)]
    failed_results = [r for r in all_results if not r.get('success', False)]
    
    total_masks = sum(r['masks'] for r in successful_results if 'masks' in r)
    successful_json_saves = sum(1 for r in successful_results if r.get('json_path'))
    
    # Count optimization statistics (relevant for optimized mode)
    variant_results = [r for r in successful_results if r.get('has_variants', False)]
    
    print("\n" + "="*70)
    print(f"OPTIMIZED GPU SAM PROCESSING SUMMARY")
    print(f"DATASET: {dataset_name}")
    print("="*70)
    
    processing_mode_desc = "Optimized Single GPU" if num_gpus_active == 1 else "Optimized Dual GPU with Advanced Pre-processing"
    print(f"Processing mode: {processing_mode_desc}")
    print(f"Images processed successfully: {len(successful_results)}")
    print(f"Images failed: {len(failed_results)}")
    print(f"Total images attempted: {len(all_results)}")
    print(f"Successful JSON saves: {successful_json_saves}/{len(successful_results)}") # out of successful
    print(f"Total time: {overall_time:.2f}s")
    if overall_time > 0 and len(all_results) > 0:
        print(f"Overall throughput (attempted images): {len(all_results)/overall_time:.2f} images/sec")
    if overall_time > 0 and len(successful_results) > 0:
        print(f"Overall throughput (successful images): {len(successful_results)/overall_time:.2f} images/sec")
    print(f"Total masks generated: {total_masks:,}")
    
    # Optimization statistics (if applicable)
    if num_gpus_active > 1 or any(r.get('has_variants') is not None for r in all_results): # Check if variant info exists
        print(f"\nOptimization Statistics:")
        print(f"  Images with multi-scale variants: {len(variant_results)}")
        print(f"  Memory-efficient single-scale: {len(successful_results) - len(variant_results)}")
        print(f"  Parallel preprocessing: {'✓ Enabled (Dual GPU mode)' if num_gpus_active > 1 else 'N/A (Single GPU mode)'}")
        print(f"  Smart size selection: ✓ Enabled (if variants used)")
    
    # GPU-specific stats
    for gpu_id_iter in range(num_gpus_active):
        gpu_results = [r for r in successful_results if r.get('gpu_id') == gpu_id_iter]
        if gpu_results:
            gpu_fps = [r['fps'] for r in gpu_results if 'fps' in r]
            gpu_times = [r['time'] for r in gpu_results if 'time' in r]
            gpu_batches = [r.get('points_per_batch', 0) for r in gpu_results if 'points_per_batch' in r]
            print(f"\nGPU {gpu_id_iter} Stats:")
            print(f"  Images processed by GPU {gpu_id_iter}: {len(gpu_results)}")
            if gpu_fps: print(f"  Average FPS: {np.mean(gpu_fps):.2f}")
            if gpu_times: print(f"  Average time per image: {np.mean(gpu_times):.2f}s")
            if gpu_batches:
                print(f"  Final points_per_batch: {gpu_batches[-1] if gpu_batches else 'N/A'}")
                print(f"  Average points_per_batch: {np.mean(gpu_batches):.1f}")
        else:
            print(f"\nGPU {gpu_id_iter} Stats: No successful images processed.")

    if num_gpus_active > 1: # Dynamic distribution relevant for multi-GPU
        print(f"\nDynamic Work Distribution:")
        for gpu_id_iter in range(num_gpus_active):
            gpu_processed_count = sum(1 for r in all_results if r.get('gpu_id') == gpu_id_iter)
            if len(all_results) > 0:
                percentage = gpu_processed_count / len(all_results) * 100
                print(f"  GPU {gpu_id_iter} attempted: {gpu_processed_count} images ({percentage:.1f}%)")
            else:
                print(f"  GPU {gpu_id_iter} attempted: 0 images")
    
    # Print failed images if any
    if failed_results:
        print(f"\n⚠️  Failed to process {len(failed_results)} images:")
        for result in failed_results[:10]: # Show up to 10 failed
            gpu_info = f"(GPU {result.get('gpu_id', '?')})" if 'gpu_id' in result else ""
            print(f"  - {result['path']}: {result.get('error', 'Unknown error')} {gpu_info}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more")
    
    sam_output_dir = os.path.join(output_dir, "sam") # output_dir is dataset_output_dir
    print(f"\nJSON files saved to: {sam_output_dir}")
    
    # Return stats for dataset completion tracking
    return {
        'total_processed': len(successful_results),
        'total_failed': len(failed_results),
        'total_masks': total_masks,
        'total_time': overall_time,
        'average_fps': (len(successful_results)/overall_time if overall_time > 0 and len(successful_results) > 0 else 0),
        'processing_mode': processing_mode_desc
    }


def progress_monitor(progress_queue, total_images, dataset_name, num_gpus_expected):
    """Enhanced progress monitor with optimization statistics"""
    gpu_stats = {i: {'count': 0, 'last_image': '', 'last_fps': 0, 'last_masks': 0, 'last_batch': 0, 'total_on_gpu': 0}
                 for i in range(num_gpus_expected)}
    
    total_processed_overall = 0 # Sum of 'processed_count' from all GPUs
    updates_received = 0
    
    current_queue_size = "N/A" # Placeholder
    variant_count = 0
    
    # Use a single pbar for overall progress based on total_images
    with tqdm(total=total_images, desc=f"Processing {dataset_name}", unit="img", dynamic_ncols=True) as pbar:
        # Loop as long as not all images are accounted for or a sentinel for progress queue completion
        while total_processed_overall < total_images:
            try:
                # Get progress update with timeout
                update = progress_queue.get(timeout=1) # 1s timeout
                if update is None: # Sentinel to stop progress monitor
                    print("Progress monitor received sentinel. Finalizing.")
                    break
                
                gpu_id = update['gpu_id']
                updates_received += 1
                
                # Update GPU stats
                # 'processed_count' is the cumulative count for that specific GPU
                gpu_stats[gpu_id]['total_on_gpu'] = update['processed_count'] 
                gpu_stats[gpu_id]['last_image'] = update['image_name']
                
                # Update queue size if available
                if 'queue_size' in update and update['queue_size'] != -1:
                    current_queue_size = update['queue_size']
                
                # Track variants
                if update.get('has_variants', False) and update.get('success', False):
                    variant_count += 1
                
                if update.get('success', False):
                    gpu_stats[gpu_id]['last_fps'] = update.get('fps', 0)
                    gpu_stats[gpu_id]['last_masks'] = update.get('masks', 0)
                    gpu_stats[gpu_id]['last_batch'] = update.get('batch_size', 0)
                
                # Update total progress for pbar
                new_total_processed_overall = sum(stats['total_on_gpu'] for stats in gpu_stats.values())
                if new_total_processed_overall > total_processed_overall:
                    pbar.update(new_total_processed_overall - total_processed_overall)
                    total_processed_overall = new_total_processed_overall
                elif updates_received % 10 == 0 : # Force refresh pbar if no new images but updates came
                    pbar.refresh()

                
                # Enhanced progress bar description with optimization info
                status_parts = []
                for i in range(num_gpus_expected):
                    if gpu_stats[i]['total_on_gpu'] > 0 or num_gpus_expected == 1: # Show stats if active or only one GPU
                         status_parts.append(f"G{i}: {gpu_stats[i]['last_fps']:.1f}fps ({gpu_stats[i]['total_on_gpu']})")
                
                if current_queue_size != "N/A":
                    status_parts.append(f"Q:{current_queue_size}")
                
                if variant_count > 0:
                    status_parts.append(f"Var:{variant_count}")
                
                pbar.set_postfix_str(" | ".join(status_parts), refresh=True) # Refresh to show updated postfix
                
            except queue.Empty:  # Timeout - check if we're done or continue
                if total_processed_overall >= total_images:
                    print("Progress monitor: All images processed based on count. Exiting.")
                    break
                # If not all processed, continue waiting for updates
                if pbar.postfix:
                    pbar.set_postfix_str(pbar.postfix + " (idle)", refresh=True) # Indicate idleness
                else:
                    pbar.set_postfix_str("(idle)", refresh=True)
                continue
            except Exception as e:
                print(f"Progress monitor error: {e}")
                import traceback
                traceback.print_exc()
                # Potentially break if error is critical
                break
        
        # Ensure pbar closes at 100% if all images are done
        if total_processed_overall < total_images:
            pbar.update(total_images - total_processed_overall) # Fill to 100%
        pbar.set_postfix_str("Completed", refresh=True)

    print("Progress monitor finished.")
    return gpu_stats


def process_single_dataset(dataset_name, dataset_input_images_dir, dataset_output_dir, 
                          model_type, checkpoint_path, optimize_memory, single_gpu_mode=None, 
                          buffer_size=10, num_preprocessor_threads=4, use_resume=True):
    """Process a single dataset with optimized dual GPU or single GPU mode"""
    
    # Output directory for this specific dataset's SAM files (e.g., .../dataset_output_dir/sam)
    sam_output_dir = os.path.join(dataset_output_dir, "sam")
    os.makedirs(dataset_output_dir, exist_ok=True) # Ensure dataset_output_dir exists 
    os.makedirs(sam_output_dir, exist_ok=True)    # Ensure sam_output_dir exists

    # Check if dataset is already completed (uses dataset_output_dir)
    if use_resume and check_dataset_completion(dataset_output_dir, dataset_name):
        print(f"✓ Dataset '{dataset_name}' already completed (found .done file). Skipping.")
        return True, {} # Return empty stats
    
    print(f"Processing dataset '{dataset_name}':")
    print(f"  Input images from: '{dataset_input_images_dir}'")
    print(f"  JSON output in: '{sam_output_dir}'")
    print(f"  Completion marker in: '{dataset_output_dir}' (using name: {dataset_name}.done)")
    
    image_paths = get_image_paths(dataset_input_images_dir)
    
    if not image_paths:
        print(f"No images found in '{dataset_input_images_dir}'. Skipping this dataset.")
        # Optionally mark as completed with 0 processed if desired, or just skip
        stats = {
            'total_processed': 0, 'total_failed': 0, 'total_masks': 0, 
            'total_time': 0, 'average_fps': 0, 'processing_mode': 'Skipped - No Images'
        }
        mark_dataset_completed(dataset_output_dir, dataset_name, stats)
        return False, stats
    
    remaining_images = image_paths
    if use_resume:
        remaining_images, already_processed = check_already_processed(image_paths, sam_output_dir)
        if already_processed:
            print(f"Resume mode: Found {len(already_processed)} already processed images for '{dataset_name}'")
            print(f"Remaining to process for '{dataset_name}': {len(remaining_images)} images")
    
    if not remaining_images:
        print(f"All {len(image_paths)} images in '{dataset_name}' already processed.")
        stats = {
            'total_processed': len(image_paths), 'total_failed': 0, 
            'total_masks': -1,  # Could calculate from existing JSON files if needed, -1 for unknown
            'total_time': 0, 'average_fps': 0, 'processing_mode': 'Resumed - All Previously Completed'
        }
        mark_dataset_completed(dataset_output_dir, dataset_name, stats)
        return True, stats
    
    num_gpus_to_use = 1 if single_gpu_mode is not None else 2 # Assuming 2 GPUs for dual mode
    print(f"Processing {len(remaining_images)} images for '{dataset_name}' using {num_gpus_to_use} GPU(s).")
    if num_gpus_to_use > 1:
         print(f"Parallel preprocessing threads: {num_preprocessor_threads}")
    
    overall_start = time.time()
    
    # Get first image sample for VRAM estimation (actual image data or path)
    first_image_sample_data = None
    first_image_path_for_estimation = remaining_images[0] if remaining_images else None
    
    if first_image_path_for_estimation and num_gpus_to_use > 1: # For optimized loader, pass data
        try:
            first_img = cv2.imread(first_image_path_for_estimation)
            if first_img is not None:
                first_image_sample_data = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(f"Warning: Could not load first image sample for VRAM estimation: {e}")
            first_image_sample_data = None # Ensure it's None on failure
    
    # For single GPU mode, simple_gpu_worker uses the path directly
    # For optimized_gpu_worker (dual GPU), first_image_sample_data is preferred for initial VRAM est.

    all_results = [] # Initialize all_results

    if single_gpu_mode is not None:
        print(f"Using single GPU mode: GPU {single_gpu_mode}")
        # For single GPU mode, use simple queue-based approach
        manager = mp.Manager()
        work_queue = manager.JoinableQueue() # Use JoinableQueue
        results_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        # Add all images to queue
        for image_path in remaining_images:
            work_queue.put(image_path)
        work_queue.put(None) # Sentinel for the worker
        
        # Start progress monitor in a separate thread
        progress_thread = threading.Thread(
            target=progress_monitor, 
            args=(progress_queue, len(remaining_images), dataset_name, 1) # num_gpus_expected = 1
        )
        progress_thread.start()
        
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                simple_gpu_worker, single_gpu_mode, work_queue, results_queue, progress_queue,
                model_type, checkpoint_path, optimize_memory, dataset_output_dir, 
                dataset_name, len(remaining_images), first_image_path_for_estimation # Pass path for VRAM est.
            )
            
            try:
                processed_count_gpu = future.result(timeout=None) # Wait indefinitely
                print(f"Single GPU worker (GPU {single_gpu_mode}) completed, processed {processed_count_gpu} images.")
            except Exception as e:
                 print(f"Error in single GPU worker future: {e}")
                 import traceback
                 traceback.print_exc()

        # work_queue.join() # Wait for all items to be processed (task_done called)
        progress_queue.put(None) # Sentinel for progress monitor
        progress_thread.join()
        
        # Collect results
        while not results_queue.empty():
            try:
                all_results.append(results_queue.get_nowait())
            except queue.Empty:
                break
            
    else: # Dual GPU optimized mode
        print("Using optimized dual GPU mode with parallel preprocessing")
        print(f"Image pre-loading buffer size: {buffer_size}")
        
        # Create shared queues
        manager = mp.Manager()
        # Use JoinableQueue for prepared_queue so loader can signal completion implicitly
        # and workers can use task_done()
        prepared_queue = manager.JoinableQueue(maxsize=buffer_size)
        results_queue = manager.Queue() # Standard queue for results
        progress_queue = manager.Queue() # Standard queue for progress
        
        # Start progress monitor in a separate thread
        progress_thread = threading.Thread(
            target=progress_monitor, 
            args=(progress_queue, len(remaining_images), dataset_name, 2) # num_gpus_expected = 2
        )
        progress_thread.start()
        
        # Start optimized image loader and GPU workers
        # Max workers: 1 loader + 2 GPU workers = 3
        with ProcessPoolExecutor(max_workers=3) as executor:
            # Start optimized image loader with parallel preprocessing
            loader_future = executor.submit(
                optimized_image_loader_worker, remaining_images, prepared_queue, 
                buffer_size, num_preprocessor_threads
            )
            
            # Start both optimized GPU workers
            future_gpu0 = executor.submit(
                optimized_gpu_worker, 0, prepared_queue, results_queue, progress_queue, 
                model_type, checkpoint_path, optimize_memory, dataset_output_dir, 
                dataset_name, len(remaining_images), first_image_sample_data
            )
            future_gpu1 = executor.submit(
                optimized_gpu_worker, 1, prepared_queue, results_queue, progress_queue, 
                model_type, checkpoint_path, optimize_memory, dataset_output_dir, 
                dataset_name, len(remaining_images), first_image_sample_data
            )
            
            # Wait for optimized image loader to complete
            try:
                loaded_count = loader_future.result(timeout=None) # Wait indefinitely
                print(f"Optimized image loader completed: {loaded_count} images pre-processed and queued.")
            except Exception as e:
                print(f"Error in image loader future: {e}")
                import traceback
                traceback.print_exc()
            
            # After loader finishes, add sentinels for GPU workers
            # One sentinel per worker
            prepared_queue.put(None) 
            prepared_queue.put(None)
            
            # Wait for both GPU workers to complete
            processed_gpu0, processed_gpu1 = 0, 0
            try:
                processed_gpu0 = future_gpu0.result(timeout=None)
                print(f"GPU 0 worker completed, processed {processed_gpu0} images.")
            except Exception as e:
                print(f"Error in GPU 0 worker future: {e}")
                import traceback
                traceback.print_exc()

            try:
                processed_gpu1 = future_gpu1.result(timeout=None)
                print(f"GPU 1 worker completed, processed {processed_gpu1} images.")
            except Exception as e:
                print(f"Error in GPU 1 worker future: {e}")
                import traceback
                traceback.print_exc()

        # prepared_queue.join() # Wait for all items in prepared_queue to be processed
        progress_queue.put(None) # Sentinel for progress monitor
        progress_thread.join()
        
        # Collect all results from the results queue
        while not results_queue.empty():
            try:
                all_results.append(results_queue.get_nowait())
            except queue.Empty:
                break
    
    overall_time = time.time() - overall_start
    
    # Pass dataset_output_dir as the 'output_dir' for summary, where .sam is.
    stats = print_optimized_gpu_summary(all_results, overall_time, dataset_output_dir, dataset_name, num_gpus_to_use)
    
    # Check if all REMAINING images were processed successfully
    successful_results_count = sum(1 for r in all_results if r.get('success', False))
    
    if successful_results_count == len(remaining_images):
        print(f"✓ All {len(remaining_images)} remaining images in dataset '{dataset_name}' processed successfully!")
        mark_dataset_completed(dataset_output_dir, dataset_name, stats)
        return True, stats
    else:
        print(f"⚠️  Dataset '{dataset_name}' completed with {len(remaining_images) - successful_results_count} failures out of {len(remaining_images)} remaining. Not marking as fully completed unless all were processed.")
        # Decide if partially completed datasets should also get a .done file (e.g. with failure count)
        # For now, only mark if all remaining were successful.
        if successful_results_count > 0 and len(remaining_images) - successful_results_count < len(remaining_images) * 0.1 : # e.g. <10% failure
            print(f"Marking '{dataset_name}' as completed despite some failures due to low failure rate.")
            mark_dataset_completed(dataset_output_dir, dataset_name, stats)
            return True, stats # Still count as success for overall script if mostly done
        return False, stats


def main():
    parser = argparse.ArgumentParser(description="Optimized SAM Processing for a Single Dataset - Dual GPU with Advanced Preprocessing")
    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images for a single dataset (e.g., .../YOLOPOD/images)")
    parser.add_argument("--output_dir_path", required=True,
                        help="Path for the data subfolder (e.g., for the dataset WEDD '--output_dir_path /home/abood/groundingLMM/GranD/detection_outputs/WEDD')")
    parser.add_argument("--model-type", choices=["vit_h", "vit_l", "vit_b"], 
                       default="vit_h", help="SAM model type (default: vit_h)")
    parser.add_argument("--checkpoint", default=None, # Default to None, AdaptiveSAMProcessor will try to find it
                        help="Path to SAM checkpoint. If None, tries to find default based on model type (e.g., sam_vit_h_4b8939.pth in script dir or CWD).")
    parser.add_argument("--no-memory-opt", action='store_true', help="Disable memory optimizations like Flash Attention")
    parser.add_argument("--single-gpu", type=int, choices=[0, 1], default=None, # Default to None (dual GPU)
                        help="Use single GPU (specify 0 or 1) instead of optimized dual GPU. If not set, dual GPU is used if available.")
    parser.add_argument("--buffer-size", type=int, default=20, # Increased default buffer
                        help="Number of images to pre-load in RAM buffer for dual GPU mode (default: 20)")
    parser.add_argument("--preprocessor-threads", type=int, default=min((os.cpu_count() or 2) // 2, 8), # Default to half CPUs up to 8
                        help="Number of parallel preprocessing threads for dual GPU mode (default: min(CPUs/2, 8))")
    parser.add_argument("--resume", action='store_true', default=True, help="Resume processing (skip completed images/datasets) - enabled by default.")
    parser.add_argument("--no-resume", action='store_false', dest='resume', help="Disable resume functionality (reprocess everything).")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_dir_path):
        print(f"Error: Image directory path does not exist or is not a directory: {args.image_dir_path}")
        sys.exit(1)

    # The output_dir_path is the direct dataset_output_dir. Ensure it exists.
    # process_single_dataset will also create it and sam_output_dir within it.
    try:
        os.makedirs(args.output_dir_path, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create or access target output directory {args.output_dir_path}: {e}")
        sys.exit(1)

    print(f"Resume functionality: {'Enabled' if args.resume else 'Disabled'}")
    
    if args.buffer_size < 1:
        print("Error: Buffer size must be at least 1")
        sys.exit(1)
    elif args.buffer_size > 100: # Increased warning threshold
        print("Warning: Large buffer sizes (>100) may use excessive RAM")
    
    if args.preprocessor_threads < 1:
        print("Error: Number of preprocessor threads must be at least 1")
        sys.exit(1)
    elif args.preprocessor_threads > (os.cpu_count() or 1): # Warn if more threads than CPUs
        print(f"Warning: Preprocessor threads ({args.preprocessor_threads}) > CPU cores ({os.cpu_count()}). This might not be optimal.")
    
    if args.single_gpu is None: # Dual GPU mode settings
        print(f"Dual GPU Mode Settings:")
        print(f"  Image pre-loading buffer size: {args.buffer_size} images")
        print(f"  Parallel preprocessing threads: {args.preprocessor_threads}")
        print("  ✓ Optimized preprocessing: Parallel ThreadPoolExecutor")
        print("  ✓ Smart multi-scale variants: Large images only")
        print("  ✓ CPU-optimized pipeline: Maximum efficiency")
    else:
        print(f"Single GPU Mode (GPU: {args.single_gpu}) selected.")


    # Determine dataset_name from image_dir_path (for logging and .done file naming)
    path_obj = Path(args.image_dir_path)
    # Handle cases like "/path/to/dataset/images/" (trailing slash)
    effective_name = path_obj.name if path_obj.name else path_obj.parent.name

    if effective_name.lower() == "images":
        dataset_name = path_obj.parent.name
    else:
        dataset_name = effective_name # Use the name of the image_dir_path itself if not "images"
    
    if not dataset_name: # Edge case: if image_dir_path is root like '/' or 'C:/'
        print(f"Error: Could not derive a valid dataset name identifier from image_dir_path: {args.image_dir_path}")
        print("Please provide a path like '/path/to/dataset_name/images' or '/path/to/dataset_name_images'")
        sys.exit(1)

    dataset_input_images_dir = args.image_dir_path
    # The new output_dir_path is the direct dataset_output_dir
    dataset_output_dir = args.output_dir_path

    print(f"\n--- Initiating SAM Processing for Dataset Identifier: '{dataset_name}' ---")
    print(f"  Source Images: '{dataset_input_images_dir}'")
    print(f"  Target Output Directory: '{dataset_output_dir}'")


    try:
        success, stats = process_single_dataset(
            dataset_name=dataset_name, # This name is for the .done file and logging
            dataset_input_images_dir=dataset_input_images_dir,
            dataset_output_dir=dataset_output_dir, # This is args.output_dir_path
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            optimize_memory=not args.no_memory_opt,
            single_gpu_mode=args.single_gpu,
            buffer_size=args.buffer_size,
            num_preprocessor_threads=args.preprocessor_threads,
            use_resume=args.resume
        )
        
        print("\n" + "="*70)
        print(f"OVERALL PROCESSING SUMMARY FOR DATASET IDENTIFIER: '{dataset_name}'")
        print(f" (Output stored in: '{dataset_output_dir}')")
        print("="*70)
        if success:
            print(f"✓ Dataset '{dataset_name}' processing concluded successfully or was mostly complete.")
            if stats.get('total_failed', 0) > 0:
                 print(f"  Note: {stats['total_failed']} images failed processing.")
        else:
            print(f"✗ Dataset '{dataset_name}' processing encountered significant issues or was skipped.")
            if stats: # If stats were returned
                print(f"  Processed: {stats.get('total_processed',0)}, Failed: {stats.get('total_failed',0)}")

        print("✓ Check logs above for detailed statistics and any errors.")
        
    except KeyboardInterrupt:
        print(f"\nInterrupted during processing of dataset '{dataset_name}'. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\nCritical error processing dataset '{dataset_name}': {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Required for CUDA + multiprocessing on some systems (Windows, macOS with 'spawn' or 'forkserver')
    # 'fork' is default on Linux and usually fine.
    if sys.platform.startswith('win') or sys.platform.startswith('darwin'):
        if mp.get_start_method(allow_none=True) != 'spawn': # Avoid forcing if already set or not applicable
            try:
                mp.set_start_method('spawn', force=True) # 'spawn' is generally safer
            except RuntimeError as e:
                print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}")
                print("Ensure this script is run within 'if __name__ == \"__main__\":'")
    
    main()