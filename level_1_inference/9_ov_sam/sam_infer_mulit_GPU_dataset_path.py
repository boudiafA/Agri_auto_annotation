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
        self.output_dir = output_dir
        self.device = f"cuda:0"  # Always 0 since we set CUDA_VISIBLE_DEVICES
        
        # Adaptive batch size parameters
        self.initial_points_per_batch = 32  # Default fallback
        self.current_points_per_batch = self.initial_points_per_batch
        self.min_points_per_batch = 1  # Minimum safe value
        self.max_points_per_batch = 64  # Maximum to try
        self.oom_count = 0
        self.successful_batches = 0
        
        # Create output directory
        if self.output_dir:
            self.sam_output_dir = os.path.join(self.output_dir, "sam")
            os.makedirs(self.sam_output_dir, exist_ok=True)
        else:
            self.sam_output_dir = None
        
        print(f"Initializing Adaptive SAM Processor for GPU {gpu_id}")
        print(f"Device: {self.device} (Physical GPU {gpu_id})")
        print(f"Model: {model_type}")
        
        # Load model
        self.sam = self._load_stable_model(checkpoint_path)
        
        # Initialize mask generator with adaptive settings
        self.mask_generator = None
        
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
            test_values = [32, 64]#, 96, 128 , 192, 256, 384, 512]
            optimal_batch_size = 32
            
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
            
            print(f"GPU {self.gpu_id}: Optimal points_per_batch estimated: {optimal_batch_size}")
            
            # Initialize mask generator with optimal settings
            self._initialize_mask_generator()
            
            return optimal_batch_size
            
        except Exception as e:
            print(f"GPU {self.gpu_id}: Error in VRAM estimation: {e}")
            self.current_points_per_batch = self._get_conservative_batch_size()
            self.initial_points_per_batch = self.current_points_per_batch
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
        
        return self.current_points_per_batch > 0
    
    def _try_increase_batch_size(self):
        """Occasionally try to increase batch size if we've been successful"""
        if (self.successful_batches > 0 and 
            self.successful_batches % 20 == 0 and  # Every 20 successful images
            self.current_points_per_batch < self.max_points_per_batch and
            self.oom_count == 0):  # Only if no recent OOMs
            
            old_batch_size = self.current_points_per_batch
            self.current_points_per_batch = min(self.max_points_per_batch,
                                              int(self.current_points_per_batch * 1.2))
            
            # Reinitialize mask generator
            self._initialize_mask_generator()
    
    def _choose_optimal_image_size(self, prepared_data):
        """Choose optimal image size based on current GPU state and available variants"""
        if not prepared_data.get('has_variants', False):
            # Single image - return as is
            return prepared_data['image_data'], prepared_data['actual_size']
        
        # Multiple variants available - choose based on current points_per_batch
        variants = prepared_data['variants']
        
        # Choose size based on current GPU memory pressure
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
            checkpoint_paths = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            default_checkpoint = checkpoint_paths.get(self.model_type)
            if default_checkpoint and os.path.exists(default_checkpoint):
                checkpoint_path = default_checkpoint
            else:
                raise FileNotFoundError(f"Checkpoint not found for model type {self.model_type}")
        
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        sam.eval()
        
        if self.optimize_memory:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("✓ Flash attention enabled")
            except:
                pass
        
        return sam
    
    def _setup_stable_optimizations(self):
        """Setup stable GPU optimizations"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.cuda.empty_cache()
            
            if self.optimize_memory:
                try:
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    print(f"✓ GPU {self.gpu_id}: Memory limited to 80%")
                except:
                    pass
    
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
                    'fps': 1.0 / process_time,
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
                        print(f"GPU {self.gpu_id}: Cannot reduce batch size further")
                        break
                else:
                    print(f"GPU {self.gpu_id}: Max retries reached for {image_name}")
                    break
                    
            except Exception as e:
                print(f"GPU {self.gpu_id}: Error processing {image_name}: {e}")
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
                    'fps': 1.0 / process_time,
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
                        print(f"GPU {self.gpu_id}: Cannot reduce batch size further")
                        break
                else:
                    print(f"GPU {self.gpu_id}: Max retries reached for {os.path.basename(image_path)}")
                    break
                    
            except Exception as e:
                print(f"GPU {self.gpu_id}: Error processing {os.path.basename(image_path)}: {e}")
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
                        counts = rle['counts']
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
                    rle_counts.append(0)
                for length in run_lengths:
                    rle_counts.append(int(length))
            else:
                if len(mask_flat) > 0 and mask_flat[0] == 1:
                    rle_counts = [0, len(mask_flat)]
                else:
                    rle_counts = [len(mask_flat)]
            
            return rle_counts
            
        except Exception as e:
            print(f"GPU {self.gpu_id}: Error in RLE encoding: {e}")
            return [mask.size]
    
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
                    json.load(f)
                return json_path
            except json.JSONDecodeError as e:
                print(f"✗ GPU {self.gpu_id}: Invalid JSON written for {image_name}: {e}")
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
    
    def process_single_image(image_path):
        """Process single image with error handling"""
        try:
            return preprocessor.preprocess_single_image(image_path)
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
            return None
    
    try:
        # Use ThreadPoolExecutor for parallel preprocessing
        with ThreadPoolExecutor(max_workers=num_preprocessor_threads) as executor:
            # Submit all preprocessing jobs
            future_to_path = {
                executor.submit(process_single_image, path): path 
                for path in image_paths
            }
            
            # Collect results as they complete
            for future in future_to_path:
                try:
                    prepared_data = future.result()
                    if prepared_data is not None:
                        # Track variant statistics
                        if prepared_data.get('has_variants', False):
                            variant_count += 1
                        
                        # Put in queue (blocks if queue is full - provides backpressure)
                        prepared_queue.put(prepared_data)
                        loaded_count += 1
                        
                            
                except Exception as e:
                    image_path = future_to_path[future]
                    print(f"Failed to process {os.path.basename(image_path)}: {e}", flush=True)
                    sys.stdout.flush()
                    continue
        
        print(f"Optimized Image Loader: FINISHED", flush=True)
        print(f"Successfully processed: {loaded_count} images", flush=True)
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


def optimized_gpu_worker(gpu_id, prepared_queue, results_queue, progress_queue, model_type, checkpoint_path, 
                        optimize_memory, output_dir, dataset_name, total_images, first_image_sample=None):
    """
    OPTIMIZED: GPU worker that uses optimized preprocessing data
    """
    import sys
    try:
        # Initialize adaptive processor
        processor = AdaptiveSAMProcessor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id,
            optimize_memory=optimize_memory,
            output_dir=output_dir
        )
        
        print(f"GPU {gpu_id}: Ready for optimized processing of dataset '{dataset_name}'", flush=True)
        sys.stdout.flush()
        
        # Enhanced warmup with VRAM-based estimation
        print(f"GPU {gpu_id}: Starting enhanced warmup with VRAM-based points_per_batch estimation...", flush=True)
        sys.stdout.flush()
        
        warmup_done = False
        warmup_attempts = 0
        max_warmup_attempts = 10
        
        while not warmup_done and warmup_attempts < max_warmup_attempts:
            try:
                warmup_attempts += 1
                print(f"GPU {gpu_id}: Warmup attempt {warmup_attempts}/{max_warmup_attempts}", flush=True)
                sys.stdout.flush()
                
                # Try to get a sample image for VRAM estimation
                test_image_data = None
                if first_image_sample is not None:
                    test_image_data = first_image_sample
                else:
                    try:
                        warmup_data = prepared_queue.get(timeout=10)  # 10 second timeout per attempt
                        if warmup_data.get('has_variants', False):
                            # Use one of the variants for testing
                            test_variant = list(warmup_data['variants'].values())[0]
                            test_image_data = test_variant['image_data']
                        else:
                            test_image_data = warmup_data['image_data']
                        prepared_queue.put(warmup_data)  # Put it back
                    except:
                        pass
                
                # Estimate optimal points_per_batch using VRAM measurement
                optimal_batch_size = processor.estimate_optimal_points_per_batch(test_image_data=test_image_data)
                
                print(f"GPU {gpu_id}: VRAM-based estimation completed. Optimal points_per_batch: {optimal_batch_size}", flush=True)
                sys.stdout.flush()
                
                # Test with a real image if available
                if test_image_data is not None:
                    test_data = {
                        'image_data': test_image_data, 
                        'image_name': 'warmup_test', 
                        'image_path': 'warmup',
                        'has_variants': False,
                        'actual_size': test_image_data.shape[:2]
                    }
                    warmup_result = processor.process_optimized_image(test_data)
                    if warmup_result.get('success'):
                        print(f"GPU {gpu_id}: Warmup successful: {warmup_result['masks']} masks in {warmup_result['time']:.2f}s "
                              f"(final points_per_batch: {warmup_result.get('points_per_batch', 'unknown')})", flush=True)
                        warmup_done = True
                    else:
                        print(f"GPU {gpu_id}: Warmup failed, but proceeding anyway", flush=True)
                        warmup_done = True
                else:
                    print(f"GPU {gpu_id}: No test image available for warmup, proceeding with VRAM estimation", flush=True)
                    warmup_done = True
                    
                sys.stdout.flush()
                
            except Exception as e:
                print(f"GPU {gpu_id}: Warmup attempt {warmup_attempts} failed: {e}", flush=True)
                if warmup_attempts >= max_warmup_attempts:
                    print(f"GPU {gpu_id}: All warmup attempts failed, proceeding without warmup", flush=True)
                    warmup_done = True
                sys.stdout.flush()
        
        processed_count = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 5
        variant_used_count = 0
        
        # Process prepared images from queue until empty
        while consecutive_timeouts < max_consecutive_timeouts:
            try:
                # Get prepared image from queue with timeout
                prepared_data = prepared_queue.get(timeout=5)  # 5 second timeout
                consecutive_timeouts = 0  # Reset timeout counter
                
                try:
                    # Use optimized processing method
                    result = processor.process_optimized_image(prepared_data)
                    results_queue.put(result)
                    processed_count += 1
                    
                    # Track variant usage
                    if result.get('has_variants', False):
                        variant_used_count += 1
                    
                    # Send progress update with queue size information
                    try:
                        queue_size = prepared_queue.qsize()
                    except:
                        queue_size = 0
                    
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count,
                        'image_name': prepared_data['image_name'],
                        'success': result.get('success', False),
                        'fps': result.get('fps', 0),
                        'masks': result.get('masks', 0),
                        'batch_size': result.get('points_per_batch', 0),
                        'queue_size': queue_size,
                        'has_variants': result.get('has_variants', False)
                    })
                    
                    # Mark task as done
                    prepared_queue.task_done()
                    
                except Exception as e:
                    print(f"GPU {gpu_id}: Error processing {prepared_data['image_name']}: {e}", flush=True)
                    sys.stdout.flush()
                    # Put error result
                    results_queue.put({
                        'path': prepared_data['image_name'],
                        'error': str(e),
                        'success': False,
                        'gpu_id': gpu_id
                    })
                    
                    # Send progress update for failed image
                    try:
                        queue_size = prepared_queue.qsize()
                    except:
                        queue_size = 0
                        
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count + 1,
                        'image_name': prepared_data['image_name'],
                        'success': False,
                        'error': str(e),
                        'queue_size': queue_size
                    })
                    
                    prepared_queue.task_done()
                    
            except:  # Queue timeout
                consecutive_timeouts += 1
                if consecutive_timeouts % 10 == 0:  # Only print timeout info every 10 timeouts
                    print(f"GPU {gpu_id}: Timeout {consecutive_timeouts}/{max_consecutive_timeouts} waiting for images", flush=True)
                    sys.stdout.flush()
        
        print(f"GPU {gpu_id}: No more prepared images available. Processed {processed_count} images.", flush=True)
        print(f"GPU {gpu_id}: Used multi-scale variants for {variant_used_count} images", flush=True)
        sys.stdout.flush()
        
        # Print final stats for this GPU
        final_batch_size = processor.current_points_per_batch
        initial_batch_size = processor.initial_points_per_batch
        oom_count = processor.oom_count
        
        print(f"GPU {gpu_id}: Completed optimized processing", flush=True)
        print(f"GPU {gpu_id}: Images processed: {processed_count}", flush=True)
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


def simple_gpu_worker(gpu_id, work_queue, results_queue, progress_queue, model_type, checkpoint_path, 
                     optimize_memory, output_dir, dataset_name, total_images, first_image_path=None):
    """
    Simple worker function that processes images from a work queue (on-demand loading)
    """
    import sys
    try:
        # Initialize adaptive processor
        processor = AdaptiveSAMProcessor(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id,
            optimize_memory=optimize_memory,
            output_dir=output_dir
        )
        
        print(f"GPU {gpu_id}: Ready for simple processing of dataset '{dataset_name}'", flush=True)
        sys.stdout.flush()
        
        # Enhanced warmup with VRAM-based estimation
        if first_image_path and os.path.exists(first_image_path):
            print(f"GPU {gpu_id}: Starting VRAM-based points_per_batch estimation with test image...", flush=True)
            optimal_batch_size = processor.estimate_optimal_points_per_batch(test_image_path=first_image_path)
            print(f"GPU {gpu_id}: Optimal points_per_batch: {optimal_batch_size}", flush=True)
        else:
            print(f"GPU {gpu_id}: No test image available, using conservative estimation", flush=True)
            processor.current_points_per_batch = processor._get_conservative_batch_size()
            processor.initial_points_per_batch = processor.current_points_per_batch
            processor._initialize_mask_generator()
        
        processed_count = 0
        consecutive_timeouts = 0
        max_consecutive_timeouts = 5
        
        # Process images from queue until empty
        while consecutive_timeouts < max_consecutive_timeouts:
            try:
                # Get image path from queue with timeout
                image_path = work_queue.get(timeout=5)  # 5 second timeout
                consecutive_timeouts = 0  # Reset timeout counter
                
                try:
                    result = processor.process_single_image(image_path)
                    results_queue.put(result)
                    processed_count += 1
                    
                    # Send progress update
                    try:
                        queue_size = work_queue.qsize()
                    except:
                        queue_size = 0
                    
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count,
                        'image_name': os.path.basename(image_path),
                        'success': result.get('success', False),
                        'fps': result.get('fps', 0),
                        'masks': result.get('masks', 0),
                        'batch_size': result.get('points_per_batch', 0),
                        'queue_size': queue_size
                    })
                    
                    # Mark task as done
                    work_queue.task_done()
                    
                except Exception as e:
                    print(f"GPU {gpu_id}: Error processing {os.path.basename(image_path)}: {e}", flush=True)
                    sys.stdout.flush()
                    # Put error result
                    results_queue.put({
                        'path': os.path.basename(image_path),
                        'error': str(e),
                        'success': False,
                        'gpu_id': gpu_id
                    })
                    
                    # Send progress update for failed image
                    try:
                        queue_size = work_queue.qsize()
                    except:
                        queue_size = 0
                        
                    progress_queue.put({
                        'gpu_id': gpu_id,
                        'processed_count': processed_count + 1,
                        'image_name': os.path.basename(image_path),
                        'success': False,
                        'error': str(e),
                        'queue_size': queue_size
                    })
                    
                    work_queue.task_done()
                    
            except:  # Queue timeout
                consecutive_timeouts += 1
                if consecutive_timeouts % 10 == 0:  # Only print timeout info every 10 timeouts
                    print(f"GPU {gpu_id}: Timeout {consecutive_timeouts}/{max_consecutive_timeouts} waiting for images", flush=True)
                    sys.stdout.flush()
        
        print(f"GPU {gpu_id}: No more images available. Processed {processed_count} images.", flush=True)
        sys.stdout.flush()
        
        # Print final stats for this GPU
        final_batch_size = processor.current_points_per_batch
        initial_batch_size = processor.initial_points_per_batch
        oom_count = processor.oom_count
        
        print(f"GPU {gpu_id}: Completed simple processing", flush=True)
        print(f"GPU {gpu_id}: Images processed: {processed_count}", flush=True)
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


def get_image_paths(input_path: str):
    """Get image paths efficiently from a given directory, case-insensitive."""
    if not os.path.isdir(input_path):
        raise ValueError(f"Input path is not a directory: {input_path}")
    
    valid_extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
    all_files = glob.glob(os.path.join(input_path, "*"))
    
    image_paths = [
        f for f in all_files
        if os.path.isfile(f) and f.lower().split('.')[-1] in valid_extensions
    ]
    
    return sorted(image_paths)


def check_dataset_completion(dataset_output_dir, dataset_name):
    """Check if dataset is already completed (has .done file)"""
    done_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.done")
    return os.path.exists(done_file_path)


def mark_dataset_completed(dataset_output_dir, dataset_name, stats):
    """Mark dataset as completed by creating .done file with stats"""
    try:
        done_file_path = os.path.join(dataset_output_dir, f"{dataset_name}.done")
        
        completion_info = {
            "dataset_name": dataset_name,
            "completion_time": time.time(),
            "completion_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_images_processed": stats.get('total_processed', 0),
            "total_images_failed": stats.get('total_failed', 0),
            "processing_mode": stats.get('processing_mode', 'unknown'),
            "total_masks_generated": stats.get('total_masks', 0),
            "total_processing_time": stats.get('total_time', 0),
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
                    if 'annotations' in data and 'num_masks' in data:
                        already_processed.append(image_path)
                        continue
            except (json.JSONDecodeError, IOError):
                pass
        
        remaining_images.append(image_path)
    
    return remaining_images, already_processed


def print_optimized_gpu_summary(all_results, overall_time, output_dir, dataset_name):
    """Print comprehensive summary for optimized dual GPU processing"""
    gpu0_results = [r for r in all_results if r.get('gpu_id') == 0 and r.get('success', False)]
    gpu1_results = [r for r in all_results if r.get('gpu_id') == 1 and r.get('success', False)]
    successful_results = [r for r in all_results if r.get('success', False)]
    failed_results = [r for r in all_results if not r.get('success', False)]
    
    total_masks = sum(r['masks'] for r in successful_results)
    successful_json_saves = sum(1 for r in successful_results if r.get('json_path'))
    
    # Count optimization statistics
    variant_results = [r for r in successful_results if r.get('has_variants', False)]
    
    print("\n" + "="*70)
    print(f"OPTIMIZED DUAL GPU SAM PROCESSING SUMMARY")
    print(f"DATASET: {dataset_name}")
    print("="*70)
    print(f"Processing mode: Optimized Dual GPU with Advanced Pre-processing")
    print(f"Images processed: {len(successful_results)}")
    print(f"Images failed: {len(failed_results)}")
    print(f"Successful JSON saves: {successful_json_saves}/{len(all_results)}")
    print(f"Total time: {overall_time:.2f}s")
    print(f"Overall throughput: {len(all_results)/overall_time:.2f} images/sec")
    print(f"Total masks: {total_masks:,}")
    
    # Optimization statistics
    print(f"\nOptimization Statistics:")
    print(f"  Images with multi-scale variants: {len(variant_results)}")
    print(f"  Memory-efficient single-scale: {len(successful_results) - len(variant_results)}")
    print(f"  Parallel preprocessing: ✓ Enabled")
    print(f"  Smart size selection: ✓ Enabled")
    
    # Dynamic distribution stats
    print(f"\nDynamic Work Distribution:")
    print(f"  GPU 0 processed: {len(gpu0_results)} images ({len(gpu0_results)/len(all_results)*100:.1f}%)")
    print(f"  GPU 1 processed: {len(gpu1_results)} images ({len(gpu1_results)/len(all_results)*100:.1f}%)")
    
    # GPU-specific stats with batch size info
    if gpu0_results:
        gpu0_fps = [r['fps'] for r in gpu0_results]
        gpu0_times = [r['time'] for r in gpu0_results]
        gpu0_batches = [r.get('points_per_batch', 0) for r in gpu0_results if 'points_per_batch' in r]
        print(f"\nGPU 0 Stats:")
        print(f"  Average FPS: {np.mean(gpu0_fps):.2f}")
        print(f"  Average time: {np.mean(gpu0_times):.2f}s")
        if gpu0_batches:
            print(f"  Final points_per_batch: {gpu0_batches[-1] if gpu0_batches else 'N/A'}")
            print(f"  Average points_per_batch: {np.mean(gpu0_batches):.1f}")
    
    if gpu1_results:
        gpu1_fps = [r['fps'] for r in gpu1_results]
        gpu1_times = [r['time'] for r in gpu1_results]
        gpu1_batches = [r.get('points_per_batch', 0) for r in gpu1_results if 'points_per_batch' in r]
        print(f"\nGPU 1 Stats:")
        print(f"  Average FPS: {np.mean(gpu1_fps):.2f}")
        print(f"  Average time: {np.mean(gpu1_times):.2f}s")
        if gpu1_batches:
            print(f"  Final points_per_batch: {gpu1_batches[-1] if gpu1_batches else 'N/A'}")
            print(f"  Average points_per_batch: {np.mean(gpu1_batches):.1f}")
    
    # Print failed images if any
    if failed_results:
        print(f"\n⚠️  Failed to process {len(failed_results)} images:")
        for result in failed_results[:5]:
            gpu_info = f"(GPU {result.get('gpu_id', '?')})" if 'gpu_id' in result else ""
            print(f"  - {result['path']}: {result.get('error', 'Unknown error')} {gpu_info}")
        if len(failed_results) > 5:
            print(f"  ... and {len(failed_results) - 5} more")
    
    sam_output_dir = os.path.join(output_dir, "sam")
    print(f"\nJSON files saved to: {sam_output_dir}")
    print("✓ Optimized parallel preprocessing enabled")
    print("✓ Smart multi-scale processing for large images")
    print("✓ CPU-optimized image processing pipeline")
    
    # Return stats for dataset completion tracking
    return {
        'total_processed': len(successful_results),
        'total_failed': len(failed_results),
        'total_masks': total_masks,
        'total_time': overall_time,
        'average_fps': len(all_results)/overall_time if overall_time > 0 else 0,
        'processing_mode': 'Optimized Dual GPU with Advanced Pre-processing'
    }


def progress_monitor(progress_queue, total_images, dataset_name):
    """Enhanced progress monitor with optimization statistics"""
    gpu_stats = {0: {'count': 0, 'last_image': '', 'last_fps': 0, 'last_masks': 0, 'last_batch': 0},
                 1: {'count': 0, 'last_image': '', 'last_fps': 0, 'last_masks': 0, 'last_batch': 0}}
    
    total_processed = 0
    current_queue_size = 0
    variant_count = 0
    
    with tqdm(total=total_images, desc=f"Processing {dataset_name}", unit="img") as pbar:
        while total_processed < total_images:
            try:
                # Get progress update with timeout
                update = progress_queue.get(timeout=1)
                gpu_id = update['gpu_id']
                
                # Update GPU stats
                gpu_stats[gpu_id]['count'] = update['processed_count']
                gpu_stats[gpu_id]['last_image'] = update['image_name']
                
                # Update queue size if available
                if 'queue_size' in update:
                    current_queue_size = update['queue_size']
                
                # Track variants
                if update.get('has_variants', False):
                    variant_count += 1
                
                if update.get('success', False):
                    gpu_stats[gpu_id]['last_fps'] = update.get('fps', 0)
                    gpu_stats[gpu_id]['last_masks'] = update.get('masks', 0)
                    gpu_stats[gpu_id]['last_batch'] = update.get('batch_size', 0)
                
                # Update total progress
                new_total = gpu_stats[0]['count'] + gpu_stats[1]['count']
                if new_total > total_processed:
                    pbar.update(new_total - total_processed)
                    total_processed = new_total
                
                # Enhanced progress bar description with optimization info
                gpu0_fps = gpu_stats[0]['last_fps']
                gpu1_fps = gpu_stats[1]['last_fps']
                
                status_parts = []
                status_parts.append(f"GPU0: {gpu0_fps:.2f}fps")
                status_parts.append(f"GPU1: {gpu1_fps:.2f}fps")
                
                if current_queue_size > 0:
                    status_parts.append(f"Queue: {current_queue_size}")
                
                if variant_count > 0:
                    status_parts.append(f"Variants: {variant_count}")
                
                pbar.set_postfix_str(" | ".join(status_parts))
                
            except:  # Timeout - check if we're done
                if total_processed >= total_images:
                    break
                continue
    
    return gpu_stats


def process_single_dataset(dataset_name, dataset_input_images_dir, dataset_output_dir, 
                          model_type, checkpoint_path, optimize_memory, single_gpu_mode=None, 
                          buffer_size=10, num_preprocessor_threads=4):
    """Process a single dataset with optimized dual GPU or single GPU mode"""
    
    # Check if dataset is already completed
    if check_dataset_completion(dataset_output_dir, dataset_name):
        print(f"✓ Dataset '{dataset_name}' already completed (found .done file). Skipping.")
        return True
    
    print(f"  Input images from: '{dataset_input_images_dir}'")
    print(f"  Expected JSON output: '{dataset_output_dir}/sam/'")
    
    image_paths = get_image_paths(dataset_input_images_dir)
    
    if not image_paths:
        print(f"No images found in '{dataset_input_images_dir}'. Skipping this dataset.")
        return False
    
    sam_output_dir = os.path.join(dataset_output_dir, "sam")
    remaining_images, already_processed = check_already_processed(image_paths, sam_output_dir)
    
    if already_processed:
        print(f"Resume mode: Found {len(already_processed)} already processed images")
        print(f"Remaining to process: {len(remaining_images)} images")
    
    if not remaining_images:
        print(f"All {len(image_paths)} images already processed.")
        
        # Mark as completed and create .done file
        stats = {
            'total_processed': len(image_paths),
            'total_failed': 0,
            'total_masks': 0,  # Could calculate from existing JSON files if needed
            'total_time': 0,
            'average_fps': 0,
            'processing_mode': 'Previously completed'
        }
        mark_dataset_completed(dataset_output_dir, dataset_name, stats)
        return True
    
    print(f"Processing {len(remaining_images)} images with optimized dual GPU processing")
    print(f"Parallel preprocessing threads: {num_preprocessor_threads}")
    
    overall_start = time.time()
    
    # Get first image sample for VRAM estimation
    first_image_sample = None
    first_image_path = remaining_images[0] if remaining_images else None
    
    if first_image_path:
        try:
            first_image = cv2.imread(first_image_path)
            if first_image is not None:
                first_image_sample = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
        except:
            pass
    
    if single_gpu_mode is not None:
        print(f"Using single GPU mode: GPU {single_gpu_mode}")
        # For single GPU mode, use simple queue-based approach without optimized pre-loading
        manager = mp.Manager()
        work_queue = manager.Queue()
        results_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        # Add all images to queue
        for image_path in remaining_images:
            work_queue.put(image_path)
        
        # Start progress monitor in a separate thread
        progress_thread = threading.Thread(
            target=progress_monitor, 
            args=(progress_queue, len(remaining_images), dataset_name)
        )
        progress_thread.start()
        
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                simple_gpu_worker, single_gpu_mode, work_queue, results_queue, progress_queue,
                model_type, checkpoint_path, optimize_memory, dataset_output_dir, 
                dataset_name, len(remaining_images), first_image_path
            )
            
            processed_count = future.result()
            
        # Wait for progress monitor to finish
        progress_thread.join()
        
        # Collect results
        all_results = []
        while not results_queue.empty():
            all_results.append(results_queue.get())
            
    else:
        print("Using optimized dual GPU mode with parallel preprocessing")
        print(f"Image pre-loading buffer size: {buffer_size}")
        
        # Create shared queues
        manager = mp.Manager()
        prepared_queue = manager.Queue(maxsize=buffer_size)  # Limited buffer size
        results_queue = manager.Queue()
        progress_queue = manager.Queue()
        
        # Start progress monitor in a separate thread
        progress_thread = threading.Thread(
            target=progress_monitor, 
            args=(progress_queue, len(remaining_images), dataset_name)
        )
        progress_thread.start()
        
        # Start optimized image loader and GPU workers
        with ProcessPoolExecutor(max_workers=3) as executor:  # 1 loader + 2 GPU workers
            # Start optimized image loader with parallel preprocessing
            loader_future = executor.submit(
                optimized_image_loader_worker, remaining_images, prepared_queue, 
                buffer_size, num_preprocessor_threads
            )
            
            # Start both optimized GPU workers
            future_gpu0 = executor.submit(
                optimized_gpu_worker, 0, prepared_queue, results_queue, progress_queue, 
                model_type, checkpoint_path, optimize_memory, dataset_output_dir, 
                dataset_name, len(remaining_images), first_image_sample
            )
            future_gpu1 = executor.submit(
                optimized_gpu_worker, 1, prepared_queue, results_queue, progress_queue, 
                model_type, checkpoint_path, optimize_memory, dataset_output_dir, 
                dataset_name, len(remaining_images), first_image_sample
            )
            
            # Wait for optimized image loader to complete
            loaded_count = loader_future.result()
            print(f"Optimized image loader completed: {loaded_count} images processed")
            
            # Wait for both GPU workers to complete
            processed_gpu0 = future_gpu0.result()
            processed_gpu1 = future_gpu1.result()
        
        # Wait for progress monitor to finish
        progress_thread.join()
        
        print(f"GPU 0 processed: {processed_gpu0} images")
        print(f"GPU 1 processed: {processed_gpu1} images")
        
        # Collect all results from the results queue
        all_results = []
        while not results_queue.empty():
            all_results.append(results_queue.get())
    
    overall_time = time.time() - overall_start
    
    stats = print_optimized_gpu_summary(all_results, overall_time, dataset_output_dir, dataset_name)
    
    # Check if all images were processed successfully
    successful_results = [r for r in all_results if r.get('success', False)]
    
    if len(successful_results) == len(remaining_images):
        print(f"✓ All images in dataset '{dataset_name}' processed successfully!")
        mark_dataset_completed(dataset_output_dir, dataset_name, stats)
    else:
        print(f"⚠️  Dataset '{dataset_name}' completed with some failures. Not marking as fully completed.")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Optimized SAM Processing - Dual GPU with Advanced Preprocessing")
    parser.add_argument("--input_root_dir", required=True, 
                        help="Path to the root directory containing dataset folders")
    parser.add_argument("--output_base_dir", required=True, 
                        help="Base output directory for JSON files")
    parser.add_argument("--model-type", choices=["vit_h", "vit_l", "vit_b"], 
                       default="vit_h", help="SAM model type")
    parser.add_argument("--checkpoint", default="./sam_vit_h_4b8939.pth", help="Path to SAM checkpoint")
    parser.add_argument("--no-memory-opt", action='store_true', help="Disable memory optimizations")
    parser.add_argument("--single-gpu", type=int, choices=[0, 1], help="Use single GPU instead of optimized dual GPU")
    parser.add_argument("--buffer-size", type=int, default=10, help="Number of images to pre-load in RAM buffer (default: 10)")
    parser.add_argument("--preprocessor-threads", type=int, default=4, help="Number of parallel preprocessing threads (default: 4)")
    parser.add_argument("--resume", action='store_true', default=True, help="Resume processing - enabled by default")
    parser.add_argument("--no-resume", action='store_true', help="Disable resume functionality")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_root_dir):
        print(f"Error: Input root directory does not exist: {args.input_root_dir}")
        sys.exit(1)
    
    use_resume = args.resume and not args.no_resume
    if args.no_resume:
        print("Resume functionality disabled - will reprocess all images")
    else:
        print("Resume functionality enabled with optimized processing")
    
    if args.buffer_size < 1:
        print("Error: Buffer size must be at least 1")
        sys.exit(1)
    elif args.buffer_size > 50:
        print("Warning: Large buffer sizes (>50) may use excessive RAM")
    
    if args.preprocessor_threads < 1:
        print("Error: Number of preprocessor threads must be at least 1")
        sys.exit(1)
    elif args.preprocessor_threads > 16:
        print("Warning: Very high thread counts (>16) may not provide additional benefits")
    
    print(f"Image pre-loading buffer size: {args.buffer_size} images")
    print(f"Parallel preprocessing threads: {args.preprocessor_threads}")
    print("✓ Optimized preprocessing: Parallel ThreadPoolExecutor")
    print("✓ Smart multi-scale variants: Large images only")
    print("✓ CPU-optimized pipeline: Maximum efficiency")
    
    dataset_folders = [f for f in os.listdir(args.input_root_dir) 
                      if os.path.isdir(os.path.join(args.input_root_dir, f))]
    
    if not dataset_folders:
        print(f"No dataset subfolders found in '{args.input_root_dir}'.")
        sys.exit(1)

    dataset_folders.sort()
    total_datasets = len(dataset_folders)
    print(f"Found {total_datasets} dataset folders for optimized processing.")
    
    overall_successful_datasets = 0
    overall_skipped_datasets = 0

    for i, dataset_name in enumerate(dataset_folders):
        dataset_input_images_dir = os.path.join(args.input_root_dir, dataset_name, "images")
        dataset_output_dir = os.path.join(args.output_base_dir, dataset_name)

        print(f"\n--- Processing Dataset {i+1}/{total_datasets}: '{dataset_name}' ---")
        
        if not os.path.isdir(dataset_input_images_dir):
            print(f"Warning: Skipping '{dataset_name}'. No 'images' subfolder found.")
            overall_skipped_datasets += 1
            continue

        try:
            success = process_single_dataset(
                dataset_name=dataset_name,
                dataset_input_images_dir=dataset_input_images_dir,
                dataset_output_dir=dataset_output_dir,
                model_type=args.model_type,
                checkpoint_path=args.checkpoint,
                optimize_memory=not args.no_memory_opt,
                single_gpu_mode=args.single_gpu,
                buffer_size=args.buffer_size,
                num_preprocessor_threads=args.preprocessor_threads
            )
            
            if success:
                overall_successful_datasets += 1
                print(f"Dataset '{dataset_name}' optimized processing complete!")
            else:
                overall_skipped_datasets += 1
            
        except KeyboardInterrupt:
            print(f"\nInterrupted during processing of dataset '{dataset_name}'.")
            sys.exit(0)
        except Exception as e:
            print(f"\nError processing dataset '{dataset_name}': {e}")
            import traceback
            traceback.print_exc()
            overall_skipped_datasets += 1
            continue

    print("\n" + "="*70)
    print("ALL DATASETS OPTIMIZED PROCESSING SUMMARY")
    print("="*70)
    print(f"Total datasets found: {total_datasets}")
    print(f"Successfully processed datasets: {overall_successful_datasets}")
    print(f"Skipped/Failed datasets: {overall_skipped_datasets}")
    print("✓ Optimized processing complete with advanced preprocessing")

if __name__ == "__main__":
    main()