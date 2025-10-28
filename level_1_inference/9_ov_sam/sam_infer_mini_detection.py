#!/usr/bin/env python3
"""
Stable SAM Batch Processing - Sequential processing, no threading
Now supports incremental processing - skips images with existing JSON files
"""

# python sam_infer.py --image_dir_path "/mnt/e/Desktop/AgML/datasets_sorted/detection/almond_bloom_2023/images" --output_dir_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection/almond_bloom_2023"
# python sam_infer.py --image_dir_path "/mnt/e/Desktop/AgML/datasets_sorted/detection/almond_harvest_2021/images" --output_dir_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection/almond_harvest_2021"
import os
import time
import glob
import argparse
import json
from pathlib import Path
import sys

# GPU settings - KEEP synchronous for stability
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Synchronous for stability
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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


class StableSequentialSAMProcessor:
    def __init__(self, model_type: str = "vit_h", checkpoint_path: str = None,
                 gpu_id: int = 0, optimize_memory: bool = True, output_dir: str = None):
        """
        Stable SAM processor - sequential processing for maximum stability
        
        Args:
            optimize_memory: Enable memory optimizations
        """
        self.model_type = model_type
        self.gpu_id = gpu_id
        self.optimize_memory = optimize_memory
        self.output_dir = output_dir
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        
        # Create output directory with "sam" subfolder
        if self.output_dir:
            self.sam_output_dir = os.path.join(self.output_dir, "sam")
            os.makedirs(self.sam_output_dir, exist_ok=True)
        else:
            self.sam_output_dir = None
        
        print(f"Stable Sequential SAM Processor")
        print(f"Device: {self.device}")
        print(f"Model: {model_type}")
        print(f"Processing mode: Sequential (most stable)")
        print(f"Mixed precision: DISABLED")
        print(f"Memory optimization: {optimize_memory}")
        print(f"Incremental processing: ENABLED (skips existing JSON files)")
        if self.sam_output_dir:
            print(f"JSON output directory: {self.sam_output_dir}")
        
        # Load model with conservative optimizations
        self.sam = self._load_stable_model(checkpoint_path)
        
        # Setup mask generator with optimized but stable settings
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=24,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
            points_per_batch=128
        )
        
        # Enable safe optimizations
        self._setup_stable_optimizations()
        
        print("Model loaded successfully!")
    
    def _load_stable_model(self, checkpoint_path: str):
        """Load SAM model with stable optimizations"""
        if checkpoint_path is None:
            checkpoint_paths = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth",
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            checkpoint_path = checkpoint_paths.get(self.model_type)
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load model
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        sam.eval()  # Evaluation mode
        
        # Conservative torch optimizations
        if self.optimize_memory:
            # Enable memory efficient attention if available
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("✓ Flash attention enabled")
            except:
                pass
        
        return sam
    
    def _setup_stable_optimizations(self):
        """Setup stable GPU optimizations that work with synchronous mode"""
        if torch.cuda.is_available():
            # Conservative cuDNN settings
            torch.backends.cudnn.benchmark = True  # Safe optimization
            torch.backends.cudnn.deterministic = False
            
            # Clear cache
            torch.cuda.empty_cache()
            
            # Conservative memory settings
            if self.optimize_memory:
                # Set memory fraction to avoid OOM
                try:
                    torch.cuda.set_per_process_memory_fraction(0.8)
                    print("✓ GPU memory limited to 80%")
                except:
                    pass
    
    def check_existing_json(self, image_path: str) -> bool:
        """
        Check if JSON file already exists for the given image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if JSON file exists and is valid, False otherwise
        """
        if not self.sam_output_dir:
            return False
        
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(self.sam_output_dir, f"{image_name}.json")
            
            # Check if file exists
            if not os.path.exists(json_path):
                return False
            
            # Validate JSON file by trying to load it
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Basic validation - check if it has expected structure
                if isinstance(data, dict) and 'annotations' in data:
                    # Check if it has reasonable content (not empty)
                    annotations = data.get('annotations', [])
                    if isinstance(annotations, list):
                        return True
                
                return False
                
            except (json.JSONDecodeError, Exception):
                # If JSON is corrupted, treat as if it doesn't exist
                print(f"Warning: Corrupted JSON file found for {image_name}, will regenerate")
                return False
                
        except Exception as e:
            print(f"Error checking existing JSON for {os.path.basename(image_path)}: {e}")
            return False
    
    def filter_images_for_processing(self, image_paths):
        """
        Filter image paths to only include those that need processing
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Tuple of (images_to_process, images_already_processed)
        """
        images_to_process = []
        images_already_processed = []
        
        print("🔍 Checking for existing JSON files...")
        
        for image_path in tqdm(image_paths, desc="Checking existing", unit="image"):
            if self.check_existing_json(image_path):
                images_already_processed.append(image_path)
            else:
                images_to_process.append(image_path)
        
        return images_to_process, images_already_processed
    
    def load_image_efficiently(self, path):
        """Load image with error handling and optimization"""
        try:
            # Load image
            image = cv2.imread(path)
            if image is None:
                raise ValueError(f"Could not load image: {path}")
            
            # Convert to RGB efficiently
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Optional: resize if too large to save GPU memory
            h, w = image.shape[:2]
            max_size = 1600  # Adjust based on your GPU memory
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                # print(f"Resized {os.path.basename(path)} from {w}x{h} to {new_w}x{new_h}")
            
            return image, True
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            return None, False
    
    def process_single_image(self, image_path: str):
        """Process single image - SEQUENTIAL PROCESSING ONLY"""
        try:
            # Load image
            image, success = self.load_image_efficiently(image_path)
            if not success:
                return {'path': os.path.basename(image_path), 'success': False, 'error': 'Failed to load'}
            
            start_time = time.time()
            
            # Process with memory management
            if self.optimize_memory:
                torch.cuda.empty_cache()  # Clear cache before processing
            
            # Process with full precision only (NO MIXED PRECISION)
            # Sequential processing - no threading issues
            masks = self.mask_generator.generate(image)
            
            # Clean up GPU memory after processing
            if self.optimize_memory:
                torch.cuda.empty_cache()
            
            process_time = time.time() - start_time
            
            # Convert annotations efficiently
            annotations = self.masks_to_annotations_fast(masks, image.shape)
            
            # Save JSON immediately after processing this image
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
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing {os.path.basename(image_path)}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return {
                'path': os.path.basename(image_path),
                'error': str(e),
                'success': False
            }
    
    def masks_to_annotations_fast(self, masks, image_shape):
        """Fast and stable mask to annotation conversion with robust RLE"""
        annotations = []
        h, w = image_shape[:2]
        
        for i, mask_data in enumerate(masks):
            try:
                # Extract mask data safely
                segmentation_mask = mask_data['segmentation']
                bbox = mask_data['bbox']
                area = mask_data['area']
                
                # Robust RLE encoding
                if HAS_PYCOCOTOOLS:
                    # Use pycocotools for reliable RLE
                    mask_uint8 = segmentation_mask.astype(np.uint8)
                    # Ensure proper memory layout for pycocotools
                    mask_fortran = np.asfortranarray(mask_uint8)
                    rle = mask_util.encode(mask_fortran)
                    # Decode bytes to string for JSON serialization
                    if isinstance(rle['counts'], bytes):
                        counts = rle['counts'].decode('utf-8')
                    else:
                        counts = rle['counts']
                else:
                    # Fixed numpy-based RLE encoding
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
                print(f"Warning: Failed to process mask {i}: {e}")
                continue
        
        return annotations
    
    def _robust_numpy_rle_encode(self, mask):
        """Robust RLE encoding that produces valid JSON-serializable output"""
        try:
            # Ensure mask is boolean/uint8
            mask = mask.astype(np.uint8)
            
            # Flatten in row-major order (C order)
            mask_flat = mask.flatten()
            
            # Find transitions
            diff_indices = np.where(np.diff(np.concatenate(([0], mask_flat, [0]))))[0]
            
            # Calculate run lengths
            run_lengths = np.diff(diff_indices)
            
            # Convert to standard RLE format
            # RLE format: [length_of_first_run_of_0s, length_of_first_run_of_1s, ...]
            rle_counts = []
            
            # Check if first pixel is 0 or 1
            if len(diff_indices) > 0:
                start_val = mask_flat[0] if len(mask_flat) > 0 else 0
                
                # If first pixel is 1, we need to add a 0-length run of 0s
                if start_val == 1:
                    rle_counts.append(0)
                
                # Add all run lengths
                for length in run_lengths:
                    rle_counts.append(int(length))
            else:
                # Entire mask is uniform
                if len(mask_flat) > 0 and mask_flat[0] == 1:
                    rle_counts = [0, len(mask_flat)]
                else:
                    rle_counts = [len(mask_flat)]
            
            return rle_counts
            
        except Exception as e:
            print(f"Error in RLE encoding: {e}")
            # Fallback: return empty mask
            return [mask.size]
    
    def _save_annotations_robust(self, image_path, annotations):
        """Robust JSON saving with validation - saves to sam subfolder"""
        try:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            json_path = os.path.join(self.sam_output_dir, f"{image_name}.json")
            
            # Create complete JSON structure
            output_data = {
                "image_name": image_name,
                "annotations": annotations,
                "num_masks": len(annotations)
            }
            
            # Write JSON with proper formatting and validation
            with open(json_path, 'w') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            # Validate the written JSON by trying to read it back
            try:
                with open(json_path, 'r') as f:
                    json.load(f)
                # Don't print success message for every image to reduce clutter
                return json_path
            except json.JSONDecodeError as e:
                print(f"✗ Invalid JSON written for {image_name}: {e}")
                return None
            
        except Exception as e:
            print(f"Error saving JSON for {os.path.basename(image_path)}: {e}")
            return None
    
    def get_image_paths(self, input_path: str):
        """Get image paths efficiently (case-insensitive)"""
        if os.path.isfile(input_path):
            return [input_path]

        if os.path.isdir(input_path):
            extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
            all_files = glob.glob(os.path.join(input_path, "*"))
            image_paths = [f for f in all_files if f.lower().split('.')[-1] in extensions]
            return sorted(image_paths)

        raise ValueError(f"Invalid input path: {input_path}")
    
    def process_images_sequential(self, image_paths):
        """Process images sequentially - most stable approach with incremental processing"""
        if not image_paths:
            print("No images found!")
            return
        
        print(f"Found {len(image_paths)} total images")
        
        # Filter images for processing (skip those with existing valid JSON files)
        images_to_process, images_already_processed = self.filter_images_for_processing(image_paths)
        
        print(f"\n📊 Processing Summary:")
        print(f"   Total images found: {len(image_paths)}")
        print(f"   Images already processed: {len(images_already_processed)}")
        print(f"   Images to process: {len(images_to_process)}")
        print(f"   Progress: {len(images_already_processed)}/{len(image_paths)} ({100*len(images_already_processed)/len(image_paths):.1f}%) complete")
        
        if len(images_to_process) == 0:
            print("\n✅ All images already have valid JSON files. Nothing to process!")
            return
        
        print(f"\nProcessing {len(images_to_process)} remaining images sequentially for maximum stability...")
        
        # Warmup with first image to process
        print("Warming up GPU...")
        if images_to_process:
            _, _ = self.load_image_efficiently(images_to_process[0])
            warmup_result = self.process_single_image(images_to_process[0])
            if warmup_result.get('success'):
                print(f"Warmup successful: {warmup_result['masks']} masks in {warmup_result['time']:.2f}s")
        
        all_results = []
        total_masks = 0
        total_pixels = 0
        successful_json_saves = 0
        
        overall_start = time.time()
        
        # Process images one by one with progress bar
        with tqdm(images_to_process, desc="Processing", unit="image") as pbar:
            for i, image_path in enumerate(images_to_process):
                try:
                    result = self.process_single_image(image_path)
                    all_results.append(result)
                    
                    if result.get('success', False):
                        total_masks += result['masks']
                        total_pixels += result['size'][0] * result['size'][1]
                        if result.get('json_path'):
                            successful_json_saves += 1
                        
                        # Update progress bar with current stats
                        current_total_processed = len(images_already_processed) + i + 1
                        pbar.set_postfix({
                            'FPS': f'{result["fps"]:.2f}',
                            'Masks': result['masks'],
                            'JSON': f'{successful_json_saves}/{i+1}',
                            'Total_Progress': f'{current_total_processed}/{len(image_paths)}',
                            'Avg_Time': f'{result["time"]:.1f}s'
                        })
                    else:
                        # Update progress bar even for failed images
                        current_total_processed = len(images_already_processed) + i + 1
                        pbar.set_postfix({
                            'Status': 'FAILED',
                            'JSON': f'{successful_json_saves}/{i+1}',
                            'Total_Progress': f'{current_total_processed}/{len(image_paths)}'
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Unexpected error processing {os.path.basename(image_path)}: {e}")
                    all_results.append({
                        'path': os.path.basename(image_path),
                        'error': str(e),
                        'success': False
                    })
                    pbar.update(1)
        
        overall_time = time.time() - overall_start
        
        # Print results
        successful_results = [r for r in all_results if r.get('success', False)]
        failed_results = [r for r in all_results if not r.get('success', False)]
        
        if successful_results:
            self._print_summary(successful_results, total_masks, total_pixels, 
                                     overall_time, successful_json_saves, 
                                     len(images_already_processed), len(image_paths))
        
        # Print failed images if any
        if failed_results:
            print(f"\n⚠️  Failed to process {len(failed_results)} images:")
            for result in failed_results[:10]:  # Show first 10 failures
                print(f"  - {result['path']}: {result.get('error', 'Unknown error')}")
            if len(failed_results) > 10:
                print(f"  ... and {len(failed_results) - 10} more")
    
    def _print_summary(self, results, total_masks, total_pixels, 
                            overall_time, successful_json_saves, 
                            images_already_processed, total_images):
        """Print processing summary"""
        times = [r['time'] for r in results]
        fps_values = [r['fps'] for r in results]
        
        print("\n" + "="*70)
        print("STABLE SAM PROCESSING SUMMARY")
        print("="*70)
        print(f"Mode: Sequential processing (most stable)")
        print(f"Mixed precision: DISABLED")
        print(f"Memory optimization: {self.optimize_memory}")
        print(f"Incremental processing: ENABLED")
        print(f"")
        print(f"Total images found: {total_images}")
        print(f"Images already processed (skipped): {images_already_processed}")
        print(f"Images processed this run: {len(results)}")
        print(f"Successful JSON saves this run: {successful_json_saves}/{len(results)}")
        print(f"")
        print(f"Processing time this run: {overall_time:.2f}s")
        if len(results) > 0:
            print(f"Throughput this run: {len(results)/overall_time:.2f} images/sec")
            print(f"Total masks generated this run: {total_masks:,}")
            print(f"Average FPS: {np.mean(fps_values):.2f}")
            print(f"Peak FPS: {np.max(fps_values):.2f}")
            print(f"Average time per image: {np.mean(times):.2f}s")
        
        # Overall completion status
        total_completed = images_already_processed + successful_json_saves
        completion_percentage = (total_completed / total_images) * 100 if total_images > 0 else 0
        print(f"")
        print(f"Overall completion: {total_completed}/{total_images} ({completion_percentage:.1f}%)")
        
        # GPU memory info
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")


def main():
    parser = argparse.ArgumentParser(description="Stable Sequential SAM Processing with Incremental Processing")
    parser.add_argument("--image_dir_path", required=True, help="Input images directory")
    parser.add_argument("--output_dir_path", required=True, help="Output directory for JSON")
    parser.add_argument("--model-type", choices=["vit_h", "vit_l", "vit_b"], 
                       default="vit_h", help="SAM model type")
    parser.add_argument("--checkpoint", default="./sam_vit_h_4b8939.pth", help="Model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--no-memory-opt", action='store_true', help="Disable memory optimizations")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir_path):
        print(f"Error: Input directory does not exist: {args.image_dir_path}")
        sys.exit(1)
    
    try:
        processor = StableSequentialSAMProcessor(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            gpu_id=args.gpu,
            optimize_memory=not args.no_memory_opt,
            output_dir=args.output_dir_path
        )
        
        image_paths = processor.get_image_paths(args.image_dir_path)
        processor.process_images_sequential(image_paths)
        
        print(f"\nProcessing complete! JSON files saved to: {processor.sam_output_dir}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()