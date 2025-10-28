#!/usr/bin/env python3
"""
Stable SAM Batch Processing - Sequential processing, no threading
Now supports incremental processing - skips images with existing JSON files
UPDATED: Supports classification dataset structure with class subfolders
"""

# python sam_infer.py --image_dir_path "/mnt/e/Desktop/AgML/datasets_sorted/classification/crop_disease" --output_dir_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/classification/crop_disease"
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
        Now supports classification dataset structure with class subfolders
        
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
        print(f"Dataset structure: Classification (class subfolders)")
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
    
    def check_existing_json(self, image_info: dict) -> bool:
        """
        Check if JSON file already exists for the given image
        
        Args:
            image_info: Dictionary containing 'path', 'class_name', and 'image_name'
            
        Returns:
            True if JSON file exists and is valid, False otherwise
        """
        if not self.sam_output_dir:
            return False
        
        try:
            class_name = image_info['class_name']
            image_name = image_info['image_name']
            
            # Create class-specific output directory path
            class_output_dir = os.path.join(self.sam_output_dir, class_name)
            json_path = os.path.join(class_output_dir, f"{image_name}.json")
            
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
                print(f"Warning: Corrupted JSON file found for {class_name}/{image_name}, will regenerate")
                return False
                
        except Exception as e:
            print(f"Error checking existing JSON for {image_info['class_name']}/{image_info['image_name']}: {e}")
            return False
    
    def filter_images_for_processing(self, image_infos):
        """
        Filter image info objects to only include those that need processing
        
        Args:
            image_infos: List of image info dictionaries
            
        Returns:
            Tuple of (images_to_process, images_already_processed)
        """
        images_to_process = []
        images_already_processed = []
        
        print("🔍 Checking for existing JSON files...")
        
        for image_info in tqdm(image_infos, desc="Checking existing", unit="image"):
            if self.check_existing_json(image_info):
                images_already_processed.append(image_info)
            else:
                images_to_process.append(image_info)
        
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
    
    def process_single_image(self, image_info: dict):
        """Process single image - SEQUENTIAL PROCESSING ONLY"""
        try:
            image_path = image_info['path']
            class_name = image_info['class_name']
            image_name = image_info['image_name']
            
            # Load image
            image, success = self.load_image_efficiently(image_path)
            if not success:
                return {
                    'path': f"{class_name}/{image_name}",
                    'class_name': class_name,
                    'success': False, 
                    'error': 'Failed to load'
                }
            
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
                json_path = self._save_annotations_robust(image_info, annotations)
            
            return {
                'path': f"{class_name}/{image_name}",
                'class_name': class_name,
                'size': (image.shape[1], image.shape[0]),
                'masks': len(masks),
                'time': process_time,
                'fps': 1.0 / process_time,
                'json_path': json_path,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing {image_info['class_name']}/{image_info['image_name']}: {e}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            return {
                'path': f"{image_info['class_name']}/{image_info['image_name']}",
                'class_name': image_info['class_name'],
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
    
    def _save_annotations_robust(self, image_info: dict, annotations):
        """Robust JSON saving with validation - saves to class-specific sam subfolder"""
        try:
            class_name = image_info['class_name']
            image_name = image_info['image_name']
            
            # Create class-specific output directory
            class_output_dir = os.path.join(self.sam_output_dir, class_name)
            os.makedirs(class_output_dir, exist_ok=True)
            
            json_path = os.path.join(class_output_dir, f"{image_name}.json")
            
            # Create complete JSON structure
            output_data = {
                "image_name": image_name,
                "class_name": class_name,
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
                print(f"❌ Invalid JSON written for {class_name}/{image_name}: {e}")
                return None
            
        except Exception as e:
            print(f"Error saving JSON for {image_info['class_name']}/{image_info['image_name']}: {e}")
            return None
    
    def get_image_paths(self, input_path: str):
        """Get image paths from classification dataset structure (class subfolders)"""
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path must be a directory for classification datasets: {input_path}")
        
        extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
        image_infos = []
        
        print(f"Scanning classification dataset structure in: {input_path}")
        
        # Get all subdirectories (class folders)
        class_folders = [d for d in os.listdir(input_path) 
                        if os.path.isdir(os.path.join(input_path, d))]
        
        if not class_folders:
            raise ValueError(f"No class subdirectories found in: {input_path}")
        
        print(f"Found {len(class_folders)} class folders: {', '.join(sorted(class_folders))}")
        
        # Process each class folder
        for class_name in sorted(class_folders):
            class_path = os.path.join(input_path, class_name)
            
            # Get all image files in this class folder
            all_files = glob.glob(os.path.join(class_path, "*"))
            class_images = [f for f in all_files 
                          if f.lower().split('.')[-1] in extensions]
            
            print(f"Class '{class_name}': {len(class_images)} images")
            
            # Create image info objects
            for image_path in sorted(class_images):
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                image_infos.append({
                    'path': image_path,
                    'class_name': class_name,
                    'image_name': image_name
                })
        
        print(f"Total images found: {len(image_infos)}")
        return image_infos
    
    def process_images_sequential(self, image_infos):
        """Process images sequentially - most stable approach with incremental processing"""
        if not image_infos:
            print("No images found!")
            return
        
        print(f"Found {len(image_infos)} total images")
        
        # Filter images for processing (skip those with existing valid JSON files)
        images_to_process, images_already_processed = self.filter_images_for_processing(image_infos)
        
        # Calculate per-class statistics
        classes_summary = {}
        for img_info in image_infos:
            class_name = img_info['class_name']
            if class_name not in classes_summary:
                classes_summary[class_name] = {'total': 0, 'processed': 0, 'to_process': 0}
            classes_summary[class_name]['total'] += 1
        
        for img_info in images_already_processed:
            classes_summary[img_info['class_name']]['processed'] += 1
            
        for img_info in images_to_process:
            classes_summary[img_info['class_name']]['to_process'] += 1
        
        print(f"\n📊 Processing Summary by Class:")
        for class_name, stats in sorted(classes_summary.items()):
            completion = (stats['processed'] / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {class_name}: {stats['processed']}/{stats['total']} ({completion:.1f}%) complete, {stats['to_process']} to process")
        
        print(f"\n📊 Overall Summary:")
        print(f"   Total images found: {len(image_infos)}")
        print(f"   Images already processed: {len(images_already_processed)}")
        print(f"   Images to process: {len(images_to_process)}")
        print(f"   Progress: {len(images_already_processed)}/{len(image_infos)} ({100*len(images_already_processed)/len(image_infos):.1f}%) complete")
        
        if len(images_to_process) == 0:
            print("\n✅ All images already have valid JSON files. Nothing to process!")
            return
        
        print(f"\nProcessing {len(images_to_process)} remaining images sequentially for maximum stability...")
        
        # Warmup with first image to process
        print("Warming up GPU...")
        if images_to_process:
            _, _ = self.load_image_efficiently(images_to_process[0]['path'])
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
            for i, image_info in enumerate(images_to_process):
                try:
                    result = self.process_single_image(image_info)
                    all_results.append(result)
                    
                    if result.get('success', False):
                        total_masks += result['masks']
                        total_pixels += result['size'][0] * result['size'][1]
                        if result.get('json_path'):
                            successful_json_saves += 1
                        
                        # Update progress bar with current stats
                        current_total_processed = len(images_already_processed) + i + 1
                        pbar.set_postfix({
                            'Class': result['class_name'][:8],
                            'FPS': f'{result["fps"]:.2f}',
                            'Masks': result['masks'],
                            'JSON': f'{successful_json_saves}/{i+1}',
                            'Total_Progress': f'{current_total_processed}/{len(image_infos)}',
                            'Avg_Time': f'{result["time"]:.1f}s'
                        })
                    else:
                        # Update progress bar even for failed images
                        current_total_processed = len(images_already_processed) + i + 1
                        pbar.set_postfix({
                            'Class': image_info['class_name'][:8],
                            'Status': 'FAILED',
                            'JSON': f'{successful_json_saves}/{i+1}',
                            'Total_Progress': f'{current_total_processed}/{len(image_infos)}'
                        })
                    
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Unexpected error processing {image_info['class_name']}/{image_info['image_name']}: {e}")
                    all_results.append({
                        'path': f"{image_info['class_name']}/{image_info['image_name']}",
                        'class_name': image_info['class_name'],
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
                                     len(images_already_processed), len(image_infos),
                                     classes_summary)
        
        # Print failed images if any
        if failed_results:
            print(f"\n⚠️  Failed to process {len(failed_results)} images:")
            for result in failed_results[:10]:  # Show first 10 failures
                print(f"  - {result['path']}: {result.get('error', 'Unknown error')}")
            if len(failed_results) > 10:
                print(f"  ... and {len(failed_results) - 10} more")
    
    def _print_summary(self, results, total_masks, total_pixels, 
                            overall_time, successful_json_saves, 
                            images_already_processed, total_images, classes_summary):
        """Print processing summary with class-specific information"""
        times = [r['time'] for r in results]
        fps_values = [r['fps'] for r in results]
        
        # Calculate per-class results for this run
        class_results = {}
        for result in results:
            class_name = result['class_name']
            if class_name not in class_results:
                class_results[class_name] = {'count': 0, 'masks': 0, 'time': 0}
            class_results[class_name]['count'] += 1
            class_results[class_name]['masks'] += result['masks']
            class_results[class_name]['time'] += result['time']
        
        print("\n" + "="*80)
        print("STABLE SAM PROCESSING SUMMARY - CLASSIFICATION DATASET")
        print("="*80)
        print(f"Mode: Sequential processing (most stable)")
        print(f"Mixed precision: DISABLED")
        print(f"Memory optimization: {self.optimize_memory}")
        print(f"Incremental processing: ENABLED")
        print(f"Dataset type: Classification (class subfolders)")
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
        
        # Per-class processing results
        if class_results:
            print(f"\n📊 Per-Class Processing Results (this run):")
            for class_name in sorted(class_results.keys()):
                stats = class_results[class_name]
                avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
                avg_masks = stats['masks'] / stats['count'] if stats['count'] > 0 else 0
                print(f"   {class_name}: {stats['count']} images, {stats['masks']} masks, "
                      f"{avg_time:.2f}s avg, {avg_masks:.1f} masks avg")
        
        # Overall completion status by class
        print(f"\n📊 Overall Completion Status by Class:")
        for class_name, stats in sorted(classes_summary.items()):
            total_completed = stats['processed'] + class_results.get(class_name, {}).get('count', 0)
            completion_percentage = (total_completed / stats['total'] * 100) if stats['total'] > 0 else 0
            print(f"   {class_name}: {total_completed}/{stats['total']} ({completion_percentage:.1f}%) complete")
        
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
    parser = argparse.ArgumentParser(description="Stable Sequential SAM Processing for Classification Datasets with Incremental Processing")
    parser.add_argument("--image_dir_path", required=True, help="Input dataset directory (containing class subfolders)")
    parser.add_argument("--output_dir_path", required=True, help="Output directory for JSON (will create sam/ and class subfolders)")
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
        
        image_infos = processor.get_image_paths(args.image_dir_path)
        processor.process_images_sequential(image_infos)
        
        print(f"\nProcessing complete! JSON files saved to: {processor.sam_output_dir}")
        print(f"Directory structure maintained: each class has its own subfolder")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()