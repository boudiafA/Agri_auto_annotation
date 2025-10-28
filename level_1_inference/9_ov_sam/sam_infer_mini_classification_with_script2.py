#!/usr/bin/env python3
"""
Parallel SAM Batch Processing - GPU and CPU parallelization optimized
Supports incremental processing and classification dataset structure
OPTIMIZED: Multi-GPU support, batching, and parallel I/O operations
"""

import os
import time
import glob
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import threading
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import gc

# GPU settings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import torch
import torch.nn as nn
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


@dataclass
class ImageInfo:
    """Image information container"""
    path: str
    class_name: str
    image_name: str
    size: Optional[Tuple[int, int]] = None


@dataclass
class ProcessingResult:
    """Processing result container"""
    image_info: ImageInfo
    success: bool
    masks: int = 0
    time: float = 0.0
    json_path: Optional[str] = None
    error: Optional[str] = None
    annotations: Optional[List] = None  # Store the actual annotations


class ParallelSAMProcessor:
    def __init__(self, model_type: str = "vit_h", checkpoint_path: str = None,
                 gpu_ids: List[int] = None, batch_size: int = 4, 
                 num_workers: int = None, output_dir: str = None,
                 max_image_size: int = 1600):
        """
        Parallel SAM processor with multi-GPU and CPU optimization
        
        Args:
            gpu_ids: List of GPU IDs to use (None for auto-detect)
            batch_size: Number of images to process in parallel per GPU
            num_workers: Number of CPU workers for I/O (None for auto)
            max_image_size: Maximum image dimension for GPU memory optimization
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.max_image_size = max_image_size
        self.output_dir = output_dir
        
        # Auto-detect GPU configuration
        if gpu_ids is None:
            if torch.cuda.is_available():
                self.gpu_ids = list(range(torch.cuda.device_count()))
            else:
                self.gpu_ids = []
        else:
            self.gpu_ids = gpu_ids
        
        # Set number of CPU workers
        if num_workers is None:
            self.num_workers = min(mp.cpu_count(), 8)
        else:
            self.num_workers = num_workers
        
        # Create output directory
        if self.output_dir:
            self.sam_output_dir = os.path.join(self.output_dir, "sam")
            os.makedirs(self.sam_output_dir, exist_ok=True)
        else:
            self.sam_output_dir = None
        
        # Initialize models on each GPU
        self.models = {}
        self.mask_generators = {}
        
        print(f"Parallel SAM Processor")
        print(f"GPUs: {self.gpu_ids if self.gpu_ids else 'CPU only'}")
        print(f"Batch size per GPU: {self.batch_size}")
        print(f"CPU workers: {self.num_workers}")
        print(f"Max image size: {self.max_image_size}")
        
        if self.gpu_ids:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize SAM models on all GPUs"""
        for gpu_id in self.gpu_ids:
            print(f"Loading model on GPU {gpu_id}...")
            
            # Load model
            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=f"cuda:{gpu_id}")
            sam.eval()
            
            # Enable optimizations
            sam = torch.compile(sam, mode="reduce-overhead")
            
            # Create mask generator
            mask_generator = SamAutomaticMaskGenerator(
                model=sam,
                points_per_side=24,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=0,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
                points_per_batch=64  # Reduced for memory efficiency
            )
            
            self.models[gpu_id] = sam
            self.mask_generators[gpu_id] = mask_generator
    
    def load_image_batch(self, image_infos: List[ImageInfo]) -> List[Tuple[ImageInfo, np.ndarray]]:
        """Load a batch of images in parallel"""
        def load_single_image(image_info: ImageInfo):
            try:
                image = cv2.imread(image_info.path)
                if image is None:
                    return image_info, None
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize if too large
                h, w = image.shape[:2]
                if max(h, w) > self.max_image_size:
                    scale = self.max_image_size / max(h, w)
                    new_h, new_w = int(h * scale), int(w * scale)
                    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                
                image_info.size = (image.shape[1], image.shape[0])
                return image_info, image
            except Exception as e:
                print(f"Error loading {image_info.path}: {e}")
                return image_info, None
        
        # Use ThreadPoolExecutor for I/O bound image loading
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(load_single_image, image_infos))
        
        # Filter out failed loads
        return [(info, img) for info, img in results if img is not None]
    
    def process_batch_on_gpu(self, batch_data: List[Tuple[ImageInfo, np.ndarray]], 
                           gpu_id: int) -> List[ProcessingResult]:
        """Process a batch of images on specific GPU"""
        results = []
        mask_generator = self.mask_generators[gpu_id]
        
        for image_info, image in batch_data:
            try:
                start_time = time.time()
                
                # Process on GPU
                with torch.cuda.device(gpu_id):
                    masks = mask_generator.generate(image)
                
                process_time = time.time() - start_time
                
                # Convert annotations
                annotations = self.masks_to_annotations_fast(masks, image.shape)
                
                result = ProcessingResult(
                    image_info=image_info,
                    success=True,
                    masks=len(masks),
                    time=process_time,
                    annotations=annotations  # Store the annotations here
                )
                
                results.append(result)
                
            except Exception as e:
                results.append(ProcessingResult(
                    image_info=image_info,
                    success=False,
                    error=str(e),
                    annotations=[]
                ))
        
        return results
    
    def save_annotations_parallel(self, results: List[ProcessingResult]):
        """Save annotations to JSON files in parallel"""
        def save_single_annotation(result):
            try:
                if not self.sam_output_dir or not result.success:
                    return result
                
                image_info = result.image_info
                annotations = result.annotations or []  # Use annotations from result
                
                class_output_dir = os.path.join(self.sam_output_dir, image_info.class_name)
                os.makedirs(class_output_dir, exist_ok=True)
                
                json_path = os.path.join(class_output_dir, f"{image_info.image_name}.json")
                
                output_data = {
                    "image_name": image_info.image_name,
                    "class_name": image_info.class_name,
                    "annotations": annotations,
                    "num_masks": len(annotations)
                }
                
                with open(json_path, 'w') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                
                # Validate JSON
                with open(json_path, 'r') as f:
                    json.load(f)
                
                result.json_path = json_path
                return result
                
            except Exception as e:
                result.error = f"JSON save error: {e}"
                result.success = False
                return result
        
        # Use ThreadPoolExecutor for I/O bound JSON saving
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            return list(executor.map(save_single_annotation, results))
    
    def masks_to_annotations_fast(self, masks, image_shape):
        """Fast mask to annotation conversion"""
        annotations = []
        h, w = image_shape[:2]
        
        for mask_data in masks:
            try:
                segmentation_mask = mask_data['segmentation']
                bbox = mask_data['bbox']
                area = mask_data['area']
                
                # RLE encoding
                if HAS_PYCOCOTOOLS:
                    mask_uint8 = segmentation_mask.astype(np.uint8)
                    mask_fortran = np.asfortranarray(mask_uint8)
                    rle = mask_util.encode(mask_fortran)
                    counts = rle['counts'].decode('utf-8') if isinstance(rle['counts'], bytes) else rle['counts']
                else:
                    counts = self._numpy_rle_encode(segmentation_mask)
                
                annotation = {
                    "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                    "segmentation": {"size": [int(h), int(w)], "counts": counts},
                    "area": int(area),
                    "predicted_iou": float(mask_data.get('predicted_iou', 0.0)),
                    "stability_score": float(mask_data.get('stability_score', 0.0))
                }
                
                annotations.append(annotation)
                
            except Exception as e:
                print(f"Warning: Failed to process mask: {e}")
                continue
        
        return annotations
    
    def _numpy_rle_encode(self, mask):
        """Optimized RLE encoding"""
        mask_flat = mask.flatten().astype(np.uint8)
        diff_indices = np.where(np.diff(np.concatenate(([0], mask_flat, [0]))))[0]
        run_lengths = np.diff(diff_indices)
        
        rle_counts = []
        if len(diff_indices) > 0:
            if len(mask_flat) > 0 and mask_flat[0] == 1:
                rle_counts.append(0)
            rle_counts.extend([int(length) for length in run_lengths])
        else:
            rle_counts = [len(mask_flat)] if len(mask_flat) > 0 else [0]
        
        return rle_counts
    
    def check_existing_json(self, image_info: ImageInfo) -> bool:
        """Check if JSON file already exists"""
        if not self.sam_output_dir:
            return False
        
        try:
            class_output_dir = os.path.join(self.sam_output_dir, image_info.class_name)
            json_path = os.path.join(class_output_dir, f"{image_info.image_name}.json")
            
            if not os.path.exists(json_path):
                return False
            
            # Validate JSON
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            return isinstance(data, dict) and 'annotations' in data and isinstance(data['annotations'], list)
            
        except Exception:
            return False
    
    def filter_images_for_processing(self, image_infos: List[ImageInfo]) -> Tuple[List[ImageInfo], List[ImageInfo]]:
        """Filter images for processing using parallel checking"""
        def check_single_image(image_info):
            return image_info, self.check_existing_json(image_info)
        
        # Use parallel processing for checking existing files
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(check_single_image, image_infos))
        
        images_to_process = []
        images_already_processed = []
        
        for image_info, exists in results:
            if exists:
                images_already_processed.append(image_info)
            else:
                images_to_process.append(image_info)
        
        return images_to_process, images_already_processed
    
    def get_image_paths(self, input_path: str) -> List[ImageInfo]:
        """Get image paths from classification dataset structure"""
        if not os.path.isdir(input_path):
            raise ValueError(f"Input path must be a directory: {input_path}")
        
        extensions = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp'}
        image_infos = []
        
        class_folders = [d for d in os.listdir(input_path) 
                        if os.path.isdir(os.path.join(input_path, d))]
        
        if not class_folders:
            raise ValueError(f"No class subdirectories found in: {input_path}")
        
        for class_name in sorted(class_folders):
            class_path = os.path.join(input_path, class_name)
            all_files = glob.glob(os.path.join(class_path, "*"))
            class_images = [f for f in all_files 
                          if f.lower().split('.')[-1] in extensions]
            
            for image_path in sorted(class_images):
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                image_infos.append(ImageInfo(
                    path=image_path,
                    class_name=class_name,
                    image_name=image_name
                ))
        
        return image_infos
    
    def process_images_parallel(self, image_infos: List[ImageInfo]):
        """Process images using parallel GPU and CPU operations"""
        if not image_infos:
            print("No images found!")
            return
        
        print(f"Found {len(image_infos)} total images")
        
        # Filter images for processing
        images_to_process, images_already_processed = self.filter_images_for_processing(image_infos)
        
        print(f"Images already processed: {len(images_already_processed)}")
        print(f"Images to process: {len(images_to_process)}")
        
        if not images_to_process:
            print("✅ All images already processed!")
            return
        
        if not self.gpu_ids:
            print("❌ No GPUs available for processing!")
            return
        
        # Process in batches across multiple GPUs
        total_processed = 0
        all_results = []
        
        # Create batches for each GPU
        num_gpus = len(self.gpu_ids)
        effective_batch_size = self.batch_size * num_gpus
        
        with tqdm(total=len(images_to_process), desc="Processing", unit="image") as pbar:
            for i in range(0, len(images_to_process), effective_batch_size):
                batch_images = images_to_process[i:i + effective_batch_size]
                
                # Load images in parallel
                batch_data = self.load_image_batch(batch_images)
                
                if not batch_data:
                    pbar.update(len(batch_images))
                    continue
                
                # Distribute batch across GPUs
                gpu_batches = []
                for gpu_idx, gpu_id in enumerate(self.gpu_ids):
                    start_idx = gpu_idx * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(batch_data))
                    if start_idx < len(batch_data):
                        gpu_batches.append((batch_data[start_idx:end_idx], gpu_id))
                
                # Process on multiple GPUs in parallel
                batch_results = []
                
                with ThreadPoolExecutor(max_workers=len(self.gpu_ids)) as executor:
                    gpu_futures = {
                        executor.submit(self.process_batch_on_gpu, gpu_batch, gpu_id): (gpu_batch, gpu_id)
                        for gpu_batch, gpu_id in gpu_batches
                    }
                    
                    for future in as_completed(gpu_futures):
                        gpu_results = future.result()
                        batch_results.extend(gpu_results)
                
                # Save annotations in parallel (only for successful results)
                successful_results = [r for r in batch_results if r.success]
                if successful_results:
                    updated_results = self.save_annotations_parallel(successful_results)
                    # Update the batch_results with the updated results that have json_path set
                    result_map = {id(r): r for r in updated_results}
                    for i, result in enumerate(batch_results):
                        if result.success and id(result) in result_map:
                            batch_results[i] = result_map[id(result)]
                
                all_results.extend(batch_results)
                total_processed += len(batch_data)
                
                # Update progress bar
                successful = sum(1 for r in batch_results if r.success)
                avg_time = np.mean([r.time for r in batch_results if r.success]) if successful > 0 else 0
                avg_masks = np.mean([r.masks for r in batch_results if r.success]) if successful > 0 else 0
                
                pbar.set_postfix({
                    'Success': f'{successful}/{len(batch_results)}',
                    'Avg_Time': f'{avg_time:.2f}s',
                    'Avg_Masks': f'{avg_masks:.1f}',
                    'GPUs': len(self.gpu_ids)
                })
                pbar.update(len(batch_images))
                
                # Memory cleanup
                if torch.cuda.is_available():
                    for gpu_id in self.gpu_ids:
                        with torch.cuda.device(gpu_id):
                            torch.cuda.empty_cache()
                gc.collect()
        
        # Print summary
        successful_results = [r for r in all_results if r.success]
        self._print_parallel_summary(successful_results, len(images_already_processed), len(image_infos))
    
    def _print_parallel_summary(self, results, images_already_processed, total_images):
        """Print processing summary"""
        if not results:
            print("No successful results to summarize.")
            return
        
        times = [r.time for r in results]
        total_masks = sum(r.masks for r in results)
        overall_time = sum(times)
        
        print("\n" + "="*80)
        print("PARALLEL SAM PROCESSING SUMMARY")
        print("="*80)
        print(f"GPUs used: {self.gpu_ids}")
        print(f"Batch size per GPU: {self.batch_size}")
        print(f"CPU workers: {self.num_workers}")
        print(f"")
        print(f"Total images: {total_images}")
        print(f"Already processed: {images_already_processed}")
        print(f"Processed this run: {len(results)}")
        print(f"Total masks generated: {total_masks:,}")
        print(f"")
        print(f"Processing time: {overall_time:.2f}s")
        print(f"Throughput: {len(results)/overall_time:.2f} images/sec")
        print(f"Average time per image: {np.mean(times):.2f}s")
        print(f"Peak processing speed: {1/np.min(times):.2f} images/sec")
        
        if torch.cuda.is_available():
            for gpu_id in self.gpu_ids:
                memory_used = torch.cuda.memory_allocated(gpu_id) / 1024**3
                print(f"GPU {gpu_id} memory used: {memory_used:.2f}GB")


class BatchDatasetProcessor:
    """Handles batch processing of multiple datasets with parallelization"""
    
    def __init__(self, base_input_dir: str, base_output_dir: str, 
                 model_type: str = "vit_h", checkpoint: str = None, 
                 gpu_ids: List[int] = None, batch_size: int = 4, num_workers: int = None):
        self.base_input_dir = base_input_dir
        self.base_output_dir = base_output_dir
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.gpu_ids = gpu_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def get_dataset_folders(self):
        """Get list of dataset folders to process"""
        if not os.path.exists(self.base_input_dir):
            raise ValueError(f"Base input directory does not exist: {self.base_input_dir}")
        
        dataset_folders = [item for item in os.listdir(self.base_input_dir)
                          if os.path.isdir(os.path.join(self.base_input_dir, item))]
        
        if not dataset_folders:
            raise ValueError(f"No dataset folders found in: {self.base_input_dir}")
        
        return sorted(dataset_folders)
    
    def process_all_datasets(self):
        """Process all datasets"""
        dataset_folders = self.get_dataset_folders()
        print(f"Found {len(dataset_folders)} datasets to process")
        
        for dataset_name in dataset_folders:
            print(f"\n{'='*60}")
            print(f"Processing dataset: {dataset_name}")
            print(f"{'='*60}")
            
            try:
                image_dir = os.path.join(self.base_input_dir, dataset_name)
                output_dir = os.path.join(self.base_output_dir, dataset_name)
                os.makedirs(output_dir, exist_ok=True)
                
                processor = ParallelSAMProcessor(
                    model_type=self.model_type,
                    checkpoint_path=self.checkpoint,
                    gpu_ids=self.gpu_ids,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    output_dir=output_dir
                )
                
                image_infos = processor.get_image_paths(image_dir)
                processor.process_images_parallel(image_infos)
                
                print(f"✅ Completed dataset: {dataset_name}")
                
            except Exception as e:
                print(f"❌ Error processing dataset {dataset_name}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Parallel SAM Processing with GPU and CPU optimization")
    
    # Mode selection
    parser.add_argument("--batch_mode", action='store_true', help="Process multiple datasets")
    parser.add_argument("--base_input_dir", help="Base input directory (batch mode)")
    parser.add_argument("--base_output_dir", help="Base output directory (batch mode)")
    parser.add_argument("--image_dir_path", help="Input dataset directory (single mode)")
    parser.add_argument("--output_dir_path", help="Output directory (single mode)")
    
    # Model configuration
    parser.add_argument("--model-type", choices=["vit_h", "vit_l", "vit_b"], 
                       default="vit_h", help="SAM model type")
    parser.add_argument("--checkpoint", default="./sam_vit_h_4b8939.pth", help="Model checkpoint")
    
    # Parallelization settings
    parser.add_argument("--gpu-ids", nargs='+', type=int, help="GPU IDs to use (default: all available)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, help="Number of CPU workers (default: auto)")
    parser.add_argument("--max-image-size", type=int, default=1600, help="Maximum image dimension")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.batch_mode:
        if not args.base_input_dir or not args.base_output_dir:
            print("Error: --base_input_dir and --base_output_dir required for batch mode")
            sys.exit(1)
    else:
        if not args.image_dir_path or not args.output_dir_path:
            print("Error: --image_dir_path and --output_dir_path required for single mode")
            sys.exit(1)
    
    try:
        if args.batch_mode:
            print("🚀 Starting PARALLEL BATCH PROCESSING...")
            batch_processor = BatchDatasetProcessor(
                base_input_dir=args.base_input_dir,
                base_output_dir=args.base_output_dir,
                model_type=args.model_type,
                checkpoint=args.checkpoint,
                gpu_ids=args.gpu_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers
            )
            batch_processor.process_all_datasets()
        else:
            print("🚀 Starting PARALLEL PROCESSING...")
            processor = ParallelSAMProcessor(
                model_type=args.model_type,
                checkpoint_path=args.checkpoint,
                gpu_ids=args.gpu_ids,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                output_dir=args.output_dir_path,
                max_image_size=args.max_image_size
            )
            
            image_infos = processor.get_image_paths(args.image_dir_path)
            processor.process_images_parallel(image_infos)
            
            print(f"\n✅ Processing complete! Results: {processor.sam_output_dir}")
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()