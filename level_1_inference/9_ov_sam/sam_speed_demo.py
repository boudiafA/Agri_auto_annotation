#!/usr/bin/env python3
"""
SAM Batch Processing Demo - Enhanced with parallel batch processing capabilities
"""
import os
import time
import glob
import argparse
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from queue import Queue
import multiprocessing as mp

# Set CUDA debugging BEFORE importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Only use the first GPU
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import torch
import numpy as np
from tqdm import tqdm

# Check for segment_anything import
try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
except ImportError:
    print("Error: segment_anything not found. Install with: pip install segment-anything")
    sys.exit(1)


class SAMBatchProcessor:
    def __init__(self, model_type: str = "vit_h", checkpoint_path: str = None, 
                 gpu_id: int = 1, batch_size: int = 4, num_workers: int = None):
        """Initialize SAM Batch Processor"""
        self.model_type = model_type
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.num_workers = num_workers or min(batch_size, mp.cpu_count())
        self.device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        self.checkpoint_path = checkpoint_path
        
        print(f"Initializing SAM Batch Processor")
        print(f"Device: {self.device}")
        print(f"Model: {model_type}")
        print(f"Batch size: {batch_size}")
        print(f"Workers: {self.num_workers}")
        
        # Set GPU
        if torch.cuda.is_available():
            if gpu_id >= torch.cuda.device_count():
                print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
                self.gpu_id = 0
                self.device = f"cuda:0"
            torch.cuda.set_device(self.gpu_id)
        
        # Load model
        self.sam = self._load_model(checkpoint_path)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )
        
        print("Model loaded successfully!")
        
    def _load_model(self, checkpoint_path: str):
        """Load SAM model"""
        if checkpoint_path is None:
            checkpoint_paths = {
                "vit_h": "sam_vit_h_4b8939.pth",
                "vit_l": "sam_vit_l_0b3195.pth", 
                "vit_b": "sam_vit_b_01ec64.pth"
            }
            checkpoint_path = checkpoint_paths.get(self.model_type)
            
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        sam = sam_model_registry[self.model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        return sam
    
    def get_image_paths(self, input_path: str):
        """Get list of image paths"""
        if os.path.isfile(input_path):
            return [input_path]
        
        if os.path.isdir(input_path):
            extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']
            image_paths = []
            for ext in extensions:
                patterns = [f"*.{ext}", f"*.{ext.upper()}", f"*.{ext.capitalize()}"]
                for pattern in patterns:
                    image_paths.extend(glob.glob(os.path.join(input_path, pattern)))
            return sorted(list(set(image_paths)))
        
        raise ValueError(f"Invalid input path: {input_path}")
    
    def load_image(self, image_path: str):
        """Load image"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    def process_single_image(self, image_path: str):
        """Process a single image"""
        try:
            # Load image
            image = self.load_image(image_path)
            h, w = image.shape[:2]
            
            # Process
            start_time = time.time()
            masks = self.mask_generator.generate(image)
            process_time = time.time() - start_time
            
            return {
                'path': os.path.basename(image_path),
                'size': (w, h),
                'masks': len(masks),
                'time': process_time,
                'fps': 1.0 / process_time,
                'success': True
            }
            
        except Exception as e:
            return {
                'path': os.path.basename(image_path),
                'error': str(e),
                'success': False
            }
    
    def create_batches(self, image_paths):
        """Create batches from image paths"""
        batches = []
        for i in range(0, len(image_paths), self.batch_size):
            batch = image_paths[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    def process_batch_threading(self, batch):
        """Process a batch of images using threading"""
        batch_results = []
        
        def worker(image_path, results_queue):
            result = self.process_single_image(image_path)
            results_queue.put(result)
        
        # Create threads for batch
        threads = []
        results_queue = Queue()
        
        for image_path in batch:
            thread = threading.Thread(target=worker, args=(image_path, results_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        while not results_queue.empty():
            batch_results.append(results_queue.get())
        
        return batch_results
    
    def process_batch_multiprocessing(self, batch):
        """Process a batch of images using multiprocessing"""
        with ProcessPoolExecutor(max_workers=min(len(batch), self.num_workers)) as executor:
            batch_results = list(executor.map(self.process_single_image, batch))
        return batch_results
    
    def process_images_batch(self, image_paths, method='threading'):
        """Process images in batches"""
        if not image_paths:
            print("No images found!")
            return
            
        print(f"Found {len(image_paths)} images")
        print(f"Processing method: {method}")
        
        # Warmup
        print("Warming up...")
        warmup_image = self.load_image(image_paths[0])
        _ = self.mask_generator.generate(warmup_image)
        
        # Create batches
        batches = self.create_batches(image_paths)
        print(f"Created {len(batches)} batches of size {self.batch_size}")
        
        # Process batches
        all_results = []
        total_masks = 0
        total_pixels = 0
        
        # Progress tracking
        pbar = tqdm(batches, desc="Processing batches", unit="batch")
        
        overall_start = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            
            # Process batch based on method
            if method == 'threading':
                batch_results = self.process_batch_threading(batch)
            elif method == 'multiprocessing':
                batch_results = self.process_batch_multiprocessing(batch)
            else:  # sequential
                batch_results = [self.process_single_image(path) for path in batch]
            
            batch_time = time.time() - batch_start
            
            # Update progress and stats
            successful_results = [r for r in batch_results if r.get('success', False)]
            batch_masks = sum(r['masks'] for r in successful_results)
            batch_pixels = sum(r['size'][0] * r['size'][1] for r in successful_results)
            
            total_masks += batch_masks
            total_pixels += batch_pixels
            all_results.extend(batch_results)
            
            # Update progress bar
            batch_fps = len(successful_results) / batch_time if batch_time > 0 else 0
            pbar.set_postfix({
                'Batch': f'{batch_idx + 1}/{len(batches)}',
                'Success': f'{len(successful_results)}/{len(batch)}',
                'Batch FPS': f'{batch_fps:.2f}',
                'Masks': batch_masks,
                'Time': f'{batch_time:.2f}s'
            })
        
        pbar.close()
        overall_time = time.time() - overall_start
        
        # Print summary
        successful_results = [r for r in all_results if r.get('success', False)]
        if successful_results:
            self._print_batch_summary(successful_results, total_masks, total_pixels, overall_time, method)
        
        # Print errors if any
        failed_results = [r for r in all_results if not r.get('success', False)]
        if failed_results:
            print(f"\nFailed to process {len(failed_results)} images:")
            for result in failed_results[:5]:  # Show first 5 errors
                print(f"  {result['path']}: {result['error']}")
            if len(failed_results) > 5:
                print(f"  ... and {len(failed_results) - 5} more")
    
    def _print_batch_summary(self, results, total_masks, total_pixels, overall_time, method):
        """Print batch processing summary"""
        times = [r['time'] for r in results]
        fps_values = [r['fps'] for r in results]
        mask_counts = [r['masks'] for r in results]
        
        print("\n" + "="*70)
        print("BATCH PROCESSING SUMMARY")
        print("="*70)
        print(f"Processing method: {method}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of workers: {self.num_workers}")
        print(f"Images processed: {len(results)}")
        print(f"Total processing time: {overall_time:.2f}s")
        print(f"Overall throughput: {len(results)/overall_time:.2f} images/sec")
        print()
        print(f"Total masks: {total_masks:,}")
        print(f"Total pixels: {total_pixels:,}")
        print()
        print(f"Per-image stats:")
        print(f"  Average time: {np.mean(times):.3f}s")
        print(f"  Average FPS: {np.mean(fps_values):.2f}")
        print(f"  Min/Max FPS: {np.min(fps_values):.2f} / {np.max(fps_values):.2f}")
        print(f"  Average masks per image: {np.mean(mask_counts):.1f}")
        print(f"  Pixel throughput: {total_pixels/sum(times):,.0f} pixels/sec")
        
        # Efficiency comparison
        sequential_time = sum(times)
        efficiency = sequential_time / overall_time if overall_time > 0 else 1
        print(f"\nParallelization efficiency: {efficiency:.2f}x speedup")


def worker_init(model_type, checkpoint_path, gpu_id):
    """Initialize worker process with SAM model"""
    global worker_processor
    worker_processor = SAMBatchProcessor(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        gpu_id=gpu_id,
        batch_size=1
    )


def worker_process_image(image_path):
    """Worker function for multiprocessing"""
    return worker_processor.process_single_image(image_path)


def main():
    parser = argparse.ArgumentParser(description="SAM Batch Processing Demo")
    parser.add_argument("--input_path", 
                       default="/mnt/e/Desktop/GLaMM/detection_datasets/almond_bloom_2023/images", 
                       help="Path to input images")
    parser.add_argument("--model-type", choices=["vit_h", "vit_l", "vit_b"], 
                       default="vit_h", help="SAM model type")
    parser.add_argument("--checkpoint", default="./sam_vit_h_4b8939.pth", 
                       help="Path to model checkpoint")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--batch-size", type=int, default=12, 
                       help="Number of images to process in parallel")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker processes/threads")
    parser.add_argument("--method", choices=["sequential", "threading", "multiprocessing"],
                       default="threading", help="Processing method")
    
    args = parser.parse_args()
    
    try:
        # Initialize processor
        processor = SAMBatchProcessor(
            model_type=args.model_type,
            checkpoint_path=args.checkpoint,
            gpu_id=args.gpu,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        
        # Get image paths and process
        image_paths = processor.get_image_paths(args.input_path)
        processor.process_images_batch(image_paths, method=args.method)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()