# Example Usage:
# python infer_per_iNatAg.py --input_dir "/mnt/e/Desktop/AgML/datasets_sorted/iNatAg_subset" --output_dir "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/iNatAg_subset"


import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import numpy as np
from midas.model_loader import default_models, load_model
from ddp import *
import utils
import time
import os
import glob
from PIL import Image
import gc
import traceback

first_execution = True


# --- CustomImageDataset Class ---
class CustomImageDataset(Dataset):
    """
    A custom dataset to load images from a list of file paths.
    """
    def __init__(self, image_paths, transform):
        """
        Args:
            image_paths (list): List of paths to the images.
            transform: The transformation to apply to each image.
        """
        self.image_paths = image_paths
        self.transform = transform
        self.image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)
        
        try:
            # Open image and ensure it's RGB
            image = Image.open(image_path).convert("RGB")
            original_size = image.size
            
            # Check image size and warn about very large images
            if max(original_size) > 4096:
                print(f"Warning: Very large image {image_name} ({original_size[0]}x{original_size[1]})")
            
            # Apply transformation
            transformed_image = self.transform({"image": np.array(image)/255.0})["image"]
            
            # Return as dictionary to avoid collation issues
            return {
                'image_name': image_name,
                'image': torch.from_numpy(transformed_image),
                'width': original_size[0],
                'height': original_size[1]
            }
            
        except Exception as e:
            print(f"Warning: Could not load or process image {image_path}. Error: {e}")
            # Return dummy data to prevent dataloader from crashing
            return {
                'image_name': "error.jpg",
                'image': torch.zeros(3, 224, 224),
                'width': 224,
                'height': 224
            }


def parse_args():
    parser = argparse.ArgumentParser(description="Run MiDaS depth estimation on dataset directories.")
    
    parser.add_argument("--input_dir", required=True,
                        help="Path to the root directory containing dataset folders.")
    parser.add_argument("--output_dir", required=True,
                        help="Path to the root output directory to store the predictions.")

    parser.add_argument('-m', '--model_weights', default="/home/abood/groundingLMM/GranD/checkpoints/dpt_beit_large_512.pt",
                        help='Path to the trained weights of model')
    parser.add_argument('-t', '--model_type', default='dpt_beit_large_512', help='Model type')

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Add option to skip large images
    parser.add_argument('--max_image_size', default=8192, type=int, 
                        help='Skip images larger than this size in any dimension')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of CPU workers for data loading (0=single process, 4-8 recommended for multi-core)')
    
    args = parser.parse_args()
    return args


def cleanup_memory():
    """Clean up GPU and system memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def process(model, sample, target_size):
    """
    Run the inference and interpolate with OOM handling.
    """
    global first_execution

    try:
        if first_execution:
            height, width = sample.shape[2:]
            print(f"    Input resized to {width}x{height} before entering the encoder")
            first_execution = False

        sample = sample.to(next(model.parameters()).device)

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size[::-1],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        return prediction
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"    CUDA OOM Error during model inference: {e}")
        cleanup_memory()
        raise
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"    Runtime OOM Error during model inference: {e}")
            cleanup_memory()
            raise
        else:
            print(f"    Runtime Error during model inference: {e}")
            raise
    except Exception as e:
        print(f"    Unexpected error during model inference: {e}")
        raise


def run_inference(model, transform, image_paths, output_dir, args):
    """
    Process a specific list of images and save to a specific output directory.
    
    Args:
        model: The loaded MiDaS model.
        transform: The image transformation function.
        image_paths (list): A list of absolute paths to the images to process.
        output_dir (Path): The exact directory where output depth maps should be saved.
        args: Command-line arguments.
    """
    if not image_paths:
        print(f"  No images to process for {output_dir}. Skipping.")
        return

    # Create the output folder (e.g., .../dataset_name/midas/)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Check for already processed images to allow for resuming ---
    processed_stems = set()
    if os.path.exists(output_dir):
        try:
            existing_files = os.listdir(output_dir)
            # Handle the output naming pattern: "filename_min_X_max_Y.jpg"
            for filename in existing_files:
                if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Extract original stem from patterns like "image1_min_296.6020_max_12648.9307.jpg"
                    if '_min_' in filename and '_max_' in filename:
                        # Split at first occurrence of '_min_' to get the original name
                        original_name = filename.split('_min_')[0]
                        processed_stems.add(original_name)
                    else:
                        # Fallback for normal naming
                        processed_stems.add(Path(filename).stem)
            
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                print(f"  Found {len(processed_stems)} already processed images in output directory")
                if processed_stems:
                    # Show a few examples of detected processed files
                    sample_files = [f for f in existing_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
                    print(f"  Example output files: {sample_files}")
        except Exception as e:
            print(f"  Warning: Could not scan output directory {output_dir}: {e}")
            processed_stems = set()
    
    # Filter image paths to get input image stems
    valid_image_paths = []
    input_stems = []
    
    for img_path in image_paths:
        try:
            # Check if it's a valid image file
            if Path(img_path).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                valid_image_paths.append(img_path)
                input_stems.append(Path(img_path).stem)
        except Exception as e:
            print(f"  Warning: Error processing path {img_path}: {e}")
            continue
    
    # Calculate how many images need processing
    stems_to_process = set(input_stems) - processed_stems
    total_input_images = len(input_stems)
    already_processed = len(processed_stems.intersection(set(input_stems)))
    remaining_to_process = len(stems_to_process)
    
    if not torch.distributed.is_initialized() or args.local_rank == 0:
        print(f"  Total input images: {total_input_images}")
        print(f"  Already processed: {already_processed}")
        print(f"  Remaining to process: {remaining_to_process}")
        
        if remaining_to_process == 0:
            print(f"  All images already processed! Skipping...")
            return
    
    # --- Create Dataset and DataLoader for the current batch of images ---
    dataset = CustomImageDataset(valid_image_paths, transform)
    
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(dataset, rank=args.local_rank, shuffle=False)
    else:
        sampler = None

    # Configure DataLoader with proper worker settings
    dataloader_kwargs = {
        'batch_size': args.batch_size_per_gpu,
        'sampler': sampler,
        'num_workers': args.num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
    }
    
    # Add persistent_workers only if num_workers > 0
    if args.num_workers > 0:
        dataloader_kwargs['persistent_workers'] = True
        
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    start_time = time.time()
    
    # Counters for tracking
    total_processed = 0
    skipped_already_done = 0
    skipped_large = 0
    error_count = 0
    oom_count = 0

    # --- Main processing loop ---
    pbar = tqdm(dataloader, disable=(torch.distributed.is_initialized() and args.local_rank != 0), 
                desc="Processing images")
    
    for batch in pbar:
        # Extract from batch (batch_size is typically 1)
        image_name = batch['image_name'][0] if isinstance(batch['image_name'], list) else batch['image_name'][0]
        image = batch['image']
        width = batch['width'][0].item()
        height = batch['height'][0].item()
        
        image_stem = Path(image_name).stem
        
        # Skip if already processed
        if image_stem in processed_stems:
            skipped_already_done += 1
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue
            
        # Skip if there was a loading error
        if image_name == "error.jpg":
            error_count += 1
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue

        # Check image size and skip if too large
        size_tuple = (width, height)
        if max(size_tuple) > args.max_image_size:
            print(f"    Skipping very large image {image_name} ({width}x{height})")
            skipped_large += 1
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue

        try:
            with torch.no_grad():
                prediction = process(model, image, size_tuple)

            # Output depth map
            # Pass only the stem (without extension) to get output format: "filename_min_X_max_Y.jpg"
            # instead of "filename.jpg_min_X_max_Y.jpg"
            output_filename = os.path.join(output_dir, image_stem)
            utils.write_jpeg(output_filename, prediction.astype(np.float32))
            
            total_processed += 1
            
            # Update progress bar
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            
            # Clean up variables
            del prediction
            
            # Periodic memory cleanup
            if total_processed % 10 == 0:
                cleanup_memory()
                
        except torch.cuda.OutOfMemoryError as e:
            oom_count += 1
            print(f"    CUDA OOM Error processing {image_name}: {e}")
            print(f"    Image size: {width}x{height}")
            cleanup_memory()
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom_count += 1
                print(f"    OOM Runtime Error processing {image_name}: {e}")
                print(f"    Image size: {width}x{height}")
                cleanup_memory()
            else:
                error_count += 1
                print(f"    Runtime Error processing {image_name}: {e}")
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue
                
        except MemoryError as e:
            oom_count += 1
            print(f"    System Memory Error processing {image_name}: {e}")
            print(f"    Image size: {width}x{height}")
            cleanup_memory()
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue
            
        except Exception as e:
            error_count += 1
            print(f"    Unexpected error processing {image_name}: {e}")
            print(f"    Traceback: {traceback.format_exc()}")
            cleanup_memory()
            if not torch.distributed.is_initialized() or args.local_rank == 0:
                pbar.set_postfix({
                    'processed': total_processed, 
                    'skipped_done': skipped_already_done,
                    'skipped_large': skipped_large,
                    'oom': oom_count,
                    'errors': error_count
                })
            continue

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if not torch.distributed.is_initialized() or args.local_rank == 0:
        print(f"  Finished processing. Time taken: {elapsed_time:.2f} seconds.")
        print(f"  Summary: {total_processed} newly processed, {skipped_already_done} already done, {skipped_large} too large, {oom_count} OOM errors, {error_count} other errors")


if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)

    # --- Global Setup ---
    if not torch.distributed.is_initialized() or args.local_rank == 0:
        print("Starting MiDaS depth estimation process...")
        print(f"Input directory: {args.input_dir}")
        print(f"Output directory: {args.output_dir}")
        print(f"Max image size: {args.max_image_size}")

    # Set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Load model ONCE
    device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    
    try:
        model, transform, net_w, net_h = load_model(device, args.model_weights, args.model_type, optimize=False, height=None, square=False)
        
        if torch.distributed.is_initialized():
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # --- Main Logic: Iterate through dataset folders ---
    root_input_dir = Path(args.input_dir)
    root_output_dir = Path(args.output_dir)
    
    if not root_input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {root_input_dir}")

    # Iterate over dataset folders (sorted for consistent processing order)
    dataset_paths = sorted([p for p in root_input_dir.iterdir() if p.is_dir()])

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        if not torch.distributed.is_initialized() or args.local_rank == 0:
            print(f"\nProcessing dataset: {dataset_name}")

        try:
            # Collect all image files directly from the dataset folder
            image_paths = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                image_paths.extend(glob.glob(os.path.join(dataset_path, f'*{ext}')))
                image_paths.extend(glob.glob(os.path.join(dataset_path, f'*{ext.upper()}')))
            
            if not image_paths:
                if not torch.distributed.is_initialized() or args.local_rank == 0:
                    print(f"  No images found in {dataset_name}. Skipping.")
                continue
            
            # Create output directory: output_dir/dataset_name/midas/
            output_target_dir = root_output_dir / dataset_name / "midas"
            
            # Process all images in this dataset
            run_inference(model, transform, image_paths, output_target_dir, args)
            
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            continue
            
    if not torch.distributed.is_initialized() or args.local_rank == 0:
        print('\033[92m' + "\nAll datasets processed successfully." + '\033[0m')