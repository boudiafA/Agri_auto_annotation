# Multi-Dataset GRIT Inference Script
# (grand_env_3) groundingLMM/GranD/level_1_inference/8_grit python infer_per_dataset.py --datasets_root_path "/mnt/e/Desktop/AgML/datasets_sorted/detection" --output_root_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection"

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import os
import time
import tempfile
import shutil
from PIL import Image
from torch.utils.data import DataLoader, DistributedSampler
from ddp import *
from models.grit_src.image_dense_captions import setup_grit

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys

# Add CenterNet2 to the Python path
centernet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../environments/CenterNet2'))
sys.path.insert(0, centernet_path)

# Global variable to track temp directories for cleanup
temp_directories = []

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_root_path", required=True, help="Root path containing all dataset folders")
    parser.add_argument("--output_root_path", required=True, help="Root path to save all outputs")
    parser.add_argument("--max_size", required=False, default=None, type=int, 
                       help="Maximum image size (width or height). Images larger than this will be scaled down while maintaining aspect ratio")
    parser.add_argument("--redo", action="store_true", default=False,
                       help="Force reprocessing of all images, even if output files already exist")

    parser.add_argument("--opts", required=False, default="")
    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args

def json_serializable(data):
    if isinstance(data, np.float32):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def dense_pred_dict(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    scores = predictions["instances"].scores if predictions["instances"].has("scores") else None
    object_description = predictions["instances"].pred_object_descriptions.data

    prediction_list = []
    for i in range(len(object_description)):
        bbox = [round(float(a), 2) for a in boxes[i].tensor.cpu().detach().numpy()[0]]
        score = round(float(scores[i]), 2) if scores is not None else None

        prediction_dict = {
            'bbox': bbox,
            'score': score,
            'description': object_description[i],
        }
        prediction_list.append(prediction_dict)

    return prediction_list

def discover_datasets(datasets_root_path):
    """
    Discover all datasets and their structure.
    Returns: List of tuples (dataset_name, dataset_type, image_paths_info)
    """
    datasets_info = []
    datasets_root = Path(datasets_root_path)
    
    # Ignore these folder names
    ignore_folders = {'visualizations', 'annotations'}
    
    for dataset_path in datasets_root.iterdir():
        if not dataset_path.is_dir():
            continue
            
        dataset_name = dataset_path.name
        print(f"Analyzing dataset: {dataset_name}")
        
        # Check if it's a detection dataset (has 'images' folder)
        images_folder = dataset_path / 'images'
        if images_folder.exists() and images_folder.is_dir():
            # Detection dataset
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_files.extend(images_folder.glob(ext))
                image_files.extend(images_folder.glob(ext.upper()))
            
            if image_files:
                datasets_info.append((dataset_name, 'detection', {'images': image_files}))
                print(f"  └─ Detection dataset with {len(image_files)} images")
            continue
        
        # Check if it's a classification dataset (has class folders)
        class_folders = []
        for subfolder in dataset_path.iterdir():
            if (subfolder.is_dir() and 
                subfolder.name not in ignore_folders):
                # Check if this folder contains images
                image_files = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                    image_files.extend(subfolder.glob(ext))
                    image_files.extend(subfolder.glob(ext.upper()))
                
                if image_files:
                    class_folders.append((subfolder.name, image_files))
        
        if class_folders:
            # Classification dataset
            class_info = {class_name: images for class_name, images in class_folders}
            total_images = sum(len(images) for images in class_info.values())
            datasets_info.append((dataset_name, 'classification', class_info))
            print(f"  └─ Classification dataset with {len(class_folders)} classes, {total_images} total images")
            for class_name, images in class_folders:
                print(f"      ├─ {class_name}: {len(images)} images")
    
    return datasets_info

def create_temp_image_directory(image_paths, max_size=None):
    """
    Create a temporary directory with images (resized if needed) that works with the original CustomImageDataset.
    Returns the temp directory path for cleanup.
    """
    global temp_directories
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix="grit_images_")
    temp_directories.append(temp_dir)
    
    print(f"Creating temporary directory: {temp_dir}")
    
    for i, image_path in enumerate(image_paths):
        try:
            # Open and check image size
            with Image.open(image_path) as img:
                original_width, original_height = img.size
                
                # Determine if resizing is needed
                needs_resize = False
                if max_size is not None:
                    max_dim = max(original_width, original_height)
                    if max_dim > max_size:
                        needs_resize = True
                        scale_factor = max_size / max_dim
                        new_width = int(original_width * scale_factor)
                        new_height = int(original_height * scale_factor)
                
                # Create destination path in temp directory
                base_name = os.path.basename(image_path)
                dest_path = os.path.join(temp_dir, base_name)
                
                if needs_resize:
                    # Resize and save
                    img_resized = img.convert('RGB').resize((new_width, new_height), Image.Resampling.LANCZOS)
                    img_resized.save(dest_path)
                    
                    if i == 0:  # Print info for first resized image
                        print(f"Image resized: {original_width}x{original_height} -> {new_width}x{new_height} (max_size: {max_size})")
                else:
                    # Just copy the original image
                    img.convert('RGB').save(dest_path)
        
        except Exception as e:
            print(f"Warning: Could not process image {image_path}: {e}")
            # Try to copy original file as fallback
            try:
                import shutil
                dest_path = os.path.join(temp_dir, os.path.basename(image_path))
                shutil.copy2(image_path, dest_path)
            except Exception as e2:
                print(f"Warning: Could not copy original image {image_path}: {e2}")
                continue
    
    return temp_dir

def cleanup_temp_directories():
    """Clean up any temporary directories created for resizing"""
    global temp_directories
    for temp_dir in temp_directories:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up {temp_dir}: {e}")
    temp_directories = []

def process_dataset(dataset_name, dataset_type, dataset_info, output_root_path, model, args):
    """Process a single dataset"""
    model_name = 'grit'
    
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name} (type: {dataset_type})")
    print(f"{'='*60}")
    
    try:
        if dataset_type == 'detection':
            # Detection dataset processing
            image_paths = dataset_info['images']
            output_dir = Path(output_root_path) / dataset_name / model_name
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Debug: print output_dir type and value
            print(f"Debug - output_dir type: {type(output_dir)}, value: {output_dir}")
            
            # Check already processed - with error handling (skip if redo flag is set)
            remaining_images = []
            if args.redo:
                print("--redo flag set: Reprocessing all images")
                remaining_images = list(image_paths)
            else:
                processed_files = set()
                try:
                    if output_dir.exists():
                        processed_files = set(os.listdir(str(output_dir)))  # Ensure string conversion
                except Exception as e:
                    print(f"Warning: Could not list directory {output_dir}: {e}")
                    processed_files = set()
                
                # Filter out already processed images
                for img_path in image_paths:
                    try:
                        expected_output = f"{Path(img_path).stem}.json"
                        if expected_output not in processed_files:
                            remaining_images.append(img_path)
                    except Exception as e:
                        print(f"Warning: Could not process path {img_path}: {e}")
                        continue
            
            if not remaining_images:
                print(f"All images in {dataset_name} already processed. Skipping.")
                return
            
            if args.redo:
                print(f"Reprocessing all {len(remaining_images)} images...")
            else:
                print(f"Processing {len(remaining_images)} remaining images...")
            
            # Create temporary directory with images for the original CustomImageDataset
            temp_image_dir = create_temp_image_directory(remaining_images, max_size=args.max_size)
            dataset = CustomImageDataset(temp_image_dir)
            dataloader = DataLoader(
                dataset,
                batch_size=args.batch_size_per_gpu,
                num_workers=min(4, os.cpu_count() // 2),
                shuffle=False,
                pin_memory=(device == "cuda"),
                persistent_workers=True if min(4, os.cpu_count() // 2) > 0 else False,
                prefetch_factor=2
            )
            
            # Process images
            process_images_batch(dataloader, model, output_dir, model_name)
            
        elif dataset_type == 'classification':
            # Classification dataset processing
            for class_name, image_paths in dataset_info.items():
                print(f"\nProcessing class: {class_name}")
                
                output_dir = Path(output_root_path) / dataset_name / class_name / model_name
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Debug: print output_dir type and value
                print(f"Debug - output_dir type: {type(output_dir)}, value: {output_dir}")
                
                # Check already processed - with error handling (skip if redo flag is set)
                remaining_images = []
                if args.redo:
                    print(f"--redo flag set: Reprocessing all images in class {class_name}")
                    remaining_images = list(image_paths)
                else:
                    processed_files = set()
                    try:
                        if output_dir.exists():
                            processed_files = set(os.listdir(str(output_dir)))  # Ensure string conversion
                    except Exception as e:
                        print(f"Warning: Could not list directory {output_dir}: {e}")
                        processed_files = set()
                    
                    # Filter out already processed images
                    for img_path in image_paths:
                        try:
                            expected_output = f"{Path(img_path).stem}.json"
                            if expected_output not in processed_files:
                                remaining_images.append(img_path)
                        except Exception as e:
                            print(f"Warning: Could not process path {img_path}: {e}")
                            continue
                
                if not remaining_images:
                    print(f"All images in class {class_name} already processed. Skipping.")
                    continue
                
                if args.redo:
                    print(f"Reprocessing all {len(remaining_images)} images in class {class_name}...")
                else:
                    print(f"Processing {len(remaining_images)} remaining images in class {class_name}...")
                
                # Create temporary directory with images for the original CustomImageDataset
                temp_image_dir = create_temp_image_directory(remaining_images, max_size=args.max_size)
                dataset = CustomImageDataset(temp_image_dir)
                dataloader = DataLoader(
                    dataset,
                    batch_size=args.batch_size_per_gpu,
                    num_workers=min(4, os.cpu_count() // 2),
                    shuffle=False,
                    pin_memory=(device == "cuda"),
                    persistent_workers=True if min(4, os.cpu_count() // 2) > 0 else False,
                    prefetch_factor=2
                )
                
                # Process images
                process_images_batch(dataloader, model, output_dir, model_name)
                
    except Exception as e:
        print(f"Error in process_dataset for {dataset_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def process_images_batch(dataloader, model, output_dir, model_name):
    """Process a batch of images through the model"""
    batch_results = {}
    images_processed_in_batch = 0
    write_frequency = 20
    
    for batch_data in tqdm(dataloader, desc="Processing images"):
        image_names, images, heights, widths = batch_data
        
        # Move batch to GPU
        if device == "cuda":
            images = images.to(device, non_blocking=True)
        
        # Process each image in the batch
        for i in range(len(image_names)):
            image_name = image_names[i]
            inputs = {"image": images[i], "height": int(heights[i]), "width": int(widths[i])}
            
            with torch.no_grad():
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        predictions = model([inputs])[0]
                else:
                    predictions = model([inputs])[0]
                
                # Store results
                image_data = {model_name: dense_pred_dict(predictions)}
                batch_results[image_name] = {image_name: image_data}
                images_processed_in_batch += 1
        
        # Write batch periodically
        if images_processed_in_batch >= write_frequency:
            write_batch_results(batch_results, output_dir)
            batch_results = {}
            images_processed_in_batch = 0
            
            # Clear GPU cache
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # Write remaining results
    if batch_results:
        write_batch_results(batch_results, output_dir)

def write_batch_results(batch_results, output_dir):
    """Write batch results to JSON files"""
    for image_name, image_data in batch_results.items():
        # Remove extension and add .json
        base_name = Path(image_name).stem
        output_file = output_dir / f"{base_name}.json"
        
        with open(output_file, 'w') as f:
            json.dump(image_data, f, default=json_serializable, indent=2)

def main():
    args = parse_args()
    init_distributed_mode(args)
    
    # Set up threading
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(8, os.cpu_count()))
    
    # Create output root directory
    Path(args.output_root_path).mkdir(parents=True, exist_ok=True)
    
    # Discover all datasets
    print("Discovering datasets...")
    datasets_info = discover_datasets(args.datasets_root_path)
    
    if not datasets_info:
        print("No datasets found!")
        return
    
    print(f"\nFound {len(datasets_info)} datasets to process")
    
    # Show processing options
    if args.redo:
        print("⚡ --redo flag set: Will reprocess ALL images (ignoring existing outputs)")
    else:
        print("📄 Resume mode: Will skip already processed images")
        
    if args.max_size:
        print(f"🖼️  Image size limit: {args.max_size}px (larger images will be scaled down)")
    else:
        print("🖼️  No image size limit (using original image sizes)")
    
    # Initialize model once
    print("\nInitializing GRIT model...")
    dense_caption_demo = setup_grit(device=device)
    dense_caption_model = dense_caption_demo.predictor.model
    
    # Model optimizations
    dense_caption_model.eval()
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    
    if args.local_rank == -1 or args.world_size == 1:
        model = dense_caption_demo.predictor.model.to(device)
    else:
        model = torch.nn.parallel.DistributedDataParallel(
            dense_caption_demo.predictor.model, 
            device_ids=[args.local_rank]
        )
    
    # Process each dataset
    start_time = time.time()
    
    try:
        for dataset_name, dataset_type, dataset_info in datasets_info:
            try:
                process_dataset(dataset_name, dataset_type, dataset_info, args.output_root_path, model, args)
            except Exception as e:
                print(f"Error processing dataset {dataset_name}: {str(e)}")
                continue
    
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
    finally:
        # Always clean up temporary directories
        cleanup_temp_directories()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\n{'='*60}")
        print('\033[92m' + f"✓ Processing completed!" + '\033[0m')
        print('\033[92m' + f"✓ Total time taken: {elapsed_time:.2f} seconds" + '\033[0m')
        print(f"{'='*60}")

if __name__ == "__main__":
    main()