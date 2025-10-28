# Batch inference script for multiple datasets - Optimized Version with Image Downscaling
# Usage: python infer_per_dataset.py --datasets_root_path "/mnt/e/Desktop/AgML/datasets_sorted/detection" --output_root_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection"

import argparse
import os
import glob
import json
import numpy as np
import torch
import traceback
import logging
from torch.utils.data import DataLoader, DistributedSampler
from utils import *  # Import everything from utils like infer.py does
from tqdm import tqdm
from ddp import *
from torch.utils.data._utils.collate import default_collate
from functools import lru_cache
from pathlib import Path
import torch.nn.functional as F

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_mdetr_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)

# Global device cache to avoid repeated device queries
_device_cache = None

# Maximum image dimension to prevent 32-bit indexing errors
MAX_IMAGE_DIMENSION = 1600  # Adjust this value as needed (1333, 1600, or 2000 are common)


def get_model_device(model):
    """Cache model device to avoid repeated parameter iteration"""
    global _device_cache
    if _device_cache is None:
        _device_cache = next(model.parameters()).device
    return _device_cache


def downscale_image_if_needed(image, image_size, max_dim=MAX_IMAGE_DIMENSION):
    """
    Downscale image if either dimension exceeds max_dim, preserving aspect ratio
    
    Args:
        image: Image tensor of shape (C, H, W) or (B, C, H, W)
        image_size: Original image size (width, height) - may not match tensor dims after transforms
        max_dim: Maximum allowed dimension
        
    Returns:
        tuple: (downscaled_image, new_image_size, scale_factor)
    """
    is_batched = image.dim() == 4
    if is_batched:
        _, _, h, w = image.shape
    else:
        _, h, w = image.shape
    
    # Get actual tensor dimensions (these are the real dimensions we need to check)
    actual_h, actual_w = h, w
    
    # Check if downscaling is needed based on ACTUAL tensor dimensions
    if actual_h <= max_dim and actual_w <= max_dim:
        return image, image_size, 1.0
    
    # Calculate scale factor to fit within max_dim
    scale = min(max_dim / actual_h, max_dim / actual_w)
    new_h = int(actual_h * scale)
    new_w = int(actual_w * scale)
    
    logger.info(f"Downscaling image from ({actual_w}x{actual_h}) to ({new_w}x{new_h}), scale: {scale:.3f}")
    
    # Downscale using bilinear interpolation
    if is_batched:
        downscaled = F.interpolate(
            image, 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        )
    else:
        downscaled = F.interpolate(
            image.unsqueeze(0), 
            size=(new_h, new_w), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(0)
    
    # Update image_size to match the downscaled dimensions
    # Scale the original image_size proportionally
    orig_w, orig_h = image_size
    new_image_size = (int(orig_w * scale), int(orig_h * scale))
    
    return downscaled, new_image_size, scale


def parse_args():
    parser = argparse.ArgumentParser(description="Batch MDETR Referring Expression Inference")

    parser.add_argument("--datasets_root_path", required=True, 
                       help="Root directory containing all dataset folders")
    parser.add_argument("--output_root_path", required=True,
                       help="Root directory to save all outputs")

    parser.add_argument("--ckpt_path", required=False, default="/home/abood/groundingLMM/GranD/outputs/refcocog_EB3_checkpoint.pth",
                        help="Specify the checkpoints path if you want to load from a local path.")

    # DDP related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Add debug level flag
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')
    
    # Add rerun flag to force reprocessing of existing files
    parser.add_argument('--rerun', action='store_true', 
                       help='Force rerun all files, even if output already exists')
    
    # Add max image dimension parameter
    parser.add_argument('--max_image_dim', type=int, default=1600,
                       help='Maximum image dimension (width or height) to prevent memory errors')

    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # Update global MAX_IMAGE_DIMENSION
    global MAX_IMAGE_DIMENSION
    MAX_IMAGE_DIMENSION = args.max_image_dim
    logger.info(f"Maximum image dimension set to: {MAX_IMAGE_DIMENSION}")
    
    return args


# Optimized JSON serialization with type checking optimization
def json_serializable(data):
    """Convert data to JSON serializable format - optimized version"""
    data_type = type(data)
    if data_type == np.float32:  # Direct type comparison is faster
        return round(float(data), 2)
    elif data_type == np.ndarray:
        return data.tolist()
    else:
        return data


def run_inference(model, image, image_size, nouns, phrases, device, threshold=0.5, max_dim=MAX_IMAGE_DIMENSION):
    """
    Run inference on the image with the given phrases - optimized version with downscaling
    
    Args:
        model: The MDETR model, on GPU
        image: Image tensor (currently on CPU)
        image_size: Original image size (width, height)
        nouns: List of nouns corresponding to phrases
        phrases: List of phrases to locate in the image
        device: Model device (pre-computed)
        threshold: Confidence threshold for detections
        max_dim: Maximum image dimension
        
    Returns:
        tuple: (image, all_nouns, all_phrase_boxes)
    """
    num_phrases = len(phrases)
    
    # Early return for empty phrases
    if num_phrases == 0:
        logger.warning("No phrases found to process")
        return image, [], {}
        
    logger.debug(f"Processing {num_phrases} phrases")
    
    # Downscale image if needed BEFORE moving to device
    original_image_size = image_size
    image, image_size, scale_factor = downscale_image_if_needed(image, image_size, max_dim)
    
    if scale_factor < 1.0:
        logger.info(f"Image downscaled by {scale_factor:.3f}x from {original_image_size} to {image_size}")
    
    # Move image to device only if necessary (optimization)
    if image.device != device:
        image = image.to(device, non_blocking=True)  # Use non_blocking for better performance
    
    # Repeat the image for each phrase
    image_b = image.repeat(num_phrases, 1, 1, 1)
    
    try:
        # Use torch.no_grad() context for inference (even though globally disabled, explicit is better)
        with torch.inference_mode():  # More optimized than no_grad for inference
            # Propagate through the model
            memory_cache = model(image_b, phrases, encode_and_save=True)
            outputs = model(image_b, phrases, encode_and_save=False, memory_cache=memory_cache)
        
        # Optimized processing with pre-allocated structures
        all_phrase_boxes = {}
        all_nouns = []
        
        # Process outputs more efficiently
        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        
        for i in range(num_phrases):
            probas = 1 - pred_logits.softmax(-1)[i, :, -1].cpu()
            keep = (probas > threshold).cpu()
            
            if keep.any():  # Only process if there are valid detections
                # Convert boxes from [0; 1] to image scales
                # Use the downscaled image_size for proper rescaling
                bboxes_scaled = rescale_bboxes(pred_boxes.cpu()[i, keep], image_size)
                bboxes_list = bboxes_scaled.cpu().numpy().tolist()
                if bboxes_list:
                    # If image was downscaled, scale bboxes back to original size
                    if scale_factor < 1.0:
                        bbox = bboxes_list[0]
                        bbox = [coord / scale_factor for coord in bbox]
                        all_phrase_boxes[phrases[i]] = bbox
                    else:
                        all_phrase_boxes[phrases[i]] = bboxes_list[0]
                    all_nouns.append(nouns[i])
                
        logger.debug(f"Found {len(all_phrase_boxes)} valid detections")
        return image, all_nouns, all_phrase_boxes
        
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        logger.error(traceback.format_exc())
        raise


@lru_cache(maxsize=128)
def detect_dataset_type(dataset_path):
    """
    Detect if dataset is detection type (has 'images' folder) or classification type (has class folders)
    Cached for repeated calls on same path
    """
    dataset_path = Path(dataset_path)
    
    # Use pathlib for more efficient directory operations
    exclude_dirs = {'visualizations', 'annotations', 'blip2', 'llava'}
    folders = [f.name for f in dataset_path.iterdir() 
               if f.is_dir() and f.name not in exclude_dirs]
    
    if 'images' in folders:
        return 'detection', []
    else:
        # Check if folders contain images (classification structure)
        class_folders = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for folder in folders:
            folder_path = dataset_path / folder
            # More efficient: check if any file has image extension
            if any(file.suffix.lower() in image_extensions for file in folder_path.iterdir() if file.is_file()):
                class_folders.append(folder)
        
        if class_folders:
            return 'classification', class_folders
        else:
            logger.warning(f"No valid structure found in {dataset_path}")
            return None, []


def get_dataset_image_names(dataset):
    """
    Safely extract image names from dataset, trying different possible attributes
    """
    # Try common attribute names
    possible_attrs = ['image_list', 'image_names', 'images', 'filenames', 'file_list', 'samples']
    
    for attr in possible_attrs:
        if hasattr(dataset, attr):
            attr_value = getattr(dataset, attr)
            if attr_value:
                logger.debug(f"Found image names using attribute: {attr}")
                return attr_value
    
    # If no direct attribute found, try to get names by iterating through dataset
    logger.warning("No direct image list attribute found, extracting names through iteration")
    image_names = []
    try:
        for i in range(len(dataset)):
            # Try to get the image name from the dataset item
            item = dataset[i]
            if isinstance(item, (tuple, list)) and len(item) > 0:
                # Assume first element might be the image name or path
                image_name = item[0]
                if isinstance(image_name, str):
                    # Extract just the filename if it's a full path
                    image_name = os.path.basename(image_name)
                    image_names.append(image_name)
                else:
                    # Fallback: generate name based on index
                    image_names.append(f"image_{i:06d}.jpg")
            else:
                # Fallback: generate name based on index
                image_names.append(f"image_{i:06d}.jpg")
    except Exception as e:
        logger.error(f"Error extracting image names: {str(e)}")
        # Final fallback: generate sequential names
        image_names = [f"image_{i:06d}.jpg" for i in range(len(dataset))]
    
    return image_names


def check_existing_outputs(output_dir_path, model_name, image_names):
    """
    Check which images already have output files - optimized with batch operations
    """
    model_output_dir = Path(output_dir_path) / model_name
    
    # Batch check using pathlib
    existing_files = []
    missing_files = []
    
    # Pre-compute all expected paths
    expected_paths = [(name, model_output_dir / f"{name[:-4]}.json") for name in image_names]
    
    for image_name, output_path in expected_paths:
        if output_path.exists():
            existing_files.append(image_name)
        else:
            missing_files.append(image_name)
    
    total = len(image_names)
    existing_count = len(existing_files)
    
    stats = {
        'total': total,
        'existing': existing_count,
        'missing': len(missing_files),
        'existing_percentage': (existing_count / total * 100) if total else 0
    }
    
    return existing_files, missing_files, stats


def load_model(args, use_ddp):
    """
    Load the MDETR model once and return it - extracted from original run function
    """
    # First, load the model without pre-trained weights
    try:
        model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_refcocog',
                                            pretrained=False, return_postprocessor=True)
        logger.info("Successfully loaded model architecture")
    except Exception as e:
        logger.error(f"Error loading model architecture: {str(e)}")
        return None
    
    # Download weights from Zenodo if needed
    checkpoint_path = args.ckpt_path
    if checkpoint_path == "None":
        # Use the RefCOCOg EfficientNet-B3 checkpoint from Zenodo
        url = "https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth?download=1"
        # Define a local path to save the downloaded weights
        checkpoint_path = os.path.join(args.output_root_path, "refcocog_EB3_checkpoint.pth")
        
        # Download if the file doesn't exist
        if not os.path.exists(checkpoint_path):
            logger.info(f"Downloading checkpoint from {url} to {checkpoint_path}")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, checkpoint_path)
                logger.info("Checkpoint downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading checkpoint: {str(e)}")
                return None
        else:
            logger.info(f"Using cached checkpoint at {checkpoint_path}")
    else:
        logger.info(f"Using provided checkpoint at {checkpoint_path}")
    
    # Load the weights with strict=False to ignore unexpected keys
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=False)
        logger.info("Checkpoint loaded successfully")
        del checkpoint  # Free memory immediately
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None

    try:
        model = model.cuda()
        logger.debug(f"Model moved to CUDA device: {get_model_device(model)}")
    except Exception as e:
        logger.error(f"Error moving model to CUDA: {str(e)}")
        return None
        
    model.eval()
    
    if use_ddp:
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            logger.info(f"Model wrapped with DDP on device {args.local_rank}")
        except Exception as e:
            logger.error(f"Error setting up DDP: {str(e)}")
            return None
    
    return model


def run(args, dataloader, model, dataset, model_name="mdetr-re"):
    """
    Run inference on dataloader using pre-loaded model - optimized version
    """
    output_dir = Path(args.output_dir_path) / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all image names from dataset (not dataloader) to check existing outputs
    all_image_names = get_dataset_image_names(dataset)
    
    # Check existing outputs
    existing_files, missing_files, stats = check_existing_outputs(
        args.output_dir_path, model_name, all_image_names
    )
    
    # Log resume information
    logger.info(f"Output check for {args.output_dir_path}/{model_name}:")
    logger.info(f"  Total images: {stats['total']}")
    logger.info(f"  Already processed: {stats['existing']} ({stats['existing_percentage']:.1f}%)")
    logger.info(f"  To be processed: {stats['missing']}")
    
    if args.rerun:
        logger.info("RERUN mode enabled: will reprocess all files, including existing ones")
        skipped_count = 0
    else:
        if stats['existing'] > 0:
            logger.info("RESUME mode: skipping existing files")
        skipped_count = stats['existing']
    
    # If all files are already processed and not in rerun mode, skip entirely
    if not args.rerun and stats['missing'] == 0:
        logger.info("All files already processed. Skipping dataloader iteration.")
        return 0, skipped_count, 0
    
    # Pre-compute device for efficiency
    device = get_model_device(model)
    
    # Convert existing files to set for O(1) lookup
    existing_files_set = set(name[:-4] for name in existing_files) if not args.rerun else set()
    
    # Processing loop - optimized version
    processed_count = 0
    error_count = 0
    actual_skipped = 0
    
    # Pre-allocate for memory efficiency
    memory_log_frequency = 100  # Log memory less frequently
    
    for batch_idx, (image_name, image, image_size, captions, nouns, phrases) in enumerate(
        tqdm(dataloader, desc="Processing images")):
        
        # Extract batch data
        image_name = image_name[0]
        image_size = image_size[0]
        captions = captions[0]
        nouns = nouns[0]
        phrases = phrases[0]
        
        # Quick skip check using set lookup (O(1))
        image_base = image_name[:-4]
        if not args.rerun and image_base in existing_files_set:
            logger.debug(f"Skipping {image_name}, output already exists")
            actual_skipped += 1
            continue
            
        # Check if phrases list is empty
        if len(phrases) == 0:
            logger.warning(f"No phrases extracted for image: {image_name}, skipping")
            continue
            
        try:
            # Log GPU memory less frequently for performance
            if args.debug and torch.cuda.is_available() and batch_idx % memory_log_frequency == 0:
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                logger.debug(f"GPU memory allocated: {mem_allocated:.2f} GB")
            
            _, all_nouns, all_phrase_boxes = run_inference(
                model, image, image_size, nouns, phrases, device, max_dim=MAX_IMAGE_DIMENSION)
            
            logger.debug(f"Inference successful, found {len(all_phrase_boxes)} boxes")
            processed_count += 1
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA out of memory while processing image: {image_name}")
                logger.error(f"Image size: {image_size}, Number of phrases: {len(phrases)}")
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    logger.error(f"GPU memory - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
                # Try to free some memory
                torch.cuda.empty_cache()
            elif "canUse32BitIndexMath" in str(e):
                logger.error(f"Image too large for 32-bit indexing: {image_name}")
                logger.error(f"Image size: {image_size}. Consider reducing --max_image_dim parameter (current: {MAX_IMAGE_DIMENSION})")
            else:
                logger.error(f"RuntimeError processing image: {image_name}. Error: {str(e)}")
            logger.error(traceback.format_exc())
            error_count += 1
            continue
        except Exception as e:
            logger.error(f"Error processing image: {image_name}. Error: {str(e)}")
            logger.error(traceback.format_exc())
            error_count += 1
            continue

        # Optimized data structure building
        image_data = {
            image_name: {
                model_name: [
                    {
                        'bbox': [round(float(b), 2) for b in all_phrase_boxes[phrase]],
                        'label': noun,
                        'phrase': phrase
                    }
                    for j, (phrase, noun) in enumerate(zip(all_phrase_boxes.keys(), all_nouns))
                ]
            }
        }

        # Write output file
        output_file_path = output_dir / f"{image_base}.json"
        try:
            with open(output_file_path, 'w') as f:
                json.dump(image_data, f, separators=(',', ':'))  # Compact JSON for smaller files
            logger.debug(f"Successfully saved results for {image_name}")
        except Exception as e:
            logger.error(f"Error saving results for {image_name}: {str(e)}")
            error_count += 1
    
    # Log processing summary
    logger.info(f"Processing completed for {args.output_dir_path}/{model_name}:")
    logger.info(f"  Successfully processed: {processed_count}")
    logger.info(f"  Skipped (already existed): {actual_skipped}")
    logger.info(f"  Errors encountered: {error_count}")
    
    return processed_count, actual_skipped, error_count


def custom_collate_fn(batch):
    """Custom collate function - optimized version"""
    # Use list comprehension for better performance
    return (
        [item[0] for item in batch],  # image_names
        default_collate([item[1] for item in batch]),  # images
        [item[2] for item in batch],  # image_sizes
        [item[3] for item in batch],  # captions
        [item[4] for item in batch],  # nouns
        [item[5] for item in batch]   # phrases
    )


def process_single_dataset_class(args, model, dataset_name, class_name, image_dir_path, 
                                blip2_pred_path, llava_pred_path, output_dir_path):
    """
    Process a single dataset/class combination - optimized version
    """
    logger.info(f"Processing dataset: {dataset_name}, class: {class_name if class_name else 'N/A'}")
    logger.info(f"Image directory: {image_dir_path}")
    logger.info(f"Output directory: {output_dir_path}")
    
    # Create dataset using the same class as infer.py
    try:
        image_dataset = CustomImageDataset(
            image_dir_path, blip2_pred_path, llava_pred_path, transform
        )
        logger.info(f"Created dataset with {len(image_dataset)} images")
        
        if len(image_dataset) == 0:
            logger.warning(f"No images found in {image_dir_path}, skipping")
            return 0, 0, 0
            
    except Exception as e:
        logger.error(f"Error creating dataset for {dataset_name}/{class_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return 0, 0, 0

    # Quick check: if not in rerun mode and all files exist, skip dataloader creation
    if not args.rerun:
        model_output_dir = Path(output_dir_path) / "mdetr-re"
        if model_output_dir.exists():
            # Batch check all files at once using the helper function
            image_names = get_dataset_image_names(image_dataset)
            expected_files = [
                model_output_dir / f"{name[:-4]}.json"
                for name in image_names
            ]
            
            if all(f.exists() for f in expected_files):
                logger.info(f"All {len(image_dataset)} images already processed. Skipping dataloader creation.")
                return 0, len(image_dataset), 0

    # Create dataloader using the same approach as infer.py
    if args.distributed:
        sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
        logger.info(f"Using distributed sampler with rank {args.local_rank}")
    else:
        sampler = None

    try:
        image_dataloader = DataLoader(
            image_dataset,
            batch_size=int(args.batch_size_per_gpu),
            num_workers=4,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=custom_collate_fn,
            pin_memory=True,  # Optimize GPU transfers
            persistent_workers=True  # Keep workers alive between epochs
        )
        logger.info(f"Created dataloader with batch size {args.batch_size_per_gpu}")
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        logger.error(traceback.format_exc())
        return 0, 0, 0

    # Set output directory for run function
    args.output_dir_path = output_dir_path
    
    # Run inference using the pre-loaded model
    return run(args, image_dataloader, model, image_dataset)


def main():
    args = parse_args()
    init_distributed_mode(args)

    # Determine if distributed mode is active
    use_ddp = args.distributed

    if not use_ddp:
        logger.info("Running in single GPU mode. Skipping distributed setup.")

    # Log rerun mode status
    if args.rerun:
        logger.info("RERUN MODE: Will reprocess all files, including existing outputs")
    else:
        logger.info("RESUME MODE: Will skip files that already have outputs")

    # Create output root directory
    os.makedirs(args.output_root_path, exist_ok=True)

    # Load model once for all datasets
    logger.info("Loading MDETR model...")
    model = load_model(args, use_ddp)
    if model is None:
        logger.error("Failed to load model, exiting")
        return
    
    logger.info("Model loaded successfully, will be reused for all datasets")

    # Get all dataset folders using pathlib for better performance
    datasets_root = Path(args.datasets_root_path)
    dataset_folders = [f.name for f in datasets_root.iterdir() if f.is_dir()]
    
    logger.info(f"Found {len(dataset_folders)} datasets: {dataset_folders}")

    # Initialize overall statistics
    total_processed = 0
    total_skipped = 0
    total_errors = 0

    # Process each dataset
    for dataset_name in dataset_folders:
        dataset_path = datasets_root / dataset_name
        output_dataset_path = Path(args.output_root_path) / dataset_name
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*50}")
        
        # Detect dataset type (cached)
        dataset_type, class_folders = detect_dataset_type(str(dataset_path))
        
        if dataset_type is None:
            logger.warning(f"Skipping dataset {dataset_name}: no valid structure found")
            continue
            
        logger.info(f"Dataset type: {dataset_type}")
        
        if dataset_type == 'detection':
            # Detection dataset: single images folder
            image_dir = dataset_path / 'images'
            blip2_pred_path = output_dataset_path / 'blip2'
            llava_pred_path = output_dataset_path / 'llava'
            
            # Check if prediction paths exist
            if not blip2_pred_path.exists():
                logger.warning(f"BLIP2 predictions not found at {blip2_pred_path}, skipping dataset {dataset_name}")
                continue
            if not llava_pred_path.exists():
                logger.warning(f"LLaVA predictions not found at {llava_pred_path}, skipping dataset {dataset_name}")
                continue
                
            processed, skipped, errors = process_single_dataset_class(
                args, model, dataset_name, None, str(image_dir), 
                str(blip2_pred_path), str(llava_pred_path), str(output_dataset_path))
            
            total_processed += processed
            total_skipped += skipped
            total_errors += errors
            
        elif dataset_type == 'classification':
            # Classification dataset: multiple class folders
            logger.info(f"Found classes: {class_folders}")
            
            for class_name in class_folders:
                image_dir = dataset_path / class_name
                blip2_pred_path = output_dataset_path / class_name / 'blip2'
                llava_pred_path = output_dataset_path / class_name / 'llava'
                class_output_path = output_dataset_path / class_name
                
                # Check if prediction paths exist for this specific class
                if not blip2_pred_path.exists():
                    logger.warning(f"BLIP2 predictions not found at {blip2_pred_path}, skipping class {class_name}")
                    continue
                if not llava_pred_path.exists():
                    logger.warning(f"LLaVA predictions not found at {llava_pred_path}, skipping class {class_name}")
                    continue
                    
                processed, skipped, errors = process_single_dataset_class(
                    args, model, dataset_name, class_name, str(image_dir), 
                    str(blip2_pred_path), str(llava_pred_path), str(class_output_path))
                
                total_processed += processed
                total_skipped += skipped
                total_errors += errors

    # Final summary
    logger.info("\n" + "="*50)
    logger.info("BATCH INFERENCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Total images processed: {total_processed}")
    logger.info(f"Total images skipped (already existed): {total_skipped}")
    logger.info(f"Total errors encountered: {total_errors}")
    logger.info(f"Mode used: {'RERUN' if args.rerun else 'RESUME'}")
    logger.info(f"Max image dimension used: {MAX_IMAGE_DIMENSION}")
    logger.info("="*50)


if __name__ == "__main__":
    main()