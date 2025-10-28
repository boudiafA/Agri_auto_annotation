import numpy as np
from PIL import ImageDraw, Image, ImageFont
from tqdm import tqdm
import os
import time
import argparse
import torch
import torch.nn.functional as F
import json
from mmdet.registry import MODELS
from mmengine import Config, print_log
from mmengine.structures import InstanceData
from ext.class_names.lvis_list import LVIS_CLASSES

LVIS_NAMES = LVIS_CLASSES
class_names = ['person', 'helmet', 'motorbike', 'jacket']
class_names = LVIS_NAMES
model_cfg = Config.fromfile('app/configs/sam_r50x16_fpn.py')

model = MODELS.build(model_cfg.model)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device=device)
model = model.eval()
model.init_weights()

mean = torch.tensor([123.675, 116.28, 103.53], device=device)[:, None, None]
std = torch.tensor([58.395, 57.12, 57.375], device=device)[:, None, None]
IMG_SIZE = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="Open Vocabulary SAM with Batch Processing for Multiple Datasets")

    parser.add_argument("--datasets_dir", required=True, 
                        help="Path to directory containing multiple dataset folders")
    parser.add_argument("--output_dir", required=True,
                        help="Path to directory where outputs will be saved")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images to process in each batch")
    parser.add_argument("--bbox_batch_size", type=int, default=32,
                        help="Number of bounding boxes to process in each inference batch")

    args = parser.parse_args()
    return args


def is_image_file(filename):
    """Check if file is an image"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    return os.path.splitext(filename.lower())[1] in image_extensions


def scan_dataset_structure(dataset_path):
    """
    Determine if dataset is detection-style (has 'images' folder) or 
    classification-style (has class subfolders)
    Returns: ('detection', images_list) or ('classification', {class: images_list})
    """
    if not os.path.exists(dataset_path):
        return None, None
    
    items = os.listdir(dataset_path)
    
    # Check if there's an 'images' folder (detection style)
    if 'images' in items:
        images_dir = os.path.join(dataset_path, 'images')
        if os.path.isdir(images_dir):
            images = []
            for img_file in os.listdir(images_dir):
                if is_image_file(img_file):
                    images.append(('images', img_file))  # (subfolder, filename)
            return 'detection', images
    
    # Check for classification style (class subfolders)
    class_data = {}
    for item in items:
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item not in ['visualizations', 'annotations']:
            images = []
            for img_file in os.listdir(item_path):
                if is_image_file(img_file):
                    images.append((item, img_file))  # (class_name, filename)
            if images:  # Only add if there are images
                class_data[item] = images
    
    if class_data:
        return 'classification', class_data
    
    return None, None


def batch_extract_img_feat(images):
    """Extract features for a batch of images"""
    batch_tensors = []
    original_sizes = []
    numpy_images = []
    
    for img in images:
        w, h = img.size
        original_sizes.append((w, h))
        
        # Convert image to RGB if it has an alpha channel
        if img.mode in ['RGBA', 'LA'] or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGB')
        
        scale = IMG_SIZE / max(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)
        img_numpy = np.array(img)
        numpy_images.append(img_numpy)
        
        img_tensor = torch.tensor(img_numpy, device=device, dtype=torch.float32).permute((2, 0, 1))
        img_tensor = (img_tensor - mean) / std
        img_tensor = F.pad(img_tensor, (0, IMG_SIZE - new_w, 0, IMG_SIZE - new_h), 'constant', 0)
        batch_tensors.append(img_tensor)
    
    try:
        batch_tensor = torch.stack(batch_tensors)
        feat_dict = model.extract_feat(batch_tensor)
        
        # Move features to device if needed
        if feat_dict is not None:
            for k in feat_dict:
                if isinstance(feat_dict[k], torch.Tensor):
                    feat_dict[k] = feat_dict[k].to(device)
                elif isinstance(feat_dict[k], tuple):
                    feat_dict[k] = tuple(v.to(device) for v in feat_dict[k])
                    
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print_log(f"CUDA OOM during feature extraction! Reduce batch_size", logger='current')
            return None, None, None
        else:
            raise
            
    return numpy_images, feat_dict, original_sizes


def batch_run_inference(img_feats, bboxes_list, img_indices):
    """Run inference on a batch of bounding boxes"""
    if not bboxes_list:
        return [], [], []
    
    try:
        # Prepare batch tensors
        batch_bboxes = torch.stack([torch.tensor(bbox, dtype=torch.float32, device=device) for bbox in bboxes_list])
        
        # Create batch prompts - we need to handle multiple images
        # Group by image index to create proper batch structure
        unique_img_indices = list(set(img_indices))
        all_masks = []
        all_cls_preds = []
        all_valid_indices = []
        
        for img_idx in unique_img_indices:
            # Get bboxes for this image
            img_bbox_indices = [i for i, idx in enumerate(img_indices) if idx == img_idx]
            img_bboxes = batch_bboxes[img_bbox_indices]
            
            if len(img_bboxes) == 0:
                continue
                
            # Extract features for this specific image
            img_feat = {}
            for k in img_feats:
                if isinstance(img_feats[k], torch.Tensor):
                    img_feat[k] = img_feats[k][img_idx:img_idx+1]  # Keep batch dimension
                elif isinstance(img_feats[k], tuple):
                    img_feat[k] = tuple(v[img_idx:img_idx+1] for v in img_feats[k])
            
            prompts = InstanceData(bboxes=img_bboxes)
            masks, cls_pred = model.extract_masks(img_feat, prompts)
            
            all_masks.extend([masks[i] for i in range(len(img_bboxes))])
            all_cls_preds.extend([cls_pred[i] for i in range(len(img_bboxes))])
            all_valid_indices.extend(img_bbox_indices)
            
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print_log(f"CUDA OOM during inference! Reduce bbox_batch_size", logger='current')
            return None, None, None
        else:
            raise
    
    # Process results
    results = []
    for i, (mask, cls_pred) in enumerate(zip(all_masks, all_cls_preds)):
        if len(cls_pred.shape) > 1:
            cls_pred = cls_pred[0]
        
        scores, indices = torch.topk(cls_pred, 1)
        scores, indices = scores.tolist(), indices.tolist()
        
        names = [class_names[ind].replace('_', ' ') for ind in indices]
        results.append((names[0], scores[0]))
    
    return results, all_valid_indices, all_masks


def json_serializable(data):
    """Convert numpy types to JSON serializable types"""
    if isinstance(data, np.float32):
        return round(float(data), 2)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def filter_sam_boxes(annotations, threshold):
    """Filter SAM boxes by area ratio"""
    filtered_annotations = []
    for ann in annotations:
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]
        image_area = ann['segmentation']['size'][0] * ann['segmentation']['size'][1]
        area_ratio = area / image_area
        if area_ratio * 100 > threshold:
            filtered_annotations.append(ann)
    return filtered_annotations


def process_batch(batch_data, output_dir, dataset_name):
    """Process a batch of images"""
    image_info_list, images = batch_data
    
    # Extract features for all images in batch
    numpy_images, img_feats, original_sizes = batch_extract_img_feat(images)
    if numpy_images is None:
        return {}
    
    # Collect all bounding boxes from all images in the batch
    all_bboxes = []
    all_img_indices = []
    all_ann_indices = []
    batch_image_data = {}
    
    for img_idx, (image_info, image, numpy_image, orig_size) in enumerate(
        zip(image_info_list, images, numpy_images, original_sizes)):
        
        subfolder, filename = image_info
        image_key = f"{subfolder}/{filename}"
        
        # Determine SAM annotations path based on dataset structure
        if subfolder == 'images':  # Detection style
            sam_json_path = os.path.join(output_dir, dataset_name, 'sam', f"{os.path.splitext(filename)[0]}.json")
        else:  # Classification style
            sam_json_path = os.path.join(output_dir, dataset_name, subfolder, 'sam', f"{os.path.splitext(filename)[0]}.json")
        
        try:
            with open(sam_json_path) as r:
                annotations = json.load(r)['annotations']
        except (FileNotFoundError, KeyError) as e:
            print(f"Warning: Could not load SAM annotations for {image_key} at {sam_json_path}: {e}")
            continue
        
        batch_image_data[image_key] = {
            'image': image,
            'numpy_image': numpy_image,
            'annotations': annotations,
            'orig_size': orig_size,
            'predictions': []
        }
        
        orig_width, orig_height = orig_size
        numpy_height, numpy_width = numpy_image.shape[:2]
        width_scale = numpy_width / orig_width
        height_scale = numpy_height / orig_height
        
        # Filter annotations
        filtered_annotations = filter_sam_boxes(annotations, 1)
        
        for ann_idx, ann in enumerate(filtered_annotations):
            bbox = ann['bbox']
            x, y, w, h = bbox
            xyxy = [x, y, x + w, y + h]
            
            # Scale bbox to match processed image size
            scaled_xyxy = [
                xyxy[0] * width_scale,  # x1
                xyxy[1] * height_scale,  # y1
                xyxy[2] * width_scale,   # x2
                xyxy[3] * height_scale   # y2
            ]
            
            bbox_coords = [
                min(scaled_xyxy[0], scaled_xyxy[2]),
                min(scaled_xyxy[1], scaled_xyxy[3]),
                max(scaled_xyxy[0], scaled_xyxy[2]),
                max(scaled_xyxy[1], scaled_xyxy[3])
            ]
            
            all_bboxes.append(bbox_coords)
            all_img_indices.append(img_idx)
            all_ann_indices.append((image_key, ann_idx, xyxy))
    
    if not all_bboxes:
        return {}
    
    # Process bboxes in batches
    all_results = []
    bbox_batch_size = 32  # You can make this configurable
    
    for i in range(0, len(all_bboxes), bbox_batch_size):
        batch_bboxes = all_bboxes[i:i + bbox_batch_size]
        batch_img_indices = all_img_indices[i:i + bbox_batch_size]
        
        results, valid_indices, masks = batch_run_inference(img_feats, batch_bboxes, batch_img_indices)
        if results is None:
            continue
            
        all_results.extend(results)
    
    # Map results back to images
    final_data = {}
    for result_idx, (class_name, score) in enumerate(all_results):
        if score > 0.1:
            image_key, ann_idx, xyxy = all_ann_indices[result_idx]
            
            if image_key not in final_data:
                final_data[image_key] = {'ov_sam': []}
            
            prediction = {
                'bbox': xyxy,
                'score': round(score, 2),
                'label': class_name
            }
            
            final_data[image_key]['ov_sam'].append(
                {k: json_serializable(v) for k, v in prediction.items()}
            )
    
    return final_data


def process_dataset(dataset_name, dataset_path, output_dir, batch_size):
    """Process a single dataset"""
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Determine dataset structure
    dataset_type, dataset_data = scan_dataset_structure(dataset_path)
    
    if dataset_type is None:
        print(f"Skipping {dataset_name}: No valid structure found")
        return
    
    # Collect all images to process
    all_images = []
    if dataset_type == 'detection':
        all_images = dataset_data
    else:  # classification
        for class_name, images in dataset_data.items():
            all_images.extend(images)
    
    print(f"Found {len(all_images)} images in {dataset_type} style dataset")
    
    # Check which images need processing (resume functionality)
    images_to_process = []
    skipped_count = 0
    
    for subfolder, filename in all_images:
        # Check if output already exists
        if dataset_type == 'detection':
            output_json_path = os.path.join(output_dir, dataset_name, 'ov_sam', f"{os.path.splitext(filename)[0]}.json")
            sam_json_path = os.path.join(output_dir, dataset_name, 'sam', f"{os.path.splitext(filename)[0]}.json")
        else:  # classification
            output_json_path = os.path.join(output_dir, dataset_name, subfolder, 'ov_sam', f"{os.path.splitext(filename)[0]}.json")
            sam_json_path = os.path.join(output_dir, dataset_name, subfolder, 'sam', f"{os.path.splitext(filename)[0]}.json")
        
        # Check if output already exists (resume functionality)
        if os.path.exists(output_json_path):
            skipped_count += 1
            continue
            
        # Check if SAM annotations exist
        if not os.path.exists(sam_json_path):
            print(f"Warning: SAM annotations not found for {subfolder}/{filename} at {sam_json_path}")
            continue
            
        images_to_process.append((subfolder, filename))
    
    if skipped_count > 0:
        print(f"Resuming: Skipped {skipped_count} already processed images")
    
    if not images_to_process:
        print(f"All images in {dataset_name} have been processed!")
        return
        
    print(f"Processing {len(images_to_process)} remaining images")
    
    # Process images in batches
    for i in tqdm(range(0, len(images_to_process), batch_size), desc=f"Processing {dataset_name}"):
        batch_image_info = images_to_process[i:i + batch_size]
        batch_images = []
        valid_batch_info = []
        
        # Load batch data
        for subfolder, filename in batch_image_info:
            image_path = os.path.join(dataset_path, subfolder, filename)
            
            try:
                image = Image.open(image_path)
                batch_images.append(image)
                valid_batch_info.append((subfolder, filename))
            except Exception as e:
                print(f"Error loading {subfolder}/{filename}: {e}")
                continue
        
        if not valid_batch_info:
            continue
            
        # Process the batch
        batch_data = (valid_batch_info, batch_images)
        results = process_batch(batch_data, output_dir, dataset_name)
        
        # Save results
        for (subfolder, filename) in valid_batch_info:
            image_key = f"{subfolder}/{filename}"
            
            if dataset_type == 'detection':
                output_dir_path = os.path.join(output_dir, dataset_name, 'ov_sam')
            else:  # classification
                output_dir_path = os.path.join(output_dir, dataset_name, subfolder, 'ov_sam')
            
            os.makedirs(output_dir_path, exist_ok=True)
            output_json_path = os.path.join(output_dir_path, f"{os.path.splitext(filename)[0]}.json")
            
            image_result = results.get(image_key, {'ov_sam': []})
            
            with open(output_json_path, 'w') as f:
                json.dump(image_result, f)


def main():
    args = parse_args()
    datasets_dir = args.datasets_dir
    output_dir = args.output_dir
    batch_size = args.batch_size
    
    if not os.path.exists(datasets_dir):
        print(f"Error: Datasets directory {datasets_dir} does not exist")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all dataset folders
    dataset_folders = []
    for item in os.listdir(datasets_dir):
        item_path = os.path.join(datasets_dir, item)
        if os.path.isdir(item_path):
            dataset_folders.append(item)
    
    print(f"Found {len(dataset_folders)} dataset folders")
    
    start_time = time.time()
    
    # Process each dataset
    for dataset_name in dataset_folders:
        dataset_path = os.path.join(datasets_dir, dataset_name)
        try:
            process_dataset(dataset_name, dataset_path, output_dir, batch_size)
        except Exception as e:
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print('\033[92m' + f"---- Total processing time: {elapsed_time:.2f} seconds ----" + '\033[0m')
    print(f"Processed {len(dataset_folders)} datasets with batch_size={batch_size}")


if __name__ == "__main__":
    main()