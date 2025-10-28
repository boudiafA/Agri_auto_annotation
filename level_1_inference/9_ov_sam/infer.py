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

class_names = [
    "almond", "apple", "avocado", "banana", "broccoli", "cabbage", "capsicum", "cotton",
    "cucumber", "flower", "grape", "kiwi", "leaf", "lemon", "lettuce", "maize_tassel",
    "mango", "orange", "pineapple", "plant", "potato", "pumpkin", "rice_panicles", "rockmelon",
    "sorghum_head", "soybean_pod", "strawberry", "tree", "tomato", "weed", "wheat_head"
]


class_names = list(set(class_names) | set(LVIS_CLASSES))


def parse_args():
    parser = argparse.ArgumentParser(description="Open Vocabulary SAM with Batch Processing")

    parser.add_argument("--image_names_txt_path", required=True)
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--sam_annotations_dir", required=True,
                        help="path to the directory containing all sam annotations.")
    parser.add_argument("--checkpoint", required=True,
                        help="path to the directory containing all model weights")
    parser.add_argument("--config", default='app/configs/sam_r50x16_fpn.py',
                        help="path to the model config file")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images to process in each batch")
    parser.add_argument("--bbox_batch_size", type=int, default=32,
                        help="Number of bounding boxes to process in each inference batch")

    args = parser.parse_args()
    return args


def update_checkpoint_paths(cfg, checkpoint_dir):
    """
    Update only the neck checkpoint path to use the specified checkpoint directory.
    Other checkpoints (fpn_neck and mask_decoder) remain unchanged from the config file.
    
    Expected checkpoint file in the directory:
    - sam2clip_vith_rn50.pth (for neck)
    """
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    
    # Update ONLY neck checkpoint
    if 'neck' in cfg.model and 'init_cfg' in cfg.model['neck']:
        if cfg.model['neck']['init_cfg']['type'] == 'Pretrained':
            original_checkpoint = cfg.model['neck']['init_cfg']['checkpoint']
            filename = os.path.basename(original_checkpoint)
            cfg.model['neck']['init_cfg']['checkpoint'] = os.path.join(checkpoint_dir, filename)
            print_log(f"Updated neck checkpoint: {cfg.model['neck']['init_cfg']['checkpoint']}", logger='current')
    
    # Log other checkpoint paths but DO NOT modify them
    if 'fpn_neck' in cfg.model and 'init_cfg' in cfg.model['fpn_neck']:
        if cfg.model['fpn_neck']['init_cfg']['type'] == 'Pretrained':
            print_log(f"FPN neck checkpoint (unchanged): {cfg.model['fpn_neck']['init_cfg']['checkpoint']}", logger='current')
    
    if 'mask_decoder' in cfg.model and 'init_cfg' in cfg.model['mask_decoder']:
        if cfg.model['mask_decoder']['init_cfg']['type'] == 'Pretrained':
            print_log(f"Mask decoder checkpoint (unchanged): {cfg.model['mask_decoder']['init_cfg']['checkpoint']}", logger='current')
        
        if 'load_roi_conv' in cfg.model['mask_decoder']:
            print_log(f"ROI conv checkpoint (unchanged): {cfg.model['mask_decoder']['load_roi_conv']['checkpoint']}", logger='current')
    
    return cfg


def initialize_model(config_path, checkpoint_dir, device):
    """Initialize the model with checkpoints from the specified directory."""
    print_log(f"Loading config from: {config_path}", logger='current')
    print_log(f"Loading checkpoints from: {checkpoint_dir}", logger='current')
    
    model_cfg = Config.fromfile(config_path)
    model_cfg = update_checkpoint_paths(model_cfg, checkpoint_dir)
    
    print_log("Building model...", logger='current')
    model = MODELS.build(model_cfg.model)
    model = model.to(device=device)
    model = model.eval()
    
    print_log("Initializing weights...", logger='current')
    model.init_weights()
    
    print_log("Model initialization complete!", logger='current')
    return model


mean = None
std = None
IMG_SIZE = 1024
model = None
device = None


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


def get_bbox_with_draw(image, bbox, index):
    """Draw bounding box on image"""
    point_radius, point_color, box_outline = 5, (237, 34, 13), 2
    myFont = ImageFont.load_default()
    box_color, text_color = (237, 34, 13), (255, 0, 0)
    draw = ImageDraw.Draw(image)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=box_color, width=box_outline)
    draw.text((x, y), str(index), fill=text_color, font=myFont)
    xyxy = [x, y, x + w, y + h]
    return image, xyxy


def json_serializable(data):
    """Convert numpy types to JSON serializable types"""
    if isinstance(data, np.float32):
        return round(float(data), 2)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def filter_sam_boxes(annotations, image_info, threshold):
    """Filter SAM boxes by area ratio"""
    filtered_annotations = []
    
    # Get image dimensions from image_info
    if 'original_size' in image_info:
        image_width = image_info['original_size']['width']
        image_height = image_info['original_size']['height']
    else:
        # Fallback: try to get from first annotation's segmentation size
        if annotations and 'segmentation' in annotations[0] and 'size' in annotations[0]['segmentation']:
            seg_size = annotations[0]['segmentation']['size']
            image_height, image_width = seg_size  # Note: size is [height, width]
        else:
            print("Warning: Could not determine image size, skipping filtering")
            return annotations
    
    image_area = image_width * image_height
    
    for ann in annotations:
        bbox = ann['bbox']
        area = bbox[2] * bbox[3]  # width * height
        area_ratio = area / image_area
        if area_ratio * 100 > threshold:
            filtered_annotations.append(ann)
    
    return filtered_annotations


def process_batch(batch_data, args):
    """Process a batch of images"""
    image_names, images, annotations_list, image_infos = batch_data
    
    # Extract features for all images in batch
    numpy_images, img_feats, original_sizes = batch_extract_img_feat(images)
    if numpy_images is None:
        return {}
    
    # Collect all bounding boxes from all images in the batch
    all_bboxes = []
    all_img_indices = []
    all_ann_indices = []
    batch_image_data = {}
    
    for img_idx, (image_name, image, annotations, image_info, numpy_image, orig_size) in enumerate(
        zip(image_names, images, annotations_list, image_infos, numpy_images, original_sizes)):
        
        batch_image_data[image_name] = {
            'image': image,
            'numpy_image': numpy_image,
            'annotations': annotations,
            'image_info': image_info,
            'orig_size': orig_size,
            'predictions': []
        }
        
        orig_width, orig_height = orig_size
        numpy_height, numpy_width = numpy_image.shape[:2]
        width_scale = numpy_width / orig_width
        height_scale = numpy_height / orig_height
        
        # Filter annotations
        filtered_annotations = filter_sam_boxes(annotations, image_info, 1)
        
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
            all_ann_indices.append((image_name, ann_idx, xyxy))
    
    # Process bboxes in batches
    all_results = []
    bbox_batch_size = args.bbox_batch_size
    
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
            image_name, ann_idx, xyxy = all_ann_indices[result_idx]
            
            if image_name not in final_data:
                final_data[image_name] = {}
            
            if 'ov_sam' not in final_data[image_name]:
                final_data[image_name]['ov_sam'] = []
            
            prediction = {
                'bbox': xyxy,
                'score': round(score, 2),
                'label': class_name
            }
            
            final_data[image_name]['ov_sam'].append(
                {k: json_serializable(v) for k, v in prediction.items()}
            )
    
    return final_data


if __name__ == "__main__":
    args = parse_args()
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    sam_annotations_dir = args.sam_annotations_dir
    batch_size = args.batch_size

    # Verify checkpoint directory exists
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    
    print_log(f"Using checkpoint directory: {args.checkpoint}", logger='current')

    os.makedirs(output_dir_path, exist_ok=True)
    os.makedirs(f"{output_dir_path}/ov_sam", exist_ok=True)

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print_log(f"Using device: {device}", logger='current')

    # Initialize model with checkpoint directory
    model = initialize_model(args.config, args.checkpoint, device)

    # Initialize mean and std tensors
    mean = torch.tensor([123.675, 116.28, 103.53], device=device)[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375], device=device)[:, None, None]

    with open(args.image_names_txt_path, 'r') as f:
        image_names = [line.strip() for line in f if line.strip()]

    start_time = time.time()
    
    # Process images in batches
    for i in tqdm(range(0, len(image_names), batch_size), desc="Processing batches"):
        batch_names = image_names[i:i + batch_size]
        batch_images = []
        batch_annotations = []
        batch_image_infos = []
        valid_batch_names = []
        
        # Load batch data
        for image_name in batch_names:
            output_json_path = f"{output_dir_path}/ov_sam/{os.path.splitext(image_name)[0]}.json"
            if os.path.exists(output_json_path):
                continue
                
            image_path = f"{image_dir_path}/{image_name}"
            sam_json_path = f"{sam_annotations_dir}/{os.path.splitext(image_name)[0]}.json"
            
            try:
                image = Image.open(image_path)
                with open(sam_json_path) as r:
                    data = json.load(r)
                
                # Extract annotations and image_info from new structure
                # The JSON has the image name as a key, so we need to find it
                annotations = None
                image_info = None
                
                # Try to find the image data using the image_name as key
                if image_name in data:
                    annotations = data[image_name].get('sam', [])
                    image_info = data.get('image_info', {})
                else:
                    # Fallback: try to find by checking all keys (in case filename differs slightly)
                    for key in data.keys():
                        if key.endswith(('.jpg', '.jpeg', '.png', '.bmp')) and 'sam' in data[key]:
                            annotations = data[key]['sam']
                            image_info = data.get('image_info', {})
                            break
                
                if annotations is None:
                    print(f"Skipping {image_name}: 'sam' annotations not found in JSON")
                    continue
                
                batch_images.append(image)
                batch_annotations.append(annotations)
                batch_image_infos.append(image_info)
                valid_batch_names.append(image_name)
                
            except FileNotFoundError as e:
                print(f"Error loading {image_name}: File not found - {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"Error loading {image_name}: Invalid JSON - {e}")
                continue
            except Exception as e:
                print(f"Error loading {image_name}: {e}")
                continue
        
        if not valid_batch_names:
            continue
            
        # Process the batch
        batch_data = (valid_batch_names, batch_images, batch_annotations, batch_image_infos)
        results = process_batch(batch_data, args)
        
        # Save results
        for image_name in valid_batch_names:
            output_json_path = f"{output_dir_path}/ov_sam/{os.path.splitext(image_name)[0]}.json"
            image_result = results.get(image_name, {image_name: {'ov_sam': []}})
            
            with open(output_json_path, 'w') as f:
                json.dump(image_result, f)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + f"---- Batch OV-SAM Time taken: {elapsed_time:.2f} seconds ----" + '\033[0m')
    print(f"Processed {len(image_names)} images with batch_size={batch_size}, bbox_batch_size={args.bbox_batch_size}")