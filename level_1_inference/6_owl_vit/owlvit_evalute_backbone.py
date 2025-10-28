#!/usr/bin/env python3
"""
Standalone evaluation script for fine-tuned OWL-ViT models.
Loads a saved model and evaluates it on test dataset with comprehensive metrics.
"""

import os
import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Optional
import json
import yaml
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import sys
from pathlib import Path
import time


class YOLODataset(Dataset):
    """Dataset class for loading YOLO format data for OWL-ViT evaluation."""
    
    def __init__(self, 
                 images_dir: str, 
                 labels_dir: str, 
                 class_names: List[str],
                 processor: OwlViTProcessor,
                 max_objects_per_image: int = 100):
        """
        Args:
            images_dir: Directory containing images
            labels_dir: Directory containing YOLO format label files
            class_names: List of class names corresponding to class IDs
            processor: OWL-ViT processor for preprocessing
            max_objects_per_image: Maximum number of objects per image to consider
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.class_names = class_names
        self.processor = processor
        self.max_objects_per_image = max_objects_per_image
        
        # Get all image files
        self.image_files = []
        if os.path.exists(images_dir):
            for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                self.image_files.extend([f for f in os.listdir(images_dir) if f.lower().endswith(ext)])
        
        logging.info(f"Found {len(self.image_files)} images in {images_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        image_width, image_height = image.size
        
        # Load corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_file)
        
        boxes = []
        class_ids = []
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        center_x = float(parts[1])
                        center_y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        
                        # Convert from YOLO format (center, normalized) to corner format (pixel coordinates)
                        x1 = (center_x - width / 2) * image_width
                        y1 = (center_y - height / 2) * image_height
                        x2 = (center_x + width / 2) * image_width
                        y2 = (center_y + height / 2) * image_height
                        
                        boxes.append([x1, y1, x2, y2])
                        class_ids.append(class_id)
        
        # Limit number of objects per image
        if len(boxes) > self.max_objects_per_image:
            boxes = boxes[:self.max_objects_per_image]
            class_ids = class_ids[:self.max_objects_per_image]
        
        # Convert to tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            class_ids = torch.tensor(class_ids, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            class_ids = torch.zeros((0,), dtype=torch.long)
        
        return {
            'image': image,
            'boxes': boxes,
            'class_ids': class_ids,
            'image_file': image_file,
            'image_id': idx,
            'original_size': (image_width, image_height)
        }


def parse_dataset_yaml(yaml_path: str) -> Dict:
    """Parse the dataset.yaml file to extract dataset information."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def get_class_names_from_yaml(dataset_info: Dict) -> List[str]:
    """Extract class names from dataset yaml information."""
    names = dataset_info.get('names', [])
    
    if isinstance(names, dict):
        max_idx = max(names.keys()) if names else -1
        class_names = [''] * (max_idx + 1)
        for idx, name in names.items():
            class_names[idx] = name
    elif isinstance(names, list):
        class_names = names
    else:
        raise ValueError(f"Unsupported names format in dataset.yaml: {type(names)}")
    
    return class_names


def setup_dataset_paths(dataset_path: str) -> Tuple[Dict[str, str], List[str]]:
    """Setup dataset paths and extract class names from YOLO dataset structure."""
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    
    # Parse dataset.yaml
    yaml_path = os.path.join(dataset_path, 'dataset.yaml')
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"dataset.yaml not found in {dataset_path}")
    
    dataset_info = parse_dataset_yaml(yaml_path)
    class_names = get_class_names_from_yaml(dataset_info)
    
    logging.info(f"Loaded {len(class_names)} classes from dataset.yaml: {class_names}")
    
    # Setup paths for images and labels
    images_base = os.path.join(dataset_path, 'images')
    labels_base = os.path.join(dataset_path, 'labels')
    
    paths = {}
    
    # Check for test split first (primary for evaluation)
    test_images = os.path.join(images_base, 'test')
    test_labels = os.path.join(labels_base, 'test')
    if os.path.exists(test_images) and os.path.exists(test_labels):
        paths['test'] = {'images': test_images, 'labels': test_labels}
        logging.info(f"Found test split: {test_images}")
    
    # Check for validation split
    val_images = os.path.join(images_base, 'val')
    val_labels = os.path.join(labels_base, 'val')
    if os.path.exists(val_images) and os.path.exists(val_labels):
        paths['val'] = {'images': val_images, 'labels': val_labels}
        logging.info(f"Found val split: {val_images}")
    
    # Check for train split (for baseline comparison if needed)
    train_images = os.path.join(images_base, 'train')
    train_labels = os.path.join(labels_base, 'train')
    if os.path.exists(train_images) and os.path.exists(train_labels):
        paths['train'] = {'images': train_images, 'labels': train_labels}
        logging.info(f"Found train split: {train_images}")
    
    if not paths:
        raise FileNotFoundError("No valid train/val/test splits found in the dataset")
    
    return paths, class_names


def collate_fn(batch, processor, class_names, max_text_queries=20):
    """Custom collate function for batching OWL-ViT data."""
    
    images = [item['image'] for item in batch]
    
    # Use ALL classes as text queries for consistent evaluation
    text_queries = []
    for i, class_name in enumerate(class_names):
        if i >= max_text_queries:
            break
        text_queries.append(f"a photo of a {class_name}")
    
    # Process images and text
    inputs = processor(
        text=[text_queries] * len(images),
        images=images, 
        return_tensors="pt",
        padding=True
    )
    
    # Prepare targets
    targets = []
    for item in batch:
        target = {}
        
        if len(item['boxes']) > 0:
            # Normalize boxes to [0, 1] and convert to center format for OWL-ViT
            image_width, image_height = item['original_size']
            
            # Convert corner format to center format and normalize
            boxes_corner = item['boxes']
            center_x = (boxes_corner[:, 0] + boxes_corner[:, 2]) / 2 / image_width
            center_y = (boxes_corner[:, 1] + boxes_corner[:, 3]) / 2 / image_height
            width = (boxes_corner[:, 2] - boxes_corner[:, 0]) / image_width
            height = (boxes_corner[:, 3] - boxes_corner[:, 1]) / image_height
            
            boxes_center = torch.stack([center_x, center_y, width, height], dim=1)
            
            # Use class IDs directly as labels
            labels = []
            valid_boxes = []
            for i, class_id in enumerate(item['class_ids']):
                class_id_val = class_id.item()
                if 0 <= class_id_val < len(text_queries):
                    labels.append(class_id_val)
                    valid_boxes.append(boxes_center[i])
            
            if valid_boxes:
                target['boxes'] = torch.stack(valid_boxes)
                target['labels'] = torch.tensor(labels, dtype=torch.long)
            else:
                target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['labels'] = torch.zeros((0,), dtype=torch.long)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.long)
        
        target['image_id'] = item['image_id']
        target['original_size'] = item['original_size']
        target['image_file'] = item['image_file']
        targets.append(target)
    
    return {
        'inputs': inputs,
        'targets': targets,
        'text_queries': text_queries
    }


def inference_and_evaluate(model, dataloader, processor, class_names, device, conf_threshold=0.5):
    """Run inference and collect predictions for evaluation."""
    model.eval()
    all_predictions = []
    all_ground_truths = []
    
    chunk_size = 4
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Running inference')):
            inputs = batch['inputs']
            targets = batch['targets']
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            try:
                outputs = model(**inputs)
                
                # Move outputs to CPU immediately
                logits_cpu = outputs.logits.detach().cpu()
                pred_boxes_cpu = outputs.pred_boxes.detach().cpu()
                
                del outputs
                torch.cuda.empty_cache()
                
                cpu_outputs = type('Outputs', (), {
                    'logits': logits_cpu,
                    'pred_boxes': pred_boxes_cpu
                })()
                
                # FIXED: Correct target_sizes format (height, width)
                batch_size = cpu_outputs.logits.shape[0]
                target_sizes = []
                
                for i in range(batch_size):
                    width, height = targets[i]['original_size']
                    target_sizes.append([height, width])  # height first!
                
                target_sizes = torch.tensor(target_sizes, dtype=torch.long)
                
                temp_outputs = type('Outputs', (), {
                    'logits': cpu_outputs.logits.to(device),
                    'pred_boxes': cpu_outputs.pred_boxes.to(device)
                })()
                target_sizes = target_sizes.to(device)
                
                # Get all detections (no confidence filtering here)
                results = processor.post_process_object_detection(
                    outputs=temp_outputs,
                    threshold=0.0,  # Get all detections
                    target_sizes=target_sizes
                )
                
                # Move results to CPU immediately
                cpu_results = []
                for result in results:
                    cpu_result = {
                        'boxes': result['boxes'].detach().cpu(),
                        'scores': result['scores'].detach().cpu(),
                        'labels': result['labels'].detach().cpu()
                    }
                    cpu_results.append(cpu_result)
                
                del temp_outputs, target_sizes, results
                torch.cuda.empty_cache()
                
                # Collect predictions and ground truths
                for i in range(batch_size):
                    pred = cpu_results[i]
                    all_predictions.append(pred)
                    
                    gt = {
                        'boxes': targets[i]['boxes'].clone(),
                        'labels': targets[i]['labels'].clone(),
                        'original_size': targets[i]['original_size'],
                        'image_file': targets[i]['image_file']
                    }
                    all_ground_truths.append(gt)
                
                del cpu_outputs, logits_cpu, pred_boxes_cpu, cpu_results
                
            except RuntimeError as e:
                logging.error(f"Error in inference batch {batch_idx}: {e}")
                # Add empty predictions for failed batches
                for i in range(len(targets)):
                    pred = {
                        'boxes': torch.empty((0, 4)),
                        'scores': torch.empty(0),
                        'labels': torch.empty(0, dtype=torch.long)
                    }
                    all_predictions.append(pred)
                    
                    gt = {
                        'boxes': targets[i]['boxes'].clone(),
                        'labels': targets[i]['labels'].clone(),
                        'original_size': targets[i]['original_size'],
                        'image_file': targets[i]['image_file']
                    }
                    all_ground_truths.append(gt)
                
                torch.cuda.empty_cache()
                gc.collect()
                continue
            
            del inputs, targets, batch
            
            batch_count += 1
            if batch_count % chunk_size == 0:
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                gc.collect()
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    
    return all_predictions, all_ground_truths


def compute_iou_numpy(boxes1, boxes2):
    """Compute IoU between two sets of boxes in corner format."""
    boxes1 = np.expand_dims(boxes1, axis=1)  # [N, 1, 4]
    boxes2 = np.expand_dims(boxes2, axis=0)  # [1, M, 4]
    
    # Compute intersection
    x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Compute union and IoU
    union = area1 + area2 - intersection
    iou = intersection / (union + 1e-6)
    
    return iou


def compute_ap(recalls, precisions):
    """Compute Average Precision using the 11-point interpolation method."""
    sorted_indices = np.argsort(recalls)
    recalls = recalls[sorted_indices]
    precisions = precisions[sorted_indices]
    
    # 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        mask = recalls >= t
        if np.any(mask):
            ap += np.max(precisions[mask])
    
    return ap / 11.0


def _process_image_matches(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, 
                          class_stats, num_classes, iou_threshold):
    """Helper function to process matches for a single image."""
    if len(pred_boxes) == 0:
        for class_id in range(num_classes):
            class_stats['fn'][class_id] += np.sum(gt_labels == class_id)
        return
    
    if len(gt_boxes) == 0:
        for i, pred_label in enumerate(pred_labels):
            if 0 <= pred_label < num_classes:
                class_stats['fp'][pred_label] += 1
                class_stats['detections'][pred_label].append((pred_scores[i], False))
        return
    
    # Compute IoU matrix
    iou_matrix = compute_iou_numpy(pred_boxes, gt_boxes)
    
    # Match predictions to ground truth
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    # Sort predictions by confidence (descending)
    sorted_indices = np.argsort(-pred_scores)
    
    for pred_idx in sorted_indices:
        pred_label = pred_labels[pred_idx]
        pred_score = pred_scores[pred_idx]
        
        if not (0 <= pred_label < num_classes):
            continue
        
        # Find best matching ground truth
        ious = iou_matrix[pred_idx]
        best_gt_idx = np.argmax(ious)
        best_iou = ious[best_gt_idx]
        
        # Check if it's a valid match
        if (best_iou >= iou_threshold and 
            gt_labels[best_gt_idx] == pred_label and 
            not gt_matched[best_gt_idx]):
            # True positive
            class_stats['tp'][pred_label] += 1
            class_stats['detections'][pred_label].append((pred_score, True))
            gt_matched[best_gt_idx] = True
        else:
            # False positive
            class_stats['fp'][pred_label] += 1
            class_stats['detections'][pred_label].append((pred_score, False))
    
    # Count false negatives (unmatched ground truth)
    for gt_idx, matched in enumerate(gt_matched):
        if not matched:
            gt_label = gt_labels[gt_idx]
            if 0 <= gt_label < num_classes:
                class_stats['fn'][gt_label] += 1


def evaluate_detection_metrics(predictions, ground_truths, class_names, iou_threshold=0.5, conf_threshold=0.5):
    """Evaluate detection metrics with fixed box format conversion."""
    num_classes = len(class_names)
    
    class_stats = {
        'tp': np.zeros(num_classes),
        'fp': np.zeros(num_classes),
        'fn': np.zeros(num_classes),
        'total_gt': np.zeros(num_classes),
        'detections': [[] for _ in range(num_classes)]
    }
    
    total_gt_objects = 0
    total_pred_objects = 0
    
    chunk_size = 50
    total_images = len(predictions)
    
    logging.info(f"Processing {total_images} images in chunks of {chunk_size} for metric calculation.")
    
    for chunk_start in range(0, total_images, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_images)
        
        for idx in range(chunk_start, chunk_end):
            pred = predictions[idx]
            gt = ground_truths[idx]
            
            # Process predictions
            if len(pred['scores']) > 0:
                if hasattr(pred['scores'], 'cpu'):
                    scores_np = pred['scores'].cpu().numpy()
                    pred_boxes_np = pred['boxes'].cpu().numpy()
                    pred_labels_np = pred['labels'].cpu().numpy()
                else:
                    scores_np = pred['scores'].numpy() if hasattr(pred['scores'], 'numpy') else np.array(pred['scores'])
                    pred_boxes_np = pred['boxes'].numpy() if hasattr(pred['boxes'], 'numpy') else np.array(pred['boxes'])
                    pred_labels_np = pred['labels'].numpy() if hasattr(pred['labels'], 'numpy') else np.array(pred['labels'])
                
                # Apply confidence threshold
                conf_mask = scores_np >= conf_threshold
                
                if conf_mask.any():
                    pred_boxes_np = pred_boxes_np[conf_mask]
                    pred_scores_np = scores_np[conf_mask]
                    pred_labels_np = pred_labels_np[conf_mask]
                else:
                    pred_boxes_np = np.empty((0, 4))
                    pred_scores_np = np.empty(0)
                    pred_labels_np = np.empty(0, dtype=int)
            else:
                pred_boxes_np = np.empty((0, 4))
                pred_scores_np = np.empty(0)
                pred_labels_np = np.empty(0, dtype=int)
            
            # Process ground truth
            if len(gt['boxes']) > 0:
                if hasattr(gt['boxes'], 'cpu'):
                    gt_boxes_center_norm = gt['boxes'].cpu().numpy()
                    gt_labels_np = gt['labels'].cpu().numpy()
                else:
                    gt_boxes_center_norm = gt['boxes'].numpy() if hasattr(gt['boxes'], 'numpy') else np.array(gt['boxes'])
                    gt_labels_np = gt['labels'].numpy() if hasattr(gt['labels'], 'numpy') else np.array(gt['labels'])
                
                # Handle original_size format correctly
                if isinstance(gt['original_size'], (list, tuple)) and len(gt['original_size']) == 2:
                    img_width, img_height = gt['original_size']
                else:
                    img_width = img_height = 640
                    logging.warning(f"Invalid original_size format for image {idx}: {gt['original_size']}")
                
                # Convert from center normalized to corner pixels
                cx, cy, w, h = gt_boxes_center_norm[:, 0], gt_boxes_center_norm[:, 1], gt_boxes_center_norm[:, 2], gt_boxes_center_norm[:, 3]
                x1 = (cx - w / 2) * img_width
                y1 = (cy - h / 2) * img_height
                x2 = (cx + w / 2) * img_width
                y2 = (cy + h / 2) * img_height
                gt_boxes_np = np.stack([x1, y1, x2, y2], axis=1)
                
                # Ensure boxes are valid
                valid_mask = (gt_boxes_np[:, 2] > gt_boxes_np[:, 0]) & (gt_boxes_np[:, 3] > gt_boxes_np[:, 1])
                if not valid_mask.all():
                    logging.warning(f"Found {(~valid_mask).sum()} invalid ground truth boxes in image {idx}")
                    gt_boxes_np = gt_boxes_np[valid_mask]
                    gt_labels_np = gt_labels_np[valid_mask]
            else:
                gt_boxes_np = np.empty((0, 4))
                gt_labels_np = np.empty(0, dtype=int)
            
            # Update statistics
            total_pred_objects += len(pred_boxes_np)
            total_gt_objects += len(gt_boxes_np)
            
            # Count ground truth objects per class
            for class_id in range(num_classes):
                class_stats['total_gt'][class_id] += np.sum(gt_labels_np == class_id)
            
            # Process matches for this image
            _process_image_matches(pred_boxes_np, pred_scores_np, pred_labels_np, 
                                 gt_boxes_np, gt_labels_np, class_stats, num_classes, iou_threshold)
            
            # Clear variables
            del pred_boxes_np, pred_scores_np, pred_labels_np, gt_boxes_np, gt_labels_np
            del gt_boxes_center_norm, pred, gt
        
        # Cleanup after each chunk
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Compute final metrics
    metrics = _compute_final_metrics(class_stats, class_names, num_classes, total_pred_objects, 
                                    total_gt_objects, len(predictions), iou_threshold, conf_threshold)
    
    del class_stats
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return metrics


def _compute_final_metrics(class_stats, class_names, num_classes, total_pred_objects, 
                          total_gt_objects, total_images, iou_threshold, conf_threshold):
    """Helper function to compute final metrics."""
    metrics = {
        'per_class': {},
        'overall': {},
        'summary': {}
    }
    
    total_ap = 0
    valid_classes = 0
    
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        tp = class_stats['tp'][class_id]
        fp = class_stats['fp'][class_id]
        fn = class_stats['fn'][class_id]
        total_gt = class_stats['total_gt'][class_id]
        
        # Compute precision, recall, and F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Compute AP
        ap = 0.0
        if class_stats['detections'][class_id]:
            detections = class_stats['detections'][class_id]
            detections.sort(key=lambda x: x[0], reverse=True)
            
            precisions = []
            recalls = []
            tp_count = 0
            fp_count = 0
            
            for conf, is_correct in detections:
                if is_correct:
                    tp_count += 1
                else:
                    fp_count += 1
                
                prec = tp_count / (tp_count + fp_count)
                rec = tp_count / total_gt if total_gt > 0 else 0.0
                
                precisions.append(prec)
                recalls.append(rec)
            
            if precisions and recalls:
                precisions = np.array(precisions)
                recalls = np.array(recalls)
                ap = compute_ap(recalls, precisions)
        
        metrics['per_class'][class_name] = {
            'ap': ap,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'total_gt': int(total_gt)
        }
        
        if total_gt > 0:
            total_ap += ap
            valid_classes += 1
    
    # Overall metrics
    overall_tp = np.sum(class_stats['tp'])
    overall_fp = np.sum(class_stats['fp'])
    overall_fn = np.sum(class_stats['fn'])
    
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    map_score = total_ap / valid_classes if valid_classes > 0 else 0.0
    
    metrics['overall'] = {
        'mAP': map_score,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'tp': int(overall_tp),
        'fp': int(overall_fp),
        'fn': int(overall_fn)
    }
    
    metrics['summary'] = {
        'total_images': total_images,
        'total_gt_objects': total_gt_objects,
        'total_pred_objects': total_pred_objects,
        'valid_classes': valid_classes,
        'iou_threshold': iou_threshold,
        'conf_threshold': conf_threshold
    }
    
    return metrics


def print_evaluation_results(metrics, model_name="Fine-tuned Model"):
    """Print evaluation results in a formatted way."""
    header_bar = f"\n{'='*80}"
    logging.info(header_bar)
    logging.info(f"EVALUATION RESULTS - {model_name.upper()}")
    logging.info(f"{'='*80}")
    
    # Print overall metrics
    overall = metrics['overall']
    summary = metrics['summary']
    
    logging.info(f"OVERALL METRICS:")
    logging.info(f"  mAP@{summary['iou_threshold']}: {overall['mAP']:.4f}")
    logging.info(f"  Precision: {overall['precision']:.4f}")
    logging.info(f"  Recall: {overall['recall']:.4f}")
    logging.info(f"  F1-Score: {overall['f1']:.4f}")
    
    logging.info(f"\nDETECTION SUMMARY:")
    logging.info(f"  Total Images: {summary['total_images']}")
    logging.info(f"  Total GT Objects: {summary['total_gt_objects']}")
    logging.info(f"  Total Predictions: {summary['total_pred_objects']}")
    logging.info(f"  True Positives: {overall['tp']}")
    logging.info(f"  False Positives: {overall['fp']}")
    logging.info(f"  False Negatives: {overall['fn']}")
    logging.info(f"  Valid Classes: {summary['valid_classes']}")
    
    # Print per-class metrics
    logging.info(f"\nPER-CLASS METRICS:")
    logging.info(f"{'Class':<20} {'AP':<8} {'Prec':<8} {'Rec':<8} {'F1':<8} {'GT':<6} {'TP':<6} {'FP':<6} {'FN':<6}")
    logging.info(f"{'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    
    for class_name, class_metrics in metrics['per_class'].items():
        logging.info(f"{class_name:<20} {class_metrics['ap']:<8.4f} {class_metrics['precision']:<8.4f} "
              f"{class_metrics['recall']:<8.4f} {class_metrics['f1']:<8.4f} {class_metrics['total_gt']:<6d} "
              f"{class_metrics['tp']:<6d} {class_metrics['fp']:<6d} {class_metrics['fn']:<6d}")
    
    logging.info(f"{'='*80}")


def visualize_predictions(predictions, ground_truths, class_names, dataset, 
                         output_dir, num_samples=10, conf_threshold=0.3):
    """Visualize predictions vs ground truth for sample images."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Creating visualizations for {min(num_samples, len(predictions))} sample images...")
    
    for i in range(min(num_samples, len(predictions))):
        try:
            # Get data
            pred = predictions[i]
            gt = ground_truths[i]
            
            # Load original image
            image_path = os.path.join(dataset.images_dir, gt['image_file'])
            image = Image.open(image_path).convert('RGB')
            
            # Create subplot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Ground Truth
            ax1.imshow(image)
            ax1.set_title(f"Ground Truth - {gt['image_file']}", fontsize=14)
            ax1.axis('off')
            
            if len(gt['boxes']) > 0:
                gt_boxes_np = gt['boxes'].cpu().numpy() if hasattr(gt['boxes'], 'cpu') else gt['boxes'].numpy()
                gt_labels_np = gt['labels'].cpu().numpy() if hasattr(gt['labels'], 'cpu') else gt['labels'].numpy()
                
                img_width, img_height = gt['original_size']
                
                # Convert from center normalized to corner pixels for visualization
                for j, (box, label) in enumerate(zip(gt_boxes_np, gt_labels_np)):
                    cx, cy, w, h = box
                    x1 = (cx - w / 2) * img_width
                    y1 = (cy - h / 2) * img_height
                    x2 = (cx + w / 2) * img_width
                    y2 = (cy + h / 2) * img_height
                    
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor='green', facecolor='none')
                    ax1.add_patch(rect)
                    
                    if label < len(class_names):
                        ax1.text(x1, y1-5, f'{class_names[label]}', 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.7),
                               fontsize=10, color='white')
            
            # Predictions
            ax2.imshow(image)
            ax2.set_title(f"Predictions (conf > {conf_threshold}) - {gt['image_file']}", fontsize=14)
            ax2.axis('off')
            
            if len(pred['scores']) > 0:
                pred_boxes_np = pred['boxes'].cpu().numpy() if hasattr(pred['boxes'], 'cpu') else pred['boxes'].numpy()
                pred_scores_np = pred['scores'].cpu().numpy() if hasattr(pred['scores'], 'cpu') else pred['scores'].numpy()
                pred_labels_np = pred['labels'].cpu().numpy() if hasattr(pred['labels'], 'cpu') else pred['labels'].numpy()
                
                # Filter by confidence
                conf_mask = pred_scores_np >= conf_threshold
                if conf_mask.any():
                    pred_boxes_filtered = pred_boxes_np[conf_mask]
                    pred_scores_filtered = pred_scores_np[conf_mask]
                    pred_labels_filtered = pred_labels_np[conf_mask]
                    
                    # Predictions are already in corner pixel format from post_process_object_detection
                    for j, (box, score, label) in enumerate(zip(pred_boxes_filtered, pred_scores_filtered, pred_labels_filtered)):
                        x1, y1, x2, y2 = box
                        
                        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                               linewidth=2, edgecolor='red', facecolor='none')
                        ax2.add_patch(rect)
                        
                        if label < len(class_names):
                            ax2.text(x1, y1-5, f'{class_names[label]}: {score:.2f}', 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.7),
                                   fontsize=10, color='white')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'prediction_{i:03d}_{gt["image_file"]}'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating visualization for image {i}: {e}")
            plt.close()
    
    logging.info(f"Visualizations saved to {output_dir}")


def compare_with_baseline(finetuned_metrics, baseline_metrics=None):
    """Compare fine-tuned model with baseline if available."""
    
    if baseline_metrics is None:
        logging.info("No baseline metrics available for comparison.")
        return
    
    logging.info(f"\n{'='*80}")
    logging.info("MODEL COMPARISON - FINE-TUNED vs BASELINE")
    logging.info(f"{'='*80}")
    
    ft_overall = finetuned_metrics['overall']
    base_overall = baseline_metrics['overall']
    
    # Overall comparison
    logging.info("OVERALL METRICS COMPARISON:")
    logging.info(f"{'Metric':<15} {'Baseline':<12} {'Fine-tuned':<12} {'Improvement':<12}")
    logging.info(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}")
    
    for metric in ['mAP', 'precision', 'recall', 'f1']:
        baseline_val = base_overall[metric]
        finetuned_val = ft_overall[metric]
        improvement = finetuned_val - baseline_val
        improvement_str = f"{improvement:+.4f}"
        
        logging.info(f"{metric:<15} {baseline_val:<12.4f} {finetuned_val:<12.4f} {improvement_str:<12}")
    
    # Per-class improvements
    logging.info("\nPER-CLASS AP IMPROVEMENTS:")
    logging.info(f"{'Class':<20} {'Baseline AP':<12} {'Fine-tuned AP':<14} {'Improvement':<12}")
    logging.info(f"{'-'*20} {'-'*12} {'-'*14} {'-'*12}")
    
    for class_name in finetuned_metrics['per_class'].keys():
        if class_name in baseline_metrics['per_class']:
            baseline_ap = baseline_metrics['per_class'][class_name]['ap']
            finetuned_ap = finetuned_metrics['per_class'][class_name]['ap']
            improvement = finetuned_ap - baseline_ap
            improvement_str = f"{improvement:+.4f}"
            
            logging.info(f"{class_name:<20} {baseline_ap:<12.4f} {finetuned_ap:<14.4f} {improvement_str:<12}")
    
    logging.info(f"{'='*80}")


def load_training_history(model_dir):
    """Load training history if available."""
    history_path = os.path.join(model_dir, 'training_history.json')
    if os.path.exists(history_path):
        try:
            with open(history_path, 'r') as f:
                history = json.load(f)
            return history
        except Exception as e:
            logging.warning(f"Could not load training history: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate fine-tuned OWL-ViT model')
    parser.add_argument('--model_dir', default='./owlvit_finetuned_frozen_backbone/best_model_map',
                       help='Path to fine-tuned model directory (e.g., ./owlvit_finetuned_frozen_backbone/best_model_map)')
    parser.add_argument('--dataset_path', default='/mnt/e/Desktop/GLaMM/detection_dataset_yolo',
                       help='Path to YOLO dataset directory containing images/, labels/, and dataset.yaml')
    parser.add_argument('--output_dir', default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    parser.add_argument('--conf_threshold', type=float, default=0.1, 
                       help='Confidence threshold for evaluation')
    parser.add_argument('--iou_threshold', type=float, default=0.5, 
                       help='IoU threshold for evaluation metrics')
    parser.add_argument('--max_text_queries', type=int, default=30, 
                       help='Maximum number of text queries')
    parser.add_argument('--eval_split', default='test', choices=['test', 'val'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--create_visualizations', action='store_true',
                       help='Create visualization images of predictions')
    parser.add_argument('--num_viz_samples', type=int, default=20,
                       help='Number of sample images to visualize')
    parser.add_argument('--compare_baseline', action='store_true',
                       help='Compare with baseline model from same architecture')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file_path = os.path.join(args.output_dir, 'evaluation.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"Starting evaluation of fine-tuned OWL-ViT model")
    logging.info(f"Model directory: {args.model_dir}")
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Using device: {args.device}")
    
    # Verify model directory exists
    if not os.path.exists(args.model_dir):
        logging.error(f"Model directory does not exist: {args.model_dir}")
        return
    
    # Load model and processor
    logging.info("Loading fine-tuned model and processor...")
    try:
        processor = OwlViTProcessor.from_pretrained(args.model_dir)
        model = OwlViTForObjectDetection.from_pretrained(args.model_dir)
        model.to(args.device)
        model.eval()
        logging.info("Model and processor loaded successfully!")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return
    
    # Setup dataset
    try:
        dataset_paths, class_names = setup_dataset_paths(args.dataset_path)
    except Exception as e:
        logging.error(f"Error setting up dataset: {e}")
        return
    
    # Select evaluation split
    if args.eval_split in dataset_paths:
        eval_paths = dataset_paths[args.eval_split]
        logging.info(f"Using {args.eval_split} split for evaluation")
    elif 'test' in dataset_paths:
        eval_paths = dataset_paths['test']
        logging.info(f"Requested {args.eval_split} split not found, using test split")
    elif 'val' in dataset_paths:
        eval_paths = dataset_paths['val']
        logging.info(f"No test split found, using validation split")
    else:
        logging.error("No suitable evaluation split found in dataset")
        return
    
    args.max_text_queries = max(args.max_text_queries, len(class_names))
    
    # Create evaluation dataset
    logging.info("Setting up evaluation dataset...")
    eval_dataset = YOLODataset(
        eval_paths['images'], 
        eval_paths['labels'], 
        class_names, 
        processor
    )
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_fn(batch, processor, class_names, args.max_text_queries),
        pin_memory=True if args.device == 'cuda' else False
    )
    
    logging.info(f"Evaluation dataset ready: {len(eval_dataset)} images")
    
    # Load training history for baseline comparison
    training_history = load_training_history(Path(args.model_dir).parent)
    baseline_metrics = None
    if training_history and 'baseline_metrics' in training_history:
        baseline_metrics = training_history['baseline_metrics']
        logging.info("Loaded baseline metrics from training history")
    
    # Run evaluation
    start_time = time.time()
    logging.info("Starting inference and evaluation...")
    
    predictions, ground_truths = inference_and_evaluate(
        model, eval_loader, processor, class_names, args.device, 
        conf_threshold=args.conf_threshold
    )
    
    logging.info("Computing evaluation metrics...")
    metrics = evaluate_detection_metrics(
        predictions, ground_truths, class_names, 
        iou_threshold=args.iou_threshold, 
        conf_threshold=args.conf_threshold
    )
    
    eval_time = time.time() - start_time
    logging.info(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Print results
    print_evaluation_results(metrics, model_name="Fine-tuned Model")
    
    # Compare with baseline if available
    if args.compare_baseline and baseline_metrics:
        compare_with_baseline(metrics, baseline_metrics)
    
    # Create visualizations if requested
    if args.create_visualizations:
        viz_dir = os.path.join(args.output_dir, 'visualizations')
        visualize_predictions(
            predictions, ground_truths, class_names, eval_dataset,
            viz_dir, num_samples=args.num_viz_samples, 
            conf_threshold=args.conf_threshold
        )
    
    # Save detailed results
    results = {
        'metrics': metrics,
        'evaluation_config': {
            'model_dir': args.model_dir,
            'dataset_path': args.dataset_path,
            'eval_split': args.eval_split,
            'conf_threshold': args.conf_threshold,
            'iou_threshold': args.iou_threshold,
            'batch_size': args.batch_size,
            'num_images': len(eval_dataset),
            'class_names': class_names,
            'evaluation_time_seconds': eval_time
        },
        'baseline_metrics': baseline_metrics,
        'training_history': training_history
    }
    
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"\n{'='*80}")
    logging.info("EVALUATION COMPLETED!")
    logging.info(f"Results saved to: {results_path}")
    logging.info(f"Logs saved to: {log_file_path}")
    if args.create_visualizations:
        logging.info(f"Visualizations saved to: {viz_dir}")
    logging.info(f"Total evaluation time: {eval_time:.2f} seconds")
    logging.info(f"{'='*80}")


if __name__ == "__main__":
    main()