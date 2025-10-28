# python unified_level2_processor.py --datasets_root "/mnt/e/Desktop/AgML/datasets_sorted/detection" --predictions_root "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection"

import argparse
import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import nltk
from nltk.corpus import words
import re
from difflib import SequenceMatcher
from pathlib import Path

# Download required NLTK data
try:
    nltk.download("words", quiet=True)
    word_set = set(words.words())
except:
    word_set = set()

MODEL_NAMES = ['blip2', 'llava', 'mdetr-re']
ALL_GROUNDING_MODELS = ['mdetr-re']
ALL_CAPTION_MODELS = ['blip2', 'llava']

# Supported image extensions
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Dataset Level-2 Processing")
    
    # Input/Output paths
    parser.add_argument("--datasets_root", required=True,
                        help="Root directory containing all dataset folders")
    parser.add_argument("--predictions_root", required=True,
                        help="Root directory where model predictions are stored and outputs will be saved")
    
    # Processing options
    parser.add_argument("--num_processes", type=int, default=8,
                        help="Number of parallel processes")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip processing if output file already exists")
    
    args = parser.parse_args()
    return args


# ==================== DATASET DISCOVERY ====================

def is_ignored_folder(folder_name):
    """Check if folder should be ignored"""
    ignored = {'visualizations', 'annotations', 'level-2-processed', 'level-1-processed'}
    return folder_name.lower() in ignored


def discover_images_in_dataset(dataset_path):
    """
    Discover all images in a dataset.
    Returns a dict with structure: {relative_path: full_path}
    """
    images = {}
    dataset_path = Path(dataset_path)
    
    # Check if there's an 'images' folder (detection dataset)
    images_folder = dataset_path / 'images'
    if images_folder.exists() and images_folder.is_dir():
        # Detection dataset structure
        for img_file in images_folder.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                rel_path = img_file.name
                images[rel_path] = str(img_file)
    else:
        # Classification dataset structure - check all subdirectories
        for item in dataset_path.iterdir():
            if item.is_dir() and not is_ignored_folder(item.name):
                # This is a class folder
                for img_file in item.iterdir():
                    if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
                        rel_path = f"{item.name}/{img_file.name}"
                        images[rel_path] = str(img_file)
    
    return images


def discover_all_datasets(datasets_root):
    """
    Discover all datasets and their images.
    Returns: {dataset_name: {relative_path: full_path}}
    """
    datasets = {}
    datasets_root = Path(datasets_root)
    
    if not datasets_root.exists():
        print(f"Error: Datasets root not found: {datasets_root}")
        return datasets
    
    for dataset_dir in datasets_root.iterdir():
        if dataset_dir.is_dir():
            dataset_name = dataset_dir.name
            images = discover_images_in_dataset(dataset_dir)
            if images:
                datasets[dataset_name] = images
                print(f"Found dataset '{dataset_name}' with {len(images)} images")
    
    return datasets


# ==================== UTILITY FUNCTIONS ====================

def is_file_empty(file_path):
    """Check if a file is empty"""
    try:
        return os.path.getsize(file_path) == 0
    except:
        return True


def IoU(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def dice_score_bbox(box1, box2):
    """Calculate Dice score for two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return (2 * intersection) / (area1 + area2) if (area1 + area2) > 0 else 0


def nms_same_model(boxes, labels, threshold=0.9):
    """Non-maximum suppression for same model predictions"""
    if len(boxes) == 0:
        return []
    
    indices = list(range(len(boxes)))
    keep = []
    
    while indices:
        i = indices[0]
        keep.append(i)
        indices = indices[1:]
        
        indices_to_remove = []
        for j in indices:
            if labels[i] == labels[j] and IoU(boxes[i], boxes[j]) > threshold:
                indices_to_remove.append(j)
        
        indices = [idx for idx in indices if idx not in indices_to_remove]
    
    return keep


def normalize_phrase(phrase):
    """Normalize a phrase by lowercasing and removing extra whitespace"""
    return ' '.join(phrase.lower().split())


def tokenize(text):
    """Tokenize text by extracting words"""
    return re.findall(r'\b\w+\b', text)


# ==================== PREDICTION LOADING ====================

def get_prediction_path(predictions_root, dataset_name, rel_path, model_name):
    """
    Get the path to a prediction file.
    Tries multiple possible locations.
    """
    # Remove extension and get base name
    img_name = Path(rel_path).stem
    
    # Try different possible structures
    possible_paths = [
        # Structure 1: predictions_root/dataset_name/model_name/img_name.json
        Path(predictions_root) / dataset_name / model_name / f"{img_name}.json",
        # Structure 2: predictions_root/model_name/dataset_name/img_name.json
        Path(predictions_root) / model_name / dataset_name / f"{img_name}.json",
        # Structure 3: predictions_root/dataset_name/model_name/rel_path_dir/img_name.json
        Path(predictions_root) / dataset_name / model_name / Path(rel_path).parent / f"{img_name}.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


def load_model_predictions(predictions_root, dataset_name, rel_path):
    """
    Load predictions from all models for a given image.
    Returns: dict with model predictions or None if not found
    """
    img_name = Path(rel_path).stem
    img_name_with_ext = Path(rel_path).name
    
    all_data = {img_name_with_ext: {}}
    
    for model in MODEL_NAMES:
        pred_path = get_prediction_path(predictions_root, dataset_name, rel_path, model)
        
        if pred_path is None or is_file_empty(pred_path):
            continue
        
        try:
            with open(pred_path, 'r') as f:
                json_data = json.load(f)
            
            # Extract data based on model type
            if model in ['blip2', 'llava']:
                # Try to extract from nested structure
                if img_name_with_ext in json_data and model in json_data[img_name_with_ext]:
                    model_data = json_data[img_name_with_ext][model]
                    if model == 'llava' and isinstance(model_data, dict) and model in model_data:
                        model_data = model_data[model]
                    all_data[img_name_with_ext][model] = model_data
                else:
                    # Try direct access
                    all_data[img_name_with_ext][model] = json_data
            
            elif "mdetr-re" in model:
                # Extract mdetr-re data
                if img_name_with_ext in json_data and "mdetr-re" in json_data[img_name_with_ext]:
                    mdetr_data = json_data[img_name_with_ext]["mdetr-re"]
                else:
                    mdetr_data = json_data
                
                if "mdetr-re" in all_data[img_name_with_ext]:
                    all_data[img_name_with_ext]["mdetr-re"] = \
                        all_data[img_name_with_ext]["mdetr-re"] | mdetr_data
                else:
                    all_data[img_name_with_ext]["mdetr-re"] = mdetr_data
        
        except Exception as e:
            print(f"Warning: Error loading {pred_path}: {e}")
            continue
    
    # Return None if no valid data was loaded
    if not all_data[img_name_with_ext]:
        return None
    
    return all_data


def get_level1_path(predictions_root, dataset_name, rel_path):
    """Get path to level-1 processed file"""
    img_name = Path(rel_path).stem
    
    possible_paths = [
        Path(predictions_root) / dataset_name / "level-1-processed" / f"{img_name}.json",
        Path(predictions_root) / "level-1-processed" / dataset_name / f"{img_name}.json",
        Path(predictions_root) / dataset_name / "level-1-processed" / Path(rel_path).parent / f"{img_name}.json",
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None


# ==================== MODEL COMBINATION ====================

def combine_models(image_predictions, nms_threshold=0.9, box_threshold_iou=0.65, 
                   box_threshold_dice=0.75):
    """Combine bounding boxes from different phrase grounding models"""
    combined_data = {}
    model_data = {}
    
    for model_key in ALL_GROUNDING_MODELS:
        if model_key not in image_predictions:
            continue
        model_data[model_key] = {"random caption": image_predictions[model_key]}
        
        # Perform NMS within each model
        for key in model_data[model_key]:
            if not model_data[model_key][key]:
                continue
            model_data[model_key][key] = [
                model_data[model_key][key][i] for i in nms_same_model(
                    np.array([bbox['bbox'] for bbox in model_data[model_key][key]]),
                    np.array([bbox['label'] for bbox in model_data[model_key][key]]), 
                    threshold=nms_threshold
                )
            ]
    
    if not model_data:
        return combined_data
    
    # Combine bounding boxes from all models
    for key in model_data[ALL_GROUNDING_MODELS[0]]:
        # Convert labels to lists
        for bbox in model_data[ALL_GROUNDING_MODELS[0]][key]:
            bbox['label'] = [bbox['label'].strip()] if isinstance(bbox['label'], str) else \
                           [label.strip() for label in bbox['label']]
        
        combined_data[key] = model_data[ALL_GROUNDING_MODELS[0]][key]
        
        for model_key in ALL_GROUNDING_MODELS[1:]:
            if key in model_data.get(model_key, {}):
                for bbox in model_data[model_key][key]:
                    matched = False
                    for combined_bbox in combined_data[key]:
                        if (IoU(np.array(bbox['bbox']), np.array(combined_bbox['bbox'])) > box_threshold_iou and 
                            dice_score_bbox(np.array(bbox['bbox']), np.array(combined_bbox['bbox'])) > box_threshold_dice):
                            
                            bbox['label'] = [bbox['label'].strip()] if isinstance(bbox['label'], str) else \
                                           [label.strip() for label in bbox['label']]
                            combined_bbox['label'] = [combined_bbox['label']] if isinstance(combined_bbox['label'], str) else \
                                                     combined_bbox['label']
                            combined_bbox['label'] = combined_bbox['label'] + bbox['label']
                            matched = True
                            break
                    
                    if not matched:
                        combined_data[key].append(bbox)
    
    return combined_data


def get_relationships(combined_data, level1_contents, match_threshold=0.75):
    """Determine relationships between grounded objects and level-1 objects"""
    
    if 'id_counter' not in level1_contents:
        level1_contents['id_counter'] = len(level1_contents.get('objects', [])) + \
                                        len(level1_contents.get('floating_objects', [])) + 1
    
    relationships = {'object_ids': [], 'floating_object_ids': [], 'grounding': []}
    
    for combined_bbox in combined_data:
        object_matched = False
        floating_object_matched = False
        grounding_entry = None
        
        phrase = combined_bbox.get('phrase', combined_bbox.get('label', ''))
        normalized_label = normalize_phrase(str(phrase))
        
        # Try matching with objects and floating objects
        for object_type in ['objects', 'floating_objects']:
            if object_type not in level1_contents:
                continue
            
            for obj in level1_contents[object_type]:
                if IoU(np.array(combined_bbox['bbox']), np.array(obj['bbox'])) > match_threshold:
                    
                    # Check if phrase already in grounding
                    for grounding in relationships['grounding']:
                        if grounding['phrase'] == normalized_label:
                            grounding_entry = grounding
                            break
                    
                    if object_type == 'objects':
                        relationships['object_ids'].append(obj['id'])
                        object_matched = True
                        if grounding_entry is None:
                            grounding_entry = {'phrase': normalized_label, 'object_ids': [obj['id']]}
                            relationships['grounding'].append(grounding_entry)
                        else:
                            grounding_entry['object_ids'].append(obj['id'])
                    
                    elif object_type == 'floating_objects':
                        relationships['floating_object_ids'].append(obj['id'])
                        floating_object_matched = True
                        if grounding_entry is None:
                            grounding_entry = {'phrase': normalized_label, 'object_ids': [obj['id']]}
                            relationships['grounding'].append(grounding_entry)
                        else:
                            grounding_entry['object_ids'].append(obj['id'])
                        
                        # Add labels
                        obj['labels'] = [obj['labels']] if isinstance(obj.get('labels', []), str) else obj.get('labels', [])
                        combined_bbox['label'] = [combined_bbox['label']] if isinstance(combined_bbox.get('label', []), str) else \
                                                 combined_bbox.get('label', [])
                        obj['labels'] = obj['labels'] + combined_bbox['label']
                    break
        
        # If no match found, add to floating_objects
        if not object_matched and not floating_object_matched:
            if 'floating_objects' not in level1_contents:
                level1_contents['floating_objects'] = []
            
            new_floating_object = {
                'bbox': combined_bbox['bbox'], 
                'labels': [normalized_label], 
                'attributes': None,
                'id': level1_contents['id_counter']
            }
            level1_contents['id_counter'] += 1
            level1_contents['floating_objects'].append(new_floating_object)
            relationships['floating_object_ids'].append(new_floating_object['id'])
            
            for grounding in relationships['grounding']:
                if grounding['phrase'] == normalized_label:
                    grounding_entry = grounding
                    break
            
            if grounding_entry is None:
                grounding_entry = {'phrase': normalized_label, 'object_ids': [new_floating_object['id']]}
                relationships['grounding'].append(grounding_entry)
            else:
                grounding_entry['object_ids'].append(new_floating_object['id'])
    
    return relationships, level1_contents


def update_level1_contents(level1_contents, relationship):
    """Update level1 by moving floating objects with multiple labels to objects"""
    
    if 'floating_objects' not in level1_contents:
        level1_contents['floating_objects'] = []
    if 'objects' not in level1_contents:
        level1_contents['objects'] = []
    
    to_be_moved = [obj for obj in level1_contents['floating_objects'] 
                   if len(obj.get('labels', [])) > 1]
    id_map = {}
    
    for obj in to_be_moved:
        old_id = obj['id']
        level1_contents['objects'].append(obj)
        level1_contents['floating_objects'].remove(obj)
        new_id = len(level1_contents['objects']) - 1
        obj['id'] = new_id
        id_map[old_id] = new_id
    
    # Adjust IDs for remaining floating objects
    for i, float_obj in enumerate(level1_contents['floating_objects'], 
                                  start=len(level1_contents['objects'])):
        old_id = float_obj['id']
        float_obj['id'] = i
        id_map[old_id] = i
    
    # Update relationship IDs
    for grounding in relationship['grounding']:
        grounding['object_ids'] = [id_map.get(obj_id, obj_id) for obj_id in grounding['object_ids']]
    
    all_ids = relationship['object_ids'] + relationship['floating_object_ids']
    relationship['object_ids'], relationship['floating_object_ids'] = set(), set()
    floating_object_start_id = len(level1_contents['objects'])
    
    for obj_id in all_ids:
        new_obj_id = id_map.get(obj_id, obj_id)
        if new_obj_id < floating_object_start_id:
            relationship['object_ids'].add(new_obj_id)
        else:
            relationship['floating_object_ids'].add(new_obj_id)
    
    relationship['object_ids'] = list(relationship['object_ids'])
    relationship['floating_object_ids'] = list(relationship['floating_object_ids'])
    
    return level1_contents, relationship


# ==================== IMAGE PROCESSING ====================

def process_single_image(task_info):
    """
    Process a single image.
    task_info: (dataset_name, rel_path, full_path, predictions_root, skip_existing)
    """
    dataset_name, rel_path, full_path, predictions_root, skip_existing = task_info
    
    try:
        # Determine output path
        img_name = Path(rel_path).stem
        
        # Build output directory structure - place level-2-processed alongside model folders
        if '/' in rel_path:  # Classification dataset with class folders
            class_name = Path(rel_path).parent
            output_dir = Path(predictions_root) / dataset_name / class_name / "level-2-processed"
        else:  # Detection dataset with images folder
            output_dir = Path(predictions_root) / dataset_name / "level-2-processed"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{img_name}.json"
        
        # Skip if exists
        if skip_existing and output_file.exists():
            return (dataset_name, rel_path, "skipped")
        
        # Load model predictions
        merged_predictions = load_model_predictions(predictions_root, dataset_name, rel_path)
        
        if merged_predictions is None:
            return (dataset_name, rel_path, "no_predictions")
        
        # Get image name with extension
        img_name_with_ext = list(merged_predictions.keys())[0]
        image_predictions = merged_predictions[img_name_with_ext]
        
        # Load level-1 content if available
        level1_path = get_level1_path(predictions_root, dataset_name, rel_path)
        
        if level1_path and os.path.exists(level1_path):
            with open(level1_path, 'r') as f:
                level1_data = json.load(f)
                # Extract the actual content (may be nested)
                if img_name in level1_data:
                    level1_pred = level1_data[img_name]
                else:
                    level1_pred = level1_data
        else:
            # Create minimal level-1 structure if not available
            level1_pred = {
                'objects': [],
                'floating_objects': [],
                'id_counter': 0
            }
        
        # Combine models if grounding data exists
        if 'mdetr-re' in image_predictions and image_predictions['mdetr-re']:
            if len(ALL_GROUNDING_MODELS) > 1:
                combined_data = combine_models(image_predictions)
            else:
                combined_data = {'mdetr-re': image_predictions['mdetr-re']}
            
            # Get relationships and update
            relationship, level1_content = get_relationships(
                combined_data['mdetr-re'], level1_pred
            )
            level1_content_updated, relationship = update_level1_contents(
                level1_content, relationship
            )
            level1_content_updated['relationships'] = relationship
        else:
            level1_content_updated = level1_pred
            level1_content_updated['relationships'] = {
                'object_ids': [],
                'floating_object_ids': [],
                'grounding': []
            }
        
        # Create final output
        updated_output = {img_name_with_ext: level1_content_updated}
        
        # Add captions if available
        captions = []
        if 'blip2' in image_predictions:
            captions.append(image_predictions['blip2'])
        if 'llava' in image_predictions:
            captions.append(image_predictions['llava'])
        
        if captions:
            updated_output[img_name_with_ext]['captions'] = captions
        
        # Save output
        with open(output_file, 'w') as f:
            json.dump(updated_output, f, indent=2)
        
        return (dataset_name, rel_path, "success")
    
    except Exception as e:
        return (dataset_name, rel_path, f"error: {str(e)}")


# ==================== MAIN ====================

def main():
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("MULTI-DATASET LEVEL-2 PROCESSOR")
    print("=" * 70)
    print(f"Datasets root: {args.datasets_root}")
    print(f"Predictions root: {args.predictions_root}")
    print(f"Output location: {args.predictions_root}/<dataset>/level-2-processed/")
    print(f"Number of processes: {args.num_processes}")
    print(f"Skip existing: {args.skip_existing}")
    print("=" * 70 + "\n")
    
    overall_start = time.time()
    
    # Discover all datasets and images
    print("Discovering datasets and images...")
    datasets = discover_all_datasets(args.datasets_root)
    
    if not datasets:
        print("\033[91mNo datasets found!\033[0m")
        return
    
    print(f"\nFound {len(datasets)} dataset(s)")
    total_images = sum(len(images) for images in datasets.values())
    print(f"Total images: {total_images}\n")
    
    # Prepare tasks for all images
    tasks = []
    for dataset_name, images in datasets.items():
        for rel_path, full_path in images.items():
            tasks.append((
                dataset_name,
                rel_path,
                full_path,
                args.predictions_root,
                args.skip_existing
            ))
    
    # Process all images
    print(f"Processing {len(tasks)} images with {args.num_processes} processes...\n")
    
    results = {'success': 0, 'skipped': 0, 'no_predictions': 0, 'error': 0}
    
    with Pool(args.num_processes) as pool:
        for dataset_name, rel_path, status in tqdm(
            pool.imap_unordered(process_single_image, tasks),
            total=len(tasks),
            desc="Processing images"
        ):
            if status == "success":
                results['success'] += 1
            elif status == "skipped":
                results['skipped'] += 1
            elif status == "no_predictions":
                results['no_predictions'] += 1
            else:
                results['error'] += 1
                if results['error'] <= 5:  # Show first 5 errors
                    print(f"\n  Error in {dataset_name}/{rel_path}: {status}")
    
    total_time = time.time() - overall_start
    
    # Print summary
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Successfully processed: {results['success']}")
    print(f"Skipped (existing): {results['skipped']}")
    print(f"No predictions found: {results['no_predictions']}")
    print(f"Errors: {results['error']}")
    print(f"Total time: {total_time:.2f} seconds")
    print("=" * 70)
    
    if results['success'] > 0:
        print(f"\033[92m✓ Processing completed successfully!\033[0m")
    elif results['error'] > 0:
        print(f"\033[91m✗ Processing completed with errors\033[0m")
    else:
        print(f"\033[93m⚠ No images were processed\033[0m")


if __name__ == "__main__":
    main()