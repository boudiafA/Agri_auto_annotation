import json
import os
import argparse
from group_objects_utils import *
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
from affordances.add_affordance import convert_label_to_category, get_affordances
from affordances.category_descriptions import categories as affordance_categories
import warnings
import numpy as np
import time
from PIL import Image
from pathlib import Path


def numpy_warning_filter(message, category, filename, lineno, file=None, line=None):
    return 'numpy' in filename


warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')


def parse_args():
    parser = argparse.ArgumentParser(description="Process Level 1 Detection Results")

    parser.add_argument("--image_dir_path", required=False, default="images",
                       help="Path to the images directory")
    parser.add_argument("--output_dir_path", required=False, default="predictions/level-1-processed",
                       help="Output directory for processed predictions")
    parser.add_argument("--raw_dir_path", required=False, default="predictions/level-1-raw",
                       help="Input directory with raw merged predictions")
    parser.add_argument("--depth_map_dir", required=False, default="predictions/level-1-raw",
                       help="Directory containing depth maps")

    args = parser.parse_args()
    return args


# Model configurations
all_model_keys = ['eva-02-01', 'co_detr', 'eva-02-02', 'ov_sam', 'owl_vit', 'pomp', 'grit']
tag_model_keys = ['ram', 'tag2text']
scene_model_keys = ['landmark']
all_supported_models = all_model_keys + tag_model_keys + scene_model_keys

thresholds = {
    "eva-02-02": 0.6, "co_detr": 0.2, "eva-02-01": 0.6, 
    "ov_sam": 0.3, "owl_vit": 0.1, "pomp": 0.3, "grit": 0.3
}

single_person_threshold = {
    "eva-02-02": 0.75, "co_detr": 0.5, "eva-02-01": 0.75, 
    "owl_vit": 0.2, "pomp": 0.6, "grit": 0.6
}


def json_serializable(obj):
    """Convert objects to JSON serializable format"""
    if isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'tolist'):  # numpy arrays
        return obj.tolist()
    else:
        return obj


def compute_all_iou_and_dice(bboxes):
    """Compute IoU and Dice matrices for all bounding box pairs"""
    num_boxes = len(bboxes)
    iou_matrix = np.zeros((num_boxes, num_boxes))
    dice_matrix = np.zeros((num_boxes, num_boxes))

    for i in range(num_boxes):
        for j in range(i + 1, num_boxes):
            iou_val = IoU(bboxes[i]['bbox'], bboxes[j]['bbox'])
            dice_val = dice_score_bbox(bboxes[i]['bbox'], bboxes[j]['bbox'])
            iou_matrix[i, j], iou_matrix[j, i] = iou_val, iou_val
            dice_matrix[i, j], dice_matrix[j, i] = dice_val, dice_val

    return iou_matrix, dice_matrix


def bbox_size(bbox):
    """Calculate the area of a bounding box"""
    return max(0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))


def add_big_bboxes(image, objects, floating_objects, image_dir_path):
    """Move large floating objects to main objects list"""
    try:
        img_path = os.path.join(image_dir_path, image)
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found: {img_path}")
            return objects, floating_objects
            
        img = Image.open(img_path)
        image_width, image_height = img.size
        half_image_size = 0.5 * image_height * image_width

        if not (objects + floating_objects):
            return objects, floating_objects

        # Compute the biggest bounding box size
        biggest_bbox_size = max([bbox_size(obj['bbox']) for obj in objects + floating_objects], default=0)

        # Move qualifying floating objects to main objects
        for float_obj in floating_objects.copy():
            obj_size = bbox_size(float_obj['bbox'])
            if obj_size > half_image_size and obj_size > biggest_bbox_size:
                objects.append(float_obj)
                floating_objects.remove(float_obj)

    except Exception as e:
        print(f"Warning: Error processing image {image}: {e}")

    return objects, floating_objects


def group_bounding_boxes(models_data, affordance_label_to_category, image, image_dir_path):
    """Group bounding boxes from different models based on IoU/Dice overlap"""
    checked_bboxes = set()
    objects = []
    floating_objects = []
    floating_attributes = []
    box_threshold_iou = 0.65
    box_threshold_dice = 0.75

    # Process detection models for bbox grouping
    for i, model_name in enumerate(all_model_keys):
        if model_name not in models_data:
            continue
            
        model_predictions = models_data[model_name]
        if not isinstance(model_predictions, list):
            continue

        for j in range(len(model_predictions)):
            if (model_name, j) in checked_bboxes:
                continue

            pred = model_predictions[j]
            if not isinstance(pred, dict) or 'bbox' not in pred:
                continue

            bbox_i = pred['bbox']
            score_i = pred.get('score', 0.0)
            label_i = pred.get('label', pred.get('description', ''))

            # Initialize new object
            new_object = {
                'bbox': bbox_i, 
                'labels': set(), 
                'score': score_i, 
                'attributes': set()
            }

            # Add label or attribute based on model type
            if model_name != 'grit':
                new_object['labels'].add(label_i)
            else:
                new_object['attributes'].add(label_i)

            # Look for overlapping predictions from other models
            is_matched = False
            for k, other_model_name in enumerate(all_model_keys[i + 1:]):
                if other_model_name not in models_data:
                    continue
                    
                other_predictions = models_data[other_model_name]
                if not isinstance(other_predictions, list):
                    continue

                start_l = j + 1 if other_model_name == model_name else 0
                
                for l in range(start_l, len(other_predictions)):
                    if (other_model_name, l) in checked_bboxes:
                        continue

                    other_pred = other_predictions[l]
                    if not isinstance(other_pred, dict) or 'bbox' not in other_pred:
                        continue

                    bbox_k = other_pred['bbox']
                    label_k = other_pred.get('label', other_pred.get('description', ''))

                    # Check for sufficient overlap
                    try:
                        iou_val = IoU(bbox_i, bbox_k)
                        dice_val = dice_score_bbox(bbox_i, bbox_k)
                        
                        if (iou_val >= box_threshold_iou and dice_val >= box_threshold_dice):
                            is_matched = True

                            if other_model_name != 'grit':
                                new_object['labels'].add(label_k)
                            else:
                                new_object['attributes'].add(label_k)

                            checked_bboxes.add((other_model_name, l))
                    except Exception as e:
                        print(f"Warning: Error computing overlap for {model_name}-{other_model_name}: {e}")
                        continue

            # Categorize the object based on matching and model-specific rules
            if is_matched:
                objects.append(new_object)
            elif model_name != 'grit':
                person_threshold = single_person_threshold.get(model_name, 0)
                original_threshold = thresholds.get(model_name, 0)
                
                # Special handling for person detection
                if (model_name in ['eva-02-01', 'co_detr', 'eva-02-02'] and 
                    label_i == 'person' and score_i > person_threshold):
                    objects.append(new_object)
                elif (model_name in ['eva-02-01', 'co_detr', 'eva-02-02'] and 
                      label_i == 'person' and score_i > original_threshold):
                    floating_objects.append(new_object)
                elif label_i != 'person':
                    floating_objects.append(new_object)
            else:
                floating_attributes.append(new_object)

            # Clean up attributes if empty
            if len(new_object['attributes']) == 0:
                new_object['attributes'] = None

            checked_bboxes.add((model_name, j))

    # Handle tag models (ram, tag2text) - add to floating attributes
    for model_name in tag_model_keys:
        if model_name in models_data:
            tag_data = models_data[model_name]
            tags = []
            scores = []
            
            if isinstance(tag_data, dict) and 'tags' in tag_data:
                tags = tag_data.get('tags', [])
                scores = tag_data.get('scores', [])
            elif isinstance(tag_data, list) and len(tag_data) > 0:
                if isinstance(tag_data[0], dict) and 'tags' in tag_data[0]:
                    tags = tag_data[0].get('tags', [])
                    scores = tag_data[0].get('scores', [])
            
            threshold = 0.3  # Threshold for tag models
            for tag, score in zip(tags, scores[:len(tags)]):
                if score > threshold:
                    tag_object = {
                        'bbox': [0, 0, 0, 0],
                        'labels': set(),
                        'score': score,
                        'attributes': {tag},
                        'id': None
                    }
                    floating_attributes.append(tag_object)

    # Assign object IDs
    object_id = 0
    for obj in objects:
        obj['id'] = object_id
        object_id += 1

    for obj in floating_objects:
        obj['id'] = object_id
        object_id += 1
    
    for obj in floating_attributes:
        if obj.get('id') is None:
            obj['id'] = object_id
            object_id += 1

    # Convert sets to lists for JSON serialization
    for obj_list in [objects, floating_objects, floating_attributes]:
        for obj in obj_list:
            obj = {k: json_serializable(v) for k, v in obj.items()}

    # Handle landmark data
    landmark_data = models_data.get('landmark', {})
    if isinstance(landmark_data, list) and len(landmark_data) > 0:
        if isinstance(landmark_data[0], dict):
            landmark_data = landmark_data[0]
        else:
            landmark_data = {}
    elif not isinstance(landmark_data, dict):
        landmark_data = {}

    # Create final structure
    processed_dict = {
        'objects': [{k: json_serializable(v) for k, v in obj.items()} for obj in objects],
        'floating_objects': [{k: json_serializable(v) for k, v in obj.items()} for obj in floating_objects],
        'floating_attributes': [{k: json_serializable(v) for k, v in obj.items()} for obj in floating_attributes],
        'landmark': landmark_data
    }

    # Add affordances
    processed_dict = get_affordances(processed_dict, affordance_label_to_category)

    return processed_dict


def get_threshold_for_prediction(model, label):
    """Get the appropriate threshold for a model-label combination"""
    if model in ['eva-02-01', 'co_detr', 'eva-02-02'] and label == 'person':
        return 0.2
    return thresholds.get(model, 0)


def process_prediction(prediction, model):
    """Process a single prediction with threshold filtering"""
    if not isinstance(prediction, dict):
        return None
        
    score = prediction.get('score', 0.0)
    label = prediction.get('description', prediction.get('label', ''))
    bbox = prediction.get('bbox', [])
    
    if not bbox or len(bbox) != 4:
        return None
        
    threshold = get_threshold_for_prediction(model, label)
    
    if score > threshold:
        result = {
            'bbox': bbox, 
            'label': label, 
            'score': score
        }
        # Preserve additional fields like sam_mask
        for key in prediction:
            if key not in ['bbox', 'label', 'score', 'description']:
                result[key] = prediction[key]
        return result
    return None


def process_image(raw_file_path, image_dir_path, output_dir_path):
    """Process a single image's predictions - worker function for multiprocessing"""
    try:
        with open(raw_file_path, 'r') as f:
            merged_contents = json.load(f)

        if not merged_contents:
            print(f"Warning: Empty or invalid JSON in {raw_file_path}")
            return None, None

        # Get image name (first key in the JSON)
        image_names = list(merged_contents.keys())
        if not image_names:
            print(f"Warning: No image data found in {raw_file_path}")
            return None, None
            
        image = image_names[0]  # Use first image
        
        # Process predictions from all models
        all_bbox_dict = {image: {}}
        
        for model in merged_contents[image]:
            if model in all_supported_models:
                # Apply threshold filtering for detection models
                if model not in tag_model_keys and model not in scene_model_keys:
                    processed_preds = []
                    for pred in merged_contents[image][model]:
                        processed_pred = process_prediction(pred, model)
                        if processed_pred:
                            processed_preds.append(processed_pred)
                    all_bbox_dict[image][model] = processed_preds
                else:
                    # Keep tag and scene model data as-is
                    all_bbox_dict[image][model] = merged_contents[image][model]
            else:
                # Keep other model data as-is for backward compatibility
                all_bbox_dict[image][model] = merged_contents[image][model]

        # Initialize affordances
        affordance_label_to_category = convert_label_to_category(affordance_categories)
        
        # Group bounding boxes
        processed_level_1_dict = group_bounding_boxes(
            all_bbox_dict[image], 
            affordance_label_to_category, 
            image, 
            image_dir_path
        )

        # Normalize gender labels to 'person'
        for pred in processed_level_1_dict['objects']:
            if 'labels' in pred and isinstance(pred['labels'], list):
                pred['labels'] = [
                    'person' if label.lower() in ['man', 'woman', 'men', 'women'] 
                    else label for label in pred['labels']
                ]

        # Save processed result
        image_base_name = os.path.splitext(os.path.basename(image))[0]
        output_file_path = os.path.join(output_dir_path, f'{image_base_name}.json')
        
        with open(output_file_path, 'w') as f:
            json.dump({image_base_name: processed_level_1_dict}, f, indent=2)

        return image, processed_level_1_dict

    except Exception as e:
        print(f"Error processing {raw_file_path}: {e}")
        return None, None


def main():
    start_time = time.time()
    args = parse_args()
    
    # Validate directories
    if not os.path.exists(args.raw_dir_path):
        print(f"Error: Raw directory does not exist: {args.raw_dir_path}")
        return
        
    if not os.path.exists(args.image_dir_path):
        print(f"Error: Image directory does not exist: {args.image_dir_path}")
        return

    # Create output directory
    os.makedirs(args.output_dir_path, exist_ok=True)

    print(f'Loading raw predictions from {args.raw_dir_path}')
    
    # Find all raw prediction files
    raw_files = [f for f in os.listdir(args.raw_dir_path) if f.endswith('_level_1_raw.json')]
    
    if not raw_files:
        print(f"No raw prediction files found in {args.raw_dir_path}")
        print("Expected files with format: *_level_1_raw.json")
        return

    # Filter out already processed files
    raw_file_paths = []
    for file in raw_files:
        image_name = file.replace("_level_1_raw.json", "")
        processed_json_path = os.path.join(args.output_dir_path, f'{image_name}.json')
        if not os.path.exists(processed_json_path):
            raw_file_paths.append(os.path.join(args.raw_dir_path, file))

    if not raw_file_paths:
        print("All files already processed!")
        return

    print(f'Processing {len(raw_file_paths)} raw prediction files')

    # Create worker function with arguments bound using partial
    worker_func = partial(process_image, 
                         image_dir_path=args.image_dir_path,
                         output_dir_path=args.output_dir_path)

    # Process files in parallel
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(worker_func, raw_file_paths), 
            total=len(raw_file_paths),
            desc="Processing images"
        ))

    # Count successful processing
    successful = sum(1 for result in results if result[0] is not None)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f'\033[92mProcessing completed!\033[0m')
    print(f'Successfully processed: {successful}/{len(raw_file_paths)} files')
    print(f'Time taken: {elapsed_time:.2f} seconds')
    print(f'Output directory: {args.output_dir_path}')


if __name__ == "__main__":
    main()