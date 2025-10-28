import argparse
import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing.pool import Pool
from pathlib import Path

MODEL_NAMES = ['co_detr', 'eva-02-01', 'eva-02-02', 'ov_sam', 'grit', 'owl_vit', 'pomp', 'ram', 'tag2text', 'landmark']


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSON predictions from all models")

    parser.add_argument("--image_dir_path", required=True, help="Path to the images directory")
    parser.add_argument("--predictions_dir_path", required=True, help="Path to the predictions directory")
    parser.add_argument("--output_dir_path", required=False, default="predictions/level-1-raw", 
                       help="Output directory for merged predictions")
    parser.add_argument("--num_processes", required=False, type=int, default=32, 
                       help="Number of parallel processes")

    args = parser.parse_args()
    return args


def self_nms_with_score_filter(predictions, score_threshold=0.1, iou_threshold=0.9):
    """Apply Non-Maximum Suppression with score filtering"""
    if not predictions:
        return []
    
    # Filter out detections with score less than threshold
    detections = [det for det in predictions if det.get("score", 0) >= score_threshold]
    
    if not detections:
        return []

    # Sort detections by confidence score (descending)
    detections = sorted(detections, key=lambda x: x.get("score", 0), reverse=True)

    def compute_iou(boxA, boxB):
        """Compute Intersection over Union of two bounding boxes"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        if xB <= xA or yB <= yA:
            return 0.0

        interArea = (xB - xA) * (yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / max(float(boxAArea + boxBArea - interArea), 1e-8)
        return iou

    nms_detections = []
    while detections:
        # Take detection with highest score
        current_detection = detections.pop(0)
        nms_detections.append(current_detection)

        # Remove detections with high overlap
        detections = [
            det for det in detections 
            if compute_iou(current_detection.get("bbox", [0,0,0,0]), det.get("bbox", [0,0,0,0])) <= iou_threshold
        ]

    return nms_detections


def remove_segmentation_masks(predictions):
    """Remove segmentation masks from predictions to reduce file size"""
    cleaned_predictions = []
    for pred in predictions:
        cleaned_pred = {k: v for k, v in pred.items() if k != "sam_mask"}
        cleaned_predictions.append(cleaned_pred)
    return cleaned_predictions


def get_image_extension(image_name, image_dir_path):
    """Find the actual extension of an image file"""
    possible_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    for ext in possible_extensions:
        if os.path.exists(os.path.join(image_dir_path, f"{image_name}{ext}")):
            return ext
    
    # Default to .jpg if not found
    return '.jpg'


def load_model_predictions(model, image_name, predictions_dir_path):
    """Load predictions for a specific model and image"""
    json_file_path = os.path.join(predictions_dir_path, model, f"{image_name}.json")
    
    if not os.path.exists(json_file_path):
        # Return appropriate empty structure based on model type
        if model in ['ram', 'tag2text']:
            return {"tags": [], "scores": []}
        elif model == 'landmark':
            return {"category": "", "fine_category": ""}
        else:
            return []
    
    try:
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        model_data = None
        
        # Structure 1: Direct list of predictions
        if isinstance(json_data, list):
            model_data = json_data
        
        # Structure 2: Dictionary with image keys
        elif isinstance(json_data, dict):
            # Try to find data under image name keys
            for key in json_data:
                if isinstance(json_data[key], dict) and model in json_data[key]:
                    model_data = json_data[key][model]
                    break
                elif isinstance(json_data[key], list):
                    model_data = json_data[key]
                    break
            
            # If no nested structure found, check if root contains model data
            if model_data is None and model in json_data:
                model_data = json_data[model]
        
        # Handle different data types properly
        if model_data is None:
            # Return appropriate empty structure
            if model in ['ram', 'tag2text']:
                return {"tags": [], "scores": []}
            elif model == 'landmark':
                return {"category": "", "fine_category": ""}
            else:
                return []
        
        # For tag and scene models, preserve dict structure; for detection models, return as list
        if model in ['ram', 'tag2text', 'landmark']:
            return model_data
        else:
            # Detection models should be lists
            if isinstance(model_data, list):
                return model_data
            else:
                return []
        
    except Exception as e:
        # Return appropriate empty structure on error
        if model in ['ram', 'tag2text']:
            return {"tags": [], "scores": []}
        elif model == 'landmark':
            return {"category": "", "fine_category": ""}
        else:
            return []


def worker(args):
    """Worker function for parallel processing"""
    args_obj, image_names = args
    
    for image_name in tqdm(image_names, desc="Processing images"):
        output_json_path = os.path.join(args_obj.output_dir_path, f"{image_name}_level_1_raw.json")
        
        # Skip if output already exists
        if os.path.exists(output_json_path):
            continue
        
        # Determine the correct image extension
        image_extension = get_image_extension(image_name, args_obj.image_dir_path)
        image_name_with_extension = f"{image_name}{image_extension}"
        
        # Initialize output structure
        all_data = {image_name_with_extension: {}}
        
        # Process each model
        for model in MODEL_NAMES:
            model_predictions = load_model_predictions(model, image_name, args_obj.predictions_dir_path)
            
            # Remove segmentation masks from EVA models
            if model in ['eva-02-01', 'eva-02-02'] and isinstance(model_predictions, list):
                model_predictions = remove_segmentation_masks(model_predictions)
            
            # Apply NMS for detection models only
            if model not in ['ram', 'tag2text', 'landmark']:
                if isinstance(model_predictions, list):
                    processed_predictions = self_nms_with_score_filter(model_predictions)
                else:
                    processed_predictions = []
            else:
                # Keep tag and scene models as-is
                processed_predictions = model_predictions
            
            all_data[image_name_with_extension][model] = processed_predictions
        
        # Save merged predictions
        try:
            with open(output_json_path, 'w') as f:
                json.dump(all_data, f, indent=2)
        except Exception as e:
            print(f"Error saving {output_json_path}: {e}")


def split_list(input_list, n):
    """Split a list into n roughly equal parts"""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir_path, exist_ok=True)
    
    # Get all image names (without extensions)
    image_files = [f for f in os.listdir(args.image_dir_path) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'))]
    
    # Remove extensions to get base names
    all_image_names = [os.path.splitext(img)[0] for img in image_files]
    all_image_names = list(set(all_image_names))  # Remove duplicates
    
    if not all_image_names:
        print(f"No images found in {args.image_dir_path}")
        return
    
    print(f"Found {len(all_image_names)} unique images to process")
    
    # Split work among processes
    all_tasks_image_names_list = split_list(all_image_names, n=args.num_processes)
    task_args = [(args, task_image_names) for task_image_names in all_tasks_image_names_list]
    
    # Process in parallel
    print(f"Starting parallel processing with {args.num_processes} processes...")
    with Pool(args.num_processes) as pool:
        pool.map(worker, task_args)
    
    print(f"Merged predictions saved to: {args.output_dir_path}")


if __name__ == "__main__":
    main()