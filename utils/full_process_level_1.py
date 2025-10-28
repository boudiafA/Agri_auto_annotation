# python full_process_level_1.py --datasets_root_path "/mnt/e/Desktop/AgML/datasets_sorted/detection" --output_root_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection" --num_processes 8

import argparse
import os
import numpy as np
import json
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import warnings
import time
from PIL import Image

# Import functions from the original scripts
from merge_json_level_1_with_nms import (
    self_nms_with_score_filter, 
    remove_segmentation_masks,
    MODEL_NAMES
)
from prepare_level_1 import (
    group_bounding_boxes,
    process_prediction,
    get_threshold_for_prediction,
    all_model_keys,
    thresholds,
    single_person_threshold
)
from group_objects_utils import json_serializable
from affordances.add_affordance import convert_label_to_category, get_affordances
from affordances.category_descriptions import categories as affordance_categories

def numpy_warning_filter(message, category, filename, lineno, file=None, line=None):
    return 'numpy' in filename

warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

def parse_args():
    parser = argparse.ArgumentParser(description="Process datasets to create level-1-processed outputs")
    
    parser.add_argument("--datasets_root_path", required=True, 
                        help="Root path containing dataset folders with images subfolders")
    parser.add_argument("--output_root_path", required=True,
                        help="Root path where processed datasets will be saved")
    parser.add_argument("--num_processes", type=int, default=16,
                        help="Number of processes for parallel processing")
    parser.add_argument("--dataset_filter", type=str, default=None,
                        help="Process only datasets containing this string in name")
    
    return parser.parse_args()

def merge_predictions_for_image(image_name, input_dataset_path, output_dataset_path):
    """
    Merge predictions from all models for a single image
    Adapted from merge_json_level_1_with_nms.py worker function
    """
    # Create output structure
    all_data = {}
    missing_files = []
    found_models = []
    
    # Try different possible image extensions
    possible_extensions = ['.jpg', '.png', '.jpeg']
    found_data = False
    
    for ext in possible_extensions:
        image_name_with_extension = f"{image_name}{ext}"
        
        # Try to load data for each model from output_dataset_path
        for model in MODEL_NAMES:
            json_file_path = f"{output_dataset_path}/{model}/{image_name}.json"
            if not os.path.exists(json_file_path):
                missing_files.append(f"{model}/{image_name}.json")
                continue
                
            try:
                # Read and parse JSON properly
                with open(json_file_path, 'r') as f:
                    json_data = json.load(f)
                
                # Initialize the image in all_data if not present
                if image_name_with_extension not in all_data:
                    all_data[image_name_with_extension] = {}
                
                # Handle different JSON structures
                model_data = []
                
                # Case 1: {"image.ext": {"model": [...]}}
                for img_key in json_data:
                    if isinstance(json_data[img_key], dict) and model in json_data[img_key]:
                        model_data = json_data[img_key][model]
                        found_data = True
                        found_models.append(model)
                        break
                
                # Process model-specific data
                if model in ['eva-02-01', 'eva-02-02'] and model_data:
                    model_data = remove_segmentation_masks(model_data)

                if model not in ['ram', 'tag2text', 'landmark']:  # Detection model
                    all_data[image_name_with_extension][model] = self_nms_with_score_filter(model_data) \
                        if model_data else []
                else:  # Tags model
                    all_data[image_name_with_extension][model] = model_data
                
            except Exception as e:
                print(f"ERROR: Exception while processing {json_file_path}: {e}")
                missing_files.append(f"{model}/{image_name}.json (corrupted)")
                continue
        
        # If we found data for this extension, no need to try others
        if found_data:
            break
    
    return all_data, missing_files, found_models

def process_single_image(args_tuple):
    """Process a single image through both merge and level-1 processing"""
    image_name, input_dataset_path, output_dataset_path, images_dir = args_tuple
    
    try:
        # Step 1: Merge predictions (from merge_json_level_1_with_nms.py)
        # Read model predictions from output_dataset_path
        merged_data, missing_files, found_models = merge_predictions_for_image(
            image_name, input_dataset_path, output_dataset_path
        )  # FIXED: Unpack the tuple
        
        if not merged_data:
            return None
        
        # Step 2: Process merged data (from prepare_level_1.py)
        image_key = list(merged_data.keys())[0]
        
        # Collecting raw predictions from all models based on thresholds
        all_bbox_dict = {}
        all_bbox_dict[image_key] = {}
        
        for model in merged_data[image_key].keys():
            if model in all_model_keys:
                model_data = [process_prediction(pred, model) for pred in merged_data[image_key][model] if
                              process_prediction(pred, model)]
            else:
                model_data = merged_data[image_key][model]
            all_bbox_dict[image_key][model] = model_data
        
        # Initialize affordance
        affordance_label_to_category = convert_label_to_category(affordance_categories)
        processed_level_1_dict = group_bounding_boxes(all_bbox_dict[image_key], affordance_label_to_category, image_key)
        
        # Replace all man, women, men, woman with person
        for pred in processed_level_1_dict['objects']:
            labels = pred['labels']
            pred['labels'] = ['person' if label.lower() in ['man', 'woman', 'men', 'women'] else label for label in labels]
        
        return image_name, processed_level_1_dict
        
    except Exception as e:
        print(f"Error processing image {image_name}: {e}")
        return None

def get_images_from_dataset(input_dataset_path):
    """Get list of images to process from the input dataset"""
    images_dir = os.path.join(input_dataset_path, "images")
    
    if not os.path.exists(images_dir):
        print(f"Warning: No 'images' directory found in {input_dataset_path}")
        return [], None
    
    # Get all image names (without extension)
    image_files = os.listdir(images_dir)
    image_names = [os.path.splitext(f)[0] for f in image_files 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    return image_names, images_dir

def process_dataset(input_dataset_path, output_dataset_path, num_processes=16):
    """Process a single dataset"""
    dataset_name = os.path.basename(input_dataset_path)
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Get images to process from input dataset path
    image_names, images_dir = get_images_from_dataset(input_dataset_path)
    if not image_names:
        print(f"No images found in dataset {dataset_name}")
        return
    
    # Create output level-1-processed directory in output dataset path
    output_level_1_dir = os.path.join(output_dataset_path, "level-1-processed")
    os.makedirs(output_level_1_dir, exist_ok=True)
    
    # Filter images that haven't been processed yet
    images_to_process = []
    for image_name in image_names:
        output_file = os.path.join(output_level_1_dir, f"{image_name}.json")
        if not os.path.exists(output_file):
            images_to_process.append((image_name, input_dataset_path, output_dataset_path, images_dir))
    
    if not images_to_process:
        print(f"All images in {dataset_name} already processed")
        return
    
    print(f"Processing {len(images_to_process)} images in {dataset_name}")
    
    # Process images in parallel
    with Pool(num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_single_image, images_to_process), 
            total=len(images_to_process),
            desc=f"Processing {dataset_name}"
        ))
    
    # Save results
    saved_count = 0
    for result in results:
        if result is not None:
            image_name, processed_data = result
            output_file = os.path.join(output_level_1_dir, f"{image_name}.json")
            with open(output_file, 'w') as f:
                json.dump({image_name: processed_data}, f)
            saved_count += 1
    
    print(f"Completed {dataset_name}: {saved_count} images processed")

def main():
    start_time = time.time()
    args = parse_args()
    
    input_datasets_root = args.datasets_root_path
    output_datasets_root = args.output_root_path
    
    if not os.path.exists(input_datasets_root):
        print(f"Error: Input datasets root path {input_datasets_root} does not exist")
        return
    
    if not os.path.exists(output_datasets_root):
        print(f"Error: Output datasets root path {output_datasets_root} does not exist")
        return
    
    # Get all dataset directories that exist in both input and output paths
    dataset_pairs = []
    
    # Get datasets from input path (must have 'images' subfolder)
    input_datasets = []
    for item in os.listdir(input_datasets_root):
        item_path = os.path.join(input_datasets_root, item)
        if os.path.isdir(item_path):
            images_path = os.path.join(item_path, "images")
            if os.path.exists(images_path):
                # Apply filter if specified
                if args.dataset_filter is None or args.dataset_filter in item:
                    input_datasets.append(item)
    
    # Check which input datasets also exist in output path
    for dataset_name in input_datasets:
        input_dataset_path = os.path.join(input_datasets_root, dataset_name)
        output_dataset_path = os.path.join(output_datasets_root, dataset_name)
        
        if os.path.exists(output_dataset_path):
            dataset_pairs.append((input_dataset_path, output_dataset_path))
        else:
            print(f"Warning: Dataset '{dataset_name}' found in input but not in output path. Skipping.")
    
    if not dataset_pairs:
        print("No matching dataset directories found in both input and output paths")
        return
    
    print(f"Found {len(dataset_pairs)} datasets to process")
    
    # Process each dataset
    for input_dataset_path, output_dataset_path in dataset_pairs:
        try:
            process_dataset(input_dataset_path, output_dataset_path, args.num_processes)
        except Exception as e:
            dataset_name = os.path.basename(input_dataset_path)
            print(f"Error processing dataset {dataset_name}: {e}")
            continue
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'\033[92m---- Total processing time: {elapsed_time:.2f} seconds ----\033[0m')

if __name__ == "__main__":
    main()