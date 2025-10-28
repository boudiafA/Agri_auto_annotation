import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--input_dir_path", required=False,
                        default="predictions/dense_captions_grounded")
    parser.add_argument("--sam_json_dir_path", required=False,
                        default="predictions/all_sam_annotations",
                        help="path to the directory containing all sam annotation files.")
    parser.add_argument("--eva_02_pred_dir_path", required=False,
                        default="predictions/eva-02-01")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/level-3-processed")

    parser.add_argument("--num_processes", required=False, type=int, default=32)

    args = parser.parse_args()

    return args


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def calculate_dice(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    dice_score = (2. * interArea) / float(boxAArea + boxBArea)
    return dice_score


def convert(bbox):
    x1, y1, w, h = bbox
    return [x1, y1, x1 + w, y1 + h]


def create_annotation(args, paths):
    for path in tqdm(paths):
        try:
            annotation_path, sam_annotation_path, eva_annotation_path = path
            # Load annotation file
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Collecting raw predictions from all models based on thresholds
            image = list(data.keys())[0]
            
            # Check if SAM directory exists and if the specific SAM annotation file exists
            sam_data = None
            sam_annotations = []
            if args.sam_json_dir_path:
                if os.path.exists(sam_annotation_path):
                    try:
                        # Load SAM annotation file
                        with open(sam_annotation_path, 'r') as f:
                            sam_data = json.load(f)
                        
                        # Extract SAM annotations from the new structure
                        # The image key might be with or without .jpg extension
                        image_key = None
                        for key in sam_data.keys():
                            if key.endswith('.jpg') or key.endswith('.png') or key.endswith('.jpeg'):
                                image_key = key
                                break
                        
                        if image_key and 'sam' in sam_data[image_key]:
                            sam_annotations = sam_data[image_key]['sam']
                        else:
                            print(f"Warning: Could not find SAM annotations in expected structure for {sam_annotation_path}")
                            sam_annotations = []
                            
                    except Exception as e:
                        print(f"Warning: Could not load SAM annotation file {sam_annotation_path}: {e}")
                        sam_annotations = []
                else:
                    print(f"Warning: SAM annotation file not found: {sam_annotation_path}")
                    sam_annotations = []
            
            # Load EVA annotation file if it exists
            eva_data = None
            if os.path.exists(eva_annotation_path):
                try:
                    with open(eva_annotation_path, 'r') as f:
                        eva_data = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load EVA annotation file {eva_annotation_path}: {e}")
                    eva_data = None
            else:
                print(f"Warning: EVA annotation file not found: {eva_annotation_path}")
                eva_data = None

            # Add masks to the objects
            for obj_category in ['objects', 'floating_objects']:
                if obj_category not in data[image]:
                    continue
                    
                for obj in data[image][obj_category]:
                    bbox = obj['bbox']
                    
                    # Try to find masks from SAM annotations first if available
                    if sam_annotations:
                        closest_iou = -1
                        closest_dice = -1
                        closest_segmentation = None
                        closest_bbox = None
                        for annotation in sam_annotations:
                            bbox_sam = annotation['bbox']
                            bbox_sam = convert(bbox_sam)
                            iou = calculate_iou(bbox, bbox_sam)
                            dice_score = calculate_dice(bbox, bbox_sam)
                            if iou > closest_iou and dice_score > closest_dice:
                                closest_iou = iou
                                closest_dice = dice_score
                                closest_bbox = bbox_sam
                                closest_segmentation = annotation['segmentation']

                        # If the condition is satisfied, replace bbox and add segmentation to the detail
                        # Found this object in SAM masks
                        if closest_iou >= box_threshold_iou and closest_dice >= box_threshold_dice:
                            obj['segmentation'] = closest_segmentation
                            obj['bbox'] = closest_bbox
                            obj['segmentation_source'] = "SAM"
                            continue  # Skip EVA processing if SAM match found
                    
                    # If SAM didn't provide a match or isn't available, try EVA
                    if eva_data:
                        closest_iou = -1
                        closest_dice = -1
                        closest_segmentation = None
                        closest_bbox = None
                        for annotation in eva_data[image]["eva-02-01"]:
                            bbox_eva = annotation['bbox']
                            iou = calculate_iou(bbox, bbox_eva)
                            dice_score = calculate_dice(bbox, bbox_eva)
                            if iou > closest_iou and dice_score > closest_dice:
                                closest_iou = iou
                                closest_dice = dice_score
                                closest_bbox = bbox_eva
                                closest_segmentation = annotation['sam_mask']

                        # If the condition is satisfied, replace bbox and add segmentation to the detail
                        # Found this object in EVA masks
                        if closest_iou >= box_threshold_iou and closest_dice >= box_threshold_dice:
                            obj['segmentation'] = closest_segmentation
                            obj['bbox'] = closest_bbox
                            obj['segmentation_source'] = "EVA"

            image_name = os.path.splitext(image)[0]
            output_file_path = os.path.join(args.output_dir_path, f'{image_name}.json')

            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)

        except Exception as e:
            print(f"Error processing file {path}: {e}")
            import traceback
            traceback.print_exc()


def split_list(input_list, n):
    """Split a list into 'n' parts using numpy."""
    arrays = np.array_split(np.array(input_list), n)
    return [arr.tolist() for arr in arrays]


if __name__ == "__main__":
    start_time = time.time()
    args = parse_args()

    input_dir_path = args.input_dir_path
    sam_json_dir_path = args.sam_json_dir_path
    eva_02_pred_dir_path = args.eva_02_pred_dir_path
    output_dir_path = args.output_dir_path

    box_threshold_iou = 0.65
    box_threshold_dice = 0.75

    os.makedirs(output_dir_path, exist_ok=True)

    raw_files = os.listdir(input_dir_path)
    raw_file_paths = []
    for file in raw_files:
        processed_json_path = os.path.join(output_dir_path, f'{file}.json')
        if not os.path.exists(processed_json_path):
            raw_file_paths.append([os.path.join(input_dir_path, file),
                                   os.path.join(sam_json_dir_path, file) if sam_json_dir_path else None, 
                                   os.path.join(eva_02_pred_dir_path, file)])

    raw_file_paths_splits = split_list(raw_file_paths, n=args.num_processes)
    task_args = [(args, raw_file) for raw_file in raw_file_paths_splits]

    with Pool() as pool:
        pool.starmap(create_annotation, task_args)
    
    end_time = time.time()
    print(f"\nProcessing completed in {end_time - start_time:.2f} seconds")