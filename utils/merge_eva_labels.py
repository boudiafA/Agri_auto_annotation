import argparse
import json
import os
from tqdm import tqdm
from .util import get_median_depth_mask_box_based_jpg
from .group_objects_utils import json_serializable, person_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Merge EVA Labels")

    parser.add_argument("--level_2_dir_path", required=False,
                        default="predictions/level-2-processed_gpt4roi")
    parser.add_argument("--labels_path", required=False,
                        default="predictions/level-2-processed_eva_clip")
    parser.add_argument("--output_dir_path", required=False,
                        default="predictions/level-2-processed_labelled")

    parser.add_argument("--store_depth", action='store_true')
    parser.add_argument("--depth_map_dir", required=False, default="predictions/midas")

    args = parser.parse_args()

    return args


def compute_depth_region(prediction_list, depth_map_path):
    # Now, let's iterate through each bounding box
    for prediction in prediction_list:
        # Get bounding box and mask from prediction
        bounding_box = prediction['bbox']
        # Use get_median_depth_mask function to get median depth
        median_depth = get_median_depth_mask_box_based_jpg(depth_map_path, bounding_box)
        # Update the prediction with the median depth
        prediction['depth'] = round(float(median_depth), 2)

    return prediction_list


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir_path, exist_ok=True)

    if args.store_depth:
        # Get extract all the depth maps file names
        depth_file_names = os.listdir(args.depth_map_dir)
        image_name_to_depth_file_name = {}
        for depth_file_name in depth_file_names:
            # Get base name without making assumptions about extension
            base_name = depth_file_name.split('_min_')[0]
            # Store mapping for both with and without common extensions
            image_name_to_depth_file_name[base_name] = depth_file_name
            # Also store with .jpg extension if not already present
            if not base_name.endswith('.jpg'):
                image_name_to_depth_file_name[f"{base_name}.jpg"] = depth_file_name

    all_annotations = os.listdir(args.level_2_dir_path)
    
    for annotation_file in tqdm(all_annotations):
        # Get base name without extension
        base_name = os.path.splitext(annotation_file)[0]
        
        # Load JSON contents
        json_path = os.path.join(args.level_2_dir_path, annotation_file)
        json_contents = json.load(open(json_path, 'r'))
        
        # Get the actual key from the JSON (don't assume the extension)
        # The key should be one of the keys in the json_contents dict
        if len(json_contents.keys()) == 0:
            continue
            
        # Find the actual image key in the JSON
        image_key = None
        for key in json_contents.keys():
            key_base = os.path.splitext(key)[0]
            if key_base == base_name:
                image_key = key
                break
        
        # If no matching key found, skip this file
        if image_key is None:
            print(f"Warning: No matching key found in {annotation_file}")
            continue

        # Code to handle storing of depth
        if args.store_depth:
            objects = json_contents[image_key]['objects']
            
            # Get the depth file using the image key
            depth_file = image_name_to_depth_file_name.get(image_key) or image_name_to_depth_file_name.get(base_name)
            
            if depth_file:
                depth_map_path = os.path.join(args.depth_map_dir, depth_file)
                
                if len(objects) > 0:
                    objects = compute_depth_region(objects, depth_map_path)
                    objects = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in objects]
                    json_contents[image_key]['objects'] = objects
                else:
                    floating_objects = json_contents[image_key]['floating_objects']
                    if len(floating_objects) > 0:
                        floating_objects = compute_depth_region(floating_objects, depth_map_path)
                        floating_objects = [{k: json_serializable(v) for k, v in prediction.items()} for prediction in
                                            floating_objects]
                        json_contents[image_key]['floating_objects'] = floating_objects

        # Code to add assigned label
        label_file_path = os.path.join(args.labels_path, f"{base_name}.json")
        if os.path.exists(label_file_path):
            labels = json.load(open(label_file_path))
            for object in json_contents[image_key]['objects']:
                if str(object['id']) in labels.keys():
                    selected_label = labels[str(object['id'])]
                    if selected_label.lower() in person_dict.keys():
                        selected_label = person_dict[selected_label.lower()]
                    object['label'] = selected_label
                elif 'labels' in object.keys():
                    selected_label = object['labels'][0]
                    if selected_label.lower() in person_dict.keys():
                        selected_label = person_dict[selected_label.lower()]
                    object['label'] = selected_label
        else:
            for object in json_contents[image_key]['objects']:
                if 'labels' in object.keys():
                    selected_label = object['labels'][0]
                    if selected_label.lower() in person_dict.keys():
                        selected_label = person_dict[selected_label.lower()]
                    object['label'] = selected_label
        
        # Save using base name without assuming extension
        output_path = os.path.join(args.output_dir_path, f"{base_name}.json")
        with open(output_path, 'w') as f:
            json.dump(json_contents, f)