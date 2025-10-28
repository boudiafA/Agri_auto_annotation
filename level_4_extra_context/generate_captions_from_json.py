import os
import argparse
import json
from pathlib import Path
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate caption text files for images based on dataset descriptions."
    )
    
    parser.add_argument(
        "--image_dir_path", 
        type=str, 
        required=True,
        help="Path to the directory containing images (last folder name is the dataset name)"
    )
    parser.add_argument(
        "--output_dir_path", 
        type=str, 
        required=True,
        help="Path to the output directory where .txt files will be saved"
    )
    parser.add_argument(
        "--job_id", 
        type=str, 
        required=True,
        help="Job identifier for logging purposes"
    )
    parser.add_argument(
        "--descriptions_json",
        type=str,
        default="./species_discription_cleaned.json",
        help="Path to the JSON file containing dataset descriptions (default: species_discription_cleaned.json)"
    )
    
    return parser.parse_args()


def get_dataset_name(image_dir_path):
    """
    Extract the dataset name from the image directory path.
    The dataset name is the last folder in the path.
    
    Args:
        image_dir_path: Path to the image directory
        
    Returns:
        Dataset name (last folder in the path)
    """
    path = Path(image_dir_path).resolve()
    dataset_name = path.name
    return dataset_name


def get_image_files(image_dir_path):
    """
    Get all image files from the directory.
    
    Args:
        image_dir_path: Path to the image directory
        
    Returns:
        List of image filenames
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    image_files = []
    
    for filename in os.listdir(image_dir_path):
        file_path = os.path.join(image_dir_path, filename)
        if os.path.isfile(file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)
    
    return sorted(image_files)


def main():
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir_path, exist_ok=True)
    
    # Extract dataset name from the image directory path
    dataset_name = get_dataset_name(args.image_dir_path)
    
    # Load the JSON file with dataset descriptions
    if not os.path.exists(args.descriptions_json):
        tqdm.write(f"ERROR: JSON file not found at {args.descriptions_json}")
        return
    
    with open(args.descriptions_json, 'r', encoding='utf-8') as f:
        dataset_descriptions = json.load(f)
    
    # Check if the dataset name exists in the JSON
    if dataset_name not in dataset_descriptions:
        tqdm.write(f"ERROR: Dataset '{dataset_name}' not found in JSON file")
        tqdm.write(f"Available datasets: {list(dataset_descriptions.keys())}")
        return
    
    detailed_caption = dataset_descriptions[dataset_name]
    
    # Get all image files from the directory
    image_files = get_image_files(args.image_dir_path)
    
    if not image_files:
        tqdm.write(f"WARNING: No image files found in {args.image_dir_path}")
        return
    
    # Generate text files for each image
    created_count = 0
    skipped_count = 0
    
    for image_filename in tqdm(image_files, desc=f"Processing {dataset_name}"):
        # Get the base name without extension
        base_name = os.path.splitext(image_filename)[0]
        
        # Create output text file path
        output_txt_path = os.path.join(args.output_dir_path, f"{base_name}.txt")
        
        # Skip if file already exists
        if os.path.exists(output_txt_path):
            skipped_count += 1
            continue
        
        # Write the detailed caption to the text file
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(detailed_caption)
        
        created_count += 1
    
    tqdm.write(f"Complete: Created {created_count}, Skipped {skipped_count}, Total {len(image_files)}")


if __name__ == "__main__":
    main()