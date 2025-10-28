#!/bin/bash

BASE_IMAGE_DIR="/mnt/e/Desktop/AgML/datasets_sorted/detection"
BASE_OUTPUT_DIR="/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection"

# Count total number of folders
total_folders=$(find "$BASE_IMAGE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
current_folder=0

for dataset in "$BASE_IMAGE_DIR"/*; do
    if [ -d "$dataset" ]; then
        current_folder=$((current_folder+1))
        dataset_name=$(basename "$dataset")

        IMAGE_DIR="$dataset/images"
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$dataset_name"
        TAGS_DIR="$BASE_OUTPUT_DIR/$dataset_name"

        echo "[$current_folder/$total_folders] Running inference for dataset: $dataset_name"

        # Create output directory if it doesn't exist
        mkdir -p "$OUTPUT_DIR"

        python sam_infer_mini.py \
            --image_dir_path "$IMAGE_DIR" \
            --output_dir_path "$OUTPUT_DIR"
    fi
done
