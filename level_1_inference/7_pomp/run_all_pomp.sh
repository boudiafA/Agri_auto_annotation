#!/bin/bash

BASE_IMAGE_DIR="/mnt/e/Desktop/AgML/datasets_sorted/detection"
BASE_OUTPUT_DIR="/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection"

for dataset in "$BASE_IMAGE_DIR"/*; do
    if [ -d "$dataset" ]; then
        dataset_name=$(basename "$dataset")

        IMAGE_DIR="$dataset/images"
        OUTPUT_DIR="$BASE_OUTPUT_DIR/$dataset_name"
        TAGS_DIR="$BASE_OUTPUT_DIR/$dataset_name"

        echo "Running inference for dataset: $dataset_name"

        python launch_pomp_multi_gpu_inference.py \
            --image_dir_path "$IMAGE_DIR" \
            --output_dir_path "$OUTPUT_DIR" \
            --tags_dir_path "$TAGS_DIR" \
            --gpu_ids "0"
    fi
done
