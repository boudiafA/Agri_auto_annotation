#!/bin/bash

# Configuration
DATASETS_ROOT="/mnt/e/Desktop/AgML/datasets_sorted/classification"
OUTPUTS_ROOT="/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/classification"
CKPT_DIR="/home/abood/groundingLMM/GranD/checkpoints"

# Clean up only your own stale PyTorch shm files
cleanup_shm() {
    echo "🧹 Cleaning /dev/shm torch leftovers for $USER..."
    pkill -f 'python.*DataLoader' 2>/dev/null || true
    find /dev/shm -maxdepth 1 -user "$USER" -name 'torch_*' -type f -print0 | xargs -0r rm -f
    sudo pkill -f python
}

# Function to handle errors
handle_errors() {
    echo "❌ Error in $1"
    # Add your error handling logic here
}

# Function to check if a directory contains image files
has_images() {
    local dir="$1"
    if ls "$dir"/*.{jpg,jpeg,png,JPG,JPEG,PNG} 2>/dev/null | grep -q .; then
        return 0
    else
        return 1
    fi
}

# Function to count files in a directory
count_files() {
    local dir="$1"
    local pattern="$2"
    
    if [ ! -d "$dir" ]; then
        echo "0"
        return
    fi
    
    if [ -z "$pattern" ]; then
        find "$dir" -maxdepth 1 -type f 2>/dev/null | wc -l
    else
        find "$dir" -maxdepth 1 -type f -name "$pattern" 2>/dev/null | wc -l
    fi
}

# Function to check if a step should be skipped
should_skip_step() {
    local step_name="$1"
    local input_dir="$2"
    local output_dir="$3"
    
    local input_count=$(count_files "$input_dir")
    local output_count=$(count_files "$output_dir")
    
    echo "[$step_name] Input files: $input_count, Output files: $output_count"
    
    if [ "$output_count" -ge "$input_count" ] && [ "$input_count" -gt 0 ]; then
        echo "[$step_name] ✓ Skipping - already processed ($output_count/$input_count files)"
        return 0
    else
        echo "[$step_name] → Processing ($output_count/$input_count files completed)"
        return 1
    fi
}

# CUDA version switching function - ADD YOUR CUDA SWITCHING LOGIC HERE
switch_cuda_version() {
    local env_name="$1"
    
    # Example CUDA switching logic based on environment name
    # Customize this based on your actual CUDA setup
    case "$env_name" in
        grand_env_1|grand_env_3|grand_env_7)
            export CUDA_HOME=/usr/local/cuda-11.8
            ;;
        grand_env_2|grand_env_4|grand_env_5|grand_env_6|grand_env_8|grand_env_9|grand_env_utils)
            export CUDA_HOME=/usr/local/cuda-12.1
            ;;
        *)
            # Default CUDA version
            export CUDA_HOME=/usr/local/cuda
            ;;
    esac
    
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    
    echo "🔧 CUDA_HOME set to: $CUDA_HOME"
}

# Scoped environment runner
run_in_scoped_env() {
    local env_name="$1"
    local work_dir="$2"
    shift 2

    # Switch CUDA version in the main shell context
    switch_cuda_version "$env_name"

    ( # Start subshell to localize conda env and directory changes
        echo "--- Activating conda env: $env_name in subshell ---"
        
        # Hardcoded conda paths to try (UPDATE THESE TO YOUR ACTUAL CONDA PATH!)
        CONDA_PATHS=(
            "/home/abood/miniconda3"
            "/home/abood/anaconda3"
            "$HOME/miniconda3"
            "$HOME/anaconda3"
            "/opt/conda"
            "/opt/miniconda3"
        )
        
        CONDA_BASE=""
        for path in "${CONDA_PATHS[@]}"; do
            if [ -f "$path/etc/profile.d/conda.sh" ]; then
                CONDA_BASE="$path"
                break
            fi
        done
        
        if [ -z "$CONDA_BASE" ]; then
            echo "✗ Error: Could not find conda installation"
            echo "  Tried paths:"
            for path in "${CONDA_PATHS[@]}"; do
                echo "    - $path"
            done
            echo ""
            echo "  Please update CONDA_PATHS in the script with your actual conda path"
            echo "  Run 'which conda' as your regular user to find the path"
            exit 1
        fi
        
        echo "Using conda from: $CONDA_BASE"
        source "$CONDA_BASE/etc/profile.d/conda.sh"
        if ! conda activate "$env_name"; then
            echo "✗ Failed to activate conda environment: $env_name"
            exit 1
        fi
        echo "✅ Conda env $env_name activated."

        echo "--- Changing to directory: $(realpath "$work_dir") (from $(pwd)) ---"
        if ! cd "$work_dir"; then
            echo "✗ Failed to change directory to $work_dir"
            exit 1
        fi
        
        echo "--- Executing command in $(pwd): $@ ---"
        "$@"
        cmd_exit_code=$?

        exit $cmd_exit_code
    )
    
    subshell_exit_code=$?
    if [ $subshell_exit_code -ne 0 ]; then
         echo "--- Command failed with exit code $subshell_exit_code ---"
         return $subshell_exit_code
    fi
    echo "--- Command completed successfully ---"
    return 0
}

# Function to process a  image directory
process_image_directory() {
    local IMG_DIR="$1"
    local base_output_dir="$2"  # e.g., OUTPUTS_ROOT/dataset_A or OUTPUTS_ROOT/dataset_B/cat
    local dataset_name="$3"
    
    echo "Processing: $IMG_DIR"
    echo "Base output: $base_output_dir"
    
    # Set PRED_DIR for this specific dataset
    local PRED_DIR="$base_output_dir"
    
    # Set SAM_ANNOTATIONS_DIR relative to PRED_DIR
    local SAM_ANNOTATIONS_DIR="$PRED_DIR/sam"
    
    # ============================================================================
    # 1. Captions and Tags
    # ============================================================================
    cleanup_shm
    echo "Running Gemma3..."

    run_in_scoped_env fastvlm level_1_inference/gemma3 \
        python infer_gemma3.py \
        --image_dir_path "$IMG_DIR" \
        --output_dir_path "$PRED_DIR" \
        --descriptions_json "../../detailed_class_caption/classification/${dataset_name}.json" \
        --scan-workers 1 || handle_errors "Gemma3-$dataset_name"

    
    # ============================================================================
    # 2. Depth Maps
    # ============================================================================
    cleanup_shm
    echo "Running Depth Estimation..."
    mkdir -p "$PRED_DIR/midas"
    if ! should_skip_step "Depth Maps" "$IMG_DIR" "$PRED_DIR/midas"; then
        run_in_scoped_env grand_env_2 level_1_inference/2_depth_maps \
            python infer.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --model_weights "$CKPT_DIR/dpt_beit_large_512.pt" \
            --local_rank 0 || handle_errors "Depth Maps-$dataset_name"
    fi
    
    # ============================================================================
    # 4. Object Detection using Co-DETR
    # ============================================================================
    cleanup_shm
    echo "Running Co-DETR..."
    mkdir -p "$PRED_DIR/co_detr"
    if ! should_skip_step "Co-DETR" "$IMG_DIR" "$PRED_DIR/co_detr"; then
        run_in_scoped_env grand_env_1 level_1_inference/4_co_detr \
            python launch_codetr_multi_gpu_inference.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --ckpt_path "$CKPT_DIR/co_deformable_detr_FT_Crops.pth" \
            --gpu_ids "0" || handle_errors "Co-DETR-$dataset_name"
    fi


    # ============================================================================
    # 4.5. Object Segmentation using SAM
    # ============================================================================
    echo "Running SAM..."
    mkdir -p "$PRED_DIR/sam"
    if ! should_skip_step "SAM" "$IMG_DIR" "$PRED_DIR/sam"; then
        run_in_scoped_env sam level_1_inference/9_ov_sam \
            python sam_infer.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --checkpoint "$CKPT_DIR/sam_vit_h_4b8939.pth"|| handle_errors "sam-$dataset_name"
    fi
    
    # ============================================================================
    # 5. Object Detection using EVA-02
    # ============================================================================
    cleanup_shm
    echo "Running EVA-02-01..."
    mkdir -p "$PRED_DIR/eva-02-01"
    if ! should_skip_step "EVA-02-01" "$IMG_DIR" "$PRED_DIR/eva-02-01"; then
        run_in_scoped_env grand_env_4 level_1_inference/5_eva_02 \
            python infer.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --model_name 'eva-02-01' \
            --checkpoint "$CKPT_DIR/eva_1_FT.pth" \
            --local_rank 0 || handle_errors "EVA-02-01-$dataset_name"
    fi
    
    cleanup_shm
    echo "Running EVA-02-02..."
    mkdir -p "$PRED_DIR/eva-02-02"
    if ! should_skip_step "EVA-02-02" "$IMG_DIR" "$PRED_DIR/eva-02-02"; then
        run_in_scoped_env grand_env_4 level_1_inference/5_eva_02 \
            python infer.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --model_name 'eva-02-02' \
            --checkpoint "$CKPT_DIR/eva_2_FT.pth" \
            --local_rank 0 || handle_errors "EVA-02-02-$dataset_name"
    fi
    
    # ============================================================================
    # 6. Open Vocabulary Detection using OWL-ViT
    # ============================================================================
    cleanup_shm
    echo "Running OWL-ViT..."
    mkdir -p "$PRED_DIR/owl_vit"
    if ! should_skip_step "OWL-ViT" "$IMG_DIR" "$PRED_DIR/owl_vit"; then
        run_in_scoped_env grand_env_1 level_1_inference/6_owl_vit \
            python launch_owl_vit_multi_gpu_inference.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --tags_dir_path "$PRED_DIR" \
            --gpu_ids "0" || handle_errors "OWL-ViT-$dataset_name"
    fi
    
    # ============================================================================
    # 7. Open Vocabulary Detection using POMP
    # ============================================================================
    cleanup_shm
    echo "Running POMP..."
    mkdir -p "$PRED_DIR/pomp"
    if ! should_skip_step "POMP" "$IMG_DIR" "$PRED_DIR/pomp"; then
        export CUDA_HOME=/usr/local/cuda-11.8
        export PATH=$CUDA_HOME/bin:$PATH
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        run_in_scoped_env grand_env_4 level_1_inference/7_pomp \
            python launch_pomp_multi_gpu_inference.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --tags_dir_path "$PRED_DIR" \
            --checkpoint "$CKPT_DIR/Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth" \
            --gpu_ids "0" || handle_errors "POMP-$dataset_name"
    fi
    
    # ============================================================================
    # 8. Attribute Detection and Grounding using GRIT
    # ============================================================================
    cleanup_shm
    echo "Running GRIT..."
    mkdir -p "$PRED_DIR/grit"
    if ! should_skip_step "GRIT" "$IMG_DIR" "$PRED_DIR/grit"; then
        run_in_scoped_env grand_env_3 level_1_inference/8_grit \
            python infer.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --checkpoint "$CKPT_DIR/grit_b_densecap_objectdet.pth" \
            --local_rank "0" || handle_errors "GRIT-$dataset_name"
    fi
    
    # ============================================================================
    # 9. Open Vocabulary Classification using OV-SAM
    # ============================================================================
    if [ -n "$SAM_ANNOTATIONS_DIR" ]; then
        cleanup_shm
        echo "Running OV-SAM..."
        mkdir -p "$PRED_DIR/ov_sam"
        if ! should_skip_step "OV-SAM" "$IMG_DIR" "$PRED_DIR/ov_sam"; then
            run_in_scoped_env grand_env_5 level_1_inference/9_ov_sam \
                python launch_ov_sam_multi_gpu_inference.py \
                --image_dir_path "$IMG_DIR" \
                --output_dir_path "$PRED_DIR" \
                --sam_annotations_dir "$SAM_ANNOTATIONS_DIR" \
                --checkpoint "$CKPT_DIR" \
                --gpu_ids "0" || handle_errors "OV-SAM-$dataset_name"
        fi
    else
        echo "Skipping OV-SAM step as SAM_ANNOTATIONS_DIR is not provided"
    fi
    
    # ============================================================================
    # 10. Generate Level-1 Scene Graph
    # ============================================================================
    cleanup_shm
    echo "Generating Level-1 Scene Graph..."
    mkdir -p "$PRED_DIR/level-1-raw"
    if ! should_skip_step "Level-1 Scene Graph (merge)" "$IMG_DIR" "$PRED_DIR/level-1-raw"; then
        run_in_scoped_env grand_env_utils utils \
            python merge_json_level_1_with_nms.py \
            --image_dir_path "$IMG_DIR" \
            --predictions_dir_path "$PRED_DIR" \
            --output_dir_path "$PRED_DIR/level-1-raw" || handle_errors "Level-1 Scene Graph (merge)-$dataset_name"
    fi
    
    mkdir -p "$PRED_DIR/level-1-processed"
    if ! should_skip_step "Level-1 Scene Graph (prepare)" "$PRED_DIR/level-1-raw" "$PRED_DIR/level-1-processed"; then
        run_in_scoped_env grand_env_utils utils \
            python prepare_level_1.py \
            --image_dir_path "$IMG_DIR" \
            --raw_dir_path "$PRED_DIR/level-1-raw" \
            --output_dir_path "$PRED_DIR/level-1-processed" \
            --depth_map_dir "$PRED_DIR/midas" || handle_errors "Level-1 Scene Graph (prepare)-$dataset_name"
    fi
    
    # ============================================================================
    # 13. Grounding using MDETR
    # ============================================================================
    cleanup_shm
    echo "Running MDETR Grounding..."
    mkdir -p "$PRED_DIR/mdetr-re"
    if ! should_skip_step "MDETR" "$IMG_DIR" "$PRED_DIR/mdetr-re"; then
        run_in_scoped_env grand_env_7 level_2_inference/3_mdetr \
            python infer.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR" \
            --blip2_pred_path "$PRED_DIR/blip2" \
            --llava_pred_path "$PRED_DIR/llava" \
            --checkpoint "$CKPT_DIR/refcocog_EB3_checkpoint.pth" \
            --local_rank 0 || handle_errors "MDETR-$dataset_name"
    fi
    
    # ============================================================================
    # 14. Generate Level-2 Scene Graph and Update Level-1
    # ============================================================================
    cleanup_shm
    echo "Generating Level-2 Scene Graph..."
    mkdir -p "$PRED_DIR/level-2-raw"
    if ! should_skip_step "Level-2 Scene Graph (merge)" "$IMG_DIR" "$PRED_DIR/level-2-raw"; then
        run_in_scoped_env grand_env_utils utils \
            python merge_json_level_2.py \
            --predictions_dir_path "$PRED_DIR" \
            --output_dir_path "$PRED_DIR/level-2-raw" || handle_errors "Level-2 Scene Graph (merge)-$dataset_name"
    fi
    
    mkdir -p "$PRED_DIR/level-2-processed"
    if ! should_skip_step "Level-2 Scene Graph (prepare)" "$PRED_DIR/level-2-raw" "$PRED_DIR/level-2-processed"; then
        run_in_scoped_env grand_env_utils utils \
            python prepare_level_2.py \
            --raw_dir_path "$PRED_DIR/level-2-raw" \
            --level_2_output_dir_path "$PRED_DIR/level-2-processed" \
            --level_1_dir_path "$PRED_DIR/level-1-processed" || handle_errors "Level-2 Scene Graph (prepare)-$dataset_name"
    fi
    
    # ============================================================================
    # 15. Enrich Attributes using GPT4RoI
    # ============================================================================
    cleanup_shm
    echo "Running GPT4RoI for Attribute Enrichment..."
    mkdir -p "$PRED_DIR/level-2-processed_gpt4roi"
    if ! should_skip_step "GPT4RoI" "$PRED_DIR/level-2-processed" "$PRED_DIR/level-2-processed_gpt4roi"; then
        run_in_scoped_env grand_env_8 level_2_inference/4_gpt4roi/GPT4RoI \
            python gpt4roi/infer2.py \
            --image_dir_path "$IMG_DIR" \
            --level_2_pred_path "$PRED_DIR/level-2-processed" \
            --output_dir_path "$PRED_DIR/level-2-processed_gpt4roi" \
            --model_name "$CKPT_DIR/GPT4RoI-7B-delta" \
            --batch_size_per_gpu 1 \
            --object_batch_size 2 \
            --local_rank "0" || handle_errors "GPT4RoI-$dataset_name"
    fi
    
    # ============================================================================
    # 16. Label Assignment using EVA-CLIP
    # ============================================================================
    cleanup_shm
    echo "Running EVA-CLIP for Label Assignment..."
    mkdir -p "$PRED_DIR/level-2-processed_eva_clip"
    if ! should_skip_step "EVA-CLIP" "$IMG_DIR" "$PRED_DIR/level-2-processed_eva_clip"; then
        run_in_scoped_env grand_env_4 level_2_inference/5_label_assignment \
            python infer.py \
            --image_dir_path "$IMG_DIR" \
            --level_2_dir_path "$PRED_DIR/level-2-processed_gpt4roi" \
            --output_dir_path "$PRED_DIR/level-2-processed_eva_clip" || handle_errors "EVA-CLIP-$dataset_name"
    fi
    
    # ============================================================================
    # 17. Merge EVA-CLIP Assigned Labels & Calculate and Store Depths for All Objects
    # ============================================================================
    cleanup_shm
    echo "Merging EVA-CLIP Labels and Calculating Depths..."
    mkdir -p "$PRED_DIR/level-2-processed_labelled"
    if ! should_skip_step "Merge EVA Labels" "$PRED_DIR/level-2-processed_gpt4roi" "$PRED_DIR/level-2-processed_labelled"; then
        run_in_scoped_env grand_env_utils . \
            python -m utils.merge_eva_labels \
            --level_2_dir_path "$PRED_DIR/level-2-processed_gpt4roi" \
            --labels_path "$PRED_DIR/level-2-processed_eva_clip" \
            --output_dir_path "$PRED_DIR/level-2-processed_labelled" \
            --store_depth \
            --depth_map_dir "$PRED_DIR/midas" || handle_errors "Merge EVA Labels-$dataset_name"
    fi
    
    # ============================================================================
    # 18. Generate Level-3 Dense Captions
    # ============================================================================
    echo "Generating Level-3 Dense Captions..."
    mkdir -p "$PRED_DIR/level-3-vicuna-13B"
    if ! should_skip_step "Level-3 Dense Captions" "$IMG_DIR" "$PRED_DIR/level-3-vicuna-13B"; then
        run_in_scoped_env llama31 level_3_dense_caption \
            python run_llama31.py \
            --image_dir_path "$IMG_DIR" \
            --level_2_dir_path "$PRED_DIR/level-2-processed_labelled" \
            --output_dir_path "$PRED_DIR/level-3-vicuna-13B" \
            --model_path "$CKPT_DIR/llama-3.1-8b" \
            --gpu_ids "0" \
            --job_id '111' \
            --quantization "8bit" \
            --batch_size 8 || handle_errors "Level-3 Dense Captions (local model)-$dataset_name"

    fi
    
    # # Generate image names text file for Level-4 Additional Context
    # echo "Generating image names text file..."
    # mkdir -p "$PRED_DIR"
    # ls -1 "$IMG_DIR" > "$PRED_DIR/image_names.txt"
    
    # ============================================================================
    # 19. Generate Level-4 Additional Context 
    # ============================================================================
    echo "Generating Level-4 Additional Context..."
    mkdir -p "$PRED_DIR/level-4-vicuna-13B"
    if ! should_skip_step "Level-4 Additional Context" "$IMG_DIR" "$PRED_DIR/level-4-vicuna-13B"; then
        run_in_scoped_env grand_env_9 level_4_extra_context \
            python generate_captions_from_json.py \
            --image_dir_path "$IMG_DIR" \
            --output_dir_path "$PRED_DIR/level-4-vicuna-13B" \
            --descriptions_json "../../detailed_class_caption/classification/${dataset_name}.json" \
            --job_id '111' || handle_errors "Level-4 Additional Context (local model)-$dataset_name"

    fi
    
    # ============================================================================
    # 20. Ground short & dense captions
    # ============================================================================
    echo "Grounding Short Captions..."
    mkdir -p "$PRED_DIR/short_captions_grounded"
    if ! should_skip_step "Ground Short Captions" "$PRED_DIR/level-2-processed_labelled" "$PRED_DIR/short_captions_grounded"; then
        run_in_scoped_env grand_env_utils utils \
            python ground_short_captions.py \
            --data_dir_path "$PRED_DIR/level-2-processed_labelled" \
            --output_dir_path "$PRED_DIR/short_captions_grounded" || handle_errors "Ground Short Captions-$dataset_name"
    fi
    
    echo "Grounding Dense Captions..."
    mkdir -p "$PRED_DIR/dense_captions_grounded"
    if ! should_skip_step "Ground Dense Captions" "$PRED_DIR/level-3-vicuna-13B" "$PRED_DIR/dense_captions_grounded"; then
        run_in_scoped_env grand_env_utils utils \
            python ground_dense_caption.py \
            --level_3_dense_caption_txt_dir_path "$PRED_DIR/level-3-vicuna-13B" \
            --level_2_processed_json_path "$PRED_DIR/short_captions_grounded" \
            --output_dir_path "$PRED_DIR/dense_captions_grounded" || handle_errors "Ground Dense Captions-$dataset_name"
    fi
    
    # ============================================================================
    # 21. Add Masks to the Annotations (sources: SAM Annotations & EVA Detector)
    # ============================================================================
    echo "Adding Masks to Annotations..."
    mkdir -p "$PRED_DIR/level-3-processed"
    if ! should_skip_step "Add Masks" "$PRED_DIR/dense_captions_grounded" "$PRED_DIR/level-3-processed"; then
        if [ -n "$SAM_ANNOTATIONS_DIR" ]; then
            run_in_scoped_env grand_env_utils utils \
                python add_masks_to_annotations.py \
                --input_dir_path "$PRED_DIR/dense_captions_grounded" \
                --sam_json_dir_path "$SAM_ANNOTATIONS_DIR" \
                --eva_02_pred_dir_path "$PRED_DIR/eva-02-01" \
                --output_dir_path "$PRED_DIR/level-3-processed" || handle_errors "Add Masks (with SAM)-$dataset_name"
        else
            echo "SAM_ANNOTATIONS_DIR not provided. Using only EVA detector masks."
            run_in_scoped_env grand_env_utils utils \
                python add_masks_to_annotations.py \
                --input_dir_path "$PRED_DIR/dense_captions_grounded" \
                --sam_json_dir_path "" \
                --eva_02_pred_dir_path "$PRED_DIR/eva-02-01" \
                --output_dir_path "$PRED_DIR/level-3-processed" || handle_errors "Add Masks (without SAM)-$dataset_name"
        fi
    fi
    
    # ============================================================================
    # 22. Use HQ-SAM for the Rest of the Masks not Found in SAM Annotations or EVA Detections
    # ============================================================================
    echo "Running HQ-SAM for Remaining Masks..."
    mkdir -p "$PRED_DIR/level-3-processed_with_masks"
    HQ_SAM_CKPT_PATH="$CKPT_DIR/sam_hq_vit_h.pth"
    if [ -f "$HQ_SAM_CKPT_PATH" ]; then
        run_in_scoped_env grand_env_1 utils/hq_sam \
            python run.py \
            --image_dir_path "$IMG_DIR" \
            --level_3_processed_path "$PRED_DIR/level-3-processed" \
            --output_dir_path "$PRED_DIR/level-3-processed_with_masks" \
            --checkpoints_path "$HQ_SAM_CKPT_PATH" || handle_errors "HQ-SAM-$dataset_name"
    else
        echo "Warning: HQ-SAM checkpoint sam_hq_vit_h.pth not found at $HQ_SAM_CKPT_PATH. Skipping HQ-SAM step."
        echo "Copying level-3-processed to level-3-processed_with_masks as a fallback."
        if [ -d "$PRED_DIR/level-3-processed" ] && [ -n "$(ls -A "$PRED_DIR/level-3-processed")" ]; then
            cp -r "$PRED_DIR/level-3-processed/"* "$PRED_DIR/level-3-processed_with_masks/" || handle_errors "Copy level-3-processed (fallback)-$dataset_name"
        else
            echo "Warning: Source directory $PRED_DIR/level-3-processed is empty or does not exist. Nothing to copy for fallback."
            mkdir -p "$PRED_DIR/level-3-processed_with_masks"
        fi
    fi
    
    # ============================================================================
    # 23. Add Additional Context to the Annotations
    # ============================================================================
    echo "Adding Additional Context to Annotations..."
    mkdir -p "$PRED_DIR/level-4-processed"
    run_in_scoped_env grand_env_utils utils \
        python add_additional_context.py \
        --annotations_dir_path "$PRED_DIR/level-3-processed_with_masks" \
        --level_4_additional_context_path "$PRED_DIR/level-4-vicuna-13B" \
        --output_dir_path "$PRED_DIR/level-4-processed" || handle_errors "Add Additional Context-$dataset_name"
    
    echo "=== Completed processing for $dataset_name ==="
}
# Main processing loop
for dataset_dir in "$DATASETS_ROOT"/*; do
    [ ! -d "$dataset_dir" ] && continue
    
    dataset_name=$(basename "$dataset_dir")
    echo "=== Processing dataset: $dataset_name ==="
    
    if has_images "$dataset_dir"; then
        echo "Found images directly in dataset folder: $dataset_name (flat structure)"
        img_dir="$dataset_dir"
        base_output_dir="$OUTPUTS_ROOT/$dataset_name"
        process_image_directory "$img_dir" "$base_output_dir" "$dataset_name"
    else
        echo "Processing classification dataset with class subfolders: $dataset_name"
        
        class_count=0
        for class_dir in "$dataset_dir"/*; do
            [ ! -d "$class_dir" ] && continue
            
            class_name=$(basename "$class_dir")
            
            if has_images "$class_dir"; then
                echo "  → Processing class: $class_name"
                img_dir="$class_dir"
                base_output_dir="$OUTPUTS_ROOT/$dataset_name/$class_name"
                full_path_name="$dataset_name/$class_name"
                
                process_image_directory "$img_dir" "$base_output_dir" "$full_path_name"
                ((class_count++))
            else
                echo "  → Warning: No images found in class folder: $class_name"
            fi
        done
        
        if [ $class_count -eq 0 ]; then
            echo "  → Warning: No class subfolders with images found in dataset: $dataset_name"
        else
            echo "  → Processed $class_count classes in dataset: $dataset_name"
        fi
    fi
    
    echo ""
done

trap 'cleanup_shm' EXIT HUP INT TERM

echo "=== All datasets processed ==="