# GranD Pipeline - Conda Environment Setup Guide

This guide provides detailed instructions for creating conda environments for each model in the GranD automated annotation pipeline.

## Overview

The GranD pipeline requires 10 different conda environments due to varying dependencies across models. These environments are:
- `grand_env_1` - for Landmark Detection, Co-DETR, OWL-ViT
- `grand_env_2` - for MiDaS Depth Estimation
- `grand_env_3` - for Image Tagging (Tag2Text & RAM), GRIT, BLIP-2
- `grand_env_4` - for EVA-02, POMP, EVA-CLIP
- `grand_env_5` - for OV-SAM
- `grand_env_6` - for LLaVA (Level 2)
- `grand_env_7` - for MDETR
- `grand_env_8` - for GPT4RoI
- `grand_env_9` - for Level 3 Dense Captions
- `grand_env_utils` - for utility scripts and scene graph generation
- `fastvlm` - for Llama 3.1
- `sam` - for SAM mdoel

---

## Environment Setup Instructions

### Prerequisites
- Anaconda or Miniconda installed
- CUDA-compatible GPU(s)
- Git and Git LFS installed

---

## Level 1 Inference Environments

### 1. Environment: `grand_env_2` (MiDaS Depth Maps)

**Purpose:** Depth estimation using MiDaS model

**Step 1: Create environment file**
Create a file named `grand_env_2_midas.yml`:

```yaml
name: grand_env_2
channels:
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.10.8
  - pytorch::pytorch=1.13.0
  - torchvision=0.14.0
  - nvidia::cudatoolkit=11.7
  - pip=22.3.1
  - numpy=1.23.4
  - pip:
    - opencv-python==4.6.0.66
    - imutils==0.5.4
    - timm==0.6.12
    - einops==0.6.0
```

**Step 2: Create the environment**
```bash
conda env create -f grand_env_2_midas.yml
```

**Step 3: Activate and verify**
```bash
conda activate grand_env_2
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Step 4: Download model checkpoint**
```bash
mkdir -p checkpoints
cd checkpoints
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
cd ..
```

**Usage:**
```bash
conda activate grand_env_2
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --model_weights /path/to/dpt_beit_large_512.pt
```

---

### 2. Environment: `grand_env_1` (Landmark Detection, Co-DETR, OWL-ViT)

**Purpose:** Landmark categorization using LLaVA, object detection using Co-DETR, open vocabulary detection using OWL-ViT

**Step 1: Create environment from provided file**
```bash
# Use the provided grand_env_1.yml file from the environments directory
conda env create -f environments/grand_env_1.yml
```

**Note:** The grand_env_1.yml file contains extensive dependencies. If some pip packages fail to install, you may need to install them separately.

**Step 2: Activate environment**
```bash
conda activate grand_env_1
```

**Step 3: Download model checkpoints**

For Landmark Detection (LLaVA):
```bash
cd checkpoints
git lfs install
git clone https://huggingface.co/liuhaotian/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
cd ..
```

For Co-DETR:
```bash
# Download from Google Drive link:
# https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing
# Place co_deformable_detr_swin_large_900q_3x_coco.pth in checkpoints directory
```

**Usage for Landmark Detection:**
```bash
conda activate grand_env_1
python infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --gpu_ids "0" \
    --llava_model_path /path/to/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
```

**Usage for Co-DETR:**
```bash
conda activate grand_env_1
python launch_codetr_multi_gpu_inference.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --ckpt_path /path/to/co_deformable_detr_swin_large_900q_3x_coco.pth \
    --gpu_ids "0,1"
```

---

### 3. Environment: `grand_env_3` (Image Tagging, GRIT, BLIP-2)

**Purpose:** Image tagging using Tag2Text & RAM, attribute detection using GRIT, captioning using BLIP-2

**Step 1: Create environment from provided file**
```bash
conda env create -f environments/grand_env_3.yml
```

**Step 2: Activate environment**
```bash
conda activate grand_env_3
```

**Step 3: Download model checkpoints**

For Image Tagging:
```bash
cd checkpoints
# Download tag2text_swin_14m.pth from:
# https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/tag2text_swin_14m.pth

# Download ram_swin_large_14m.pth from:
# https://huggingface.co/spaces/xinyu1205/recognize-anything/blob/main/ram_swin_large_14m.pth
cd ..
```

For GRIT:
```bash
cd checkpoints
wget -c https://datarelease.blob.core.windows.net/grit/models/grit_b_densecap_objectdet.pth
cd ..
```

**Usage for Image Tagging (Tag2Text):**
```bash
conda activate grand_env_3
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --model-type tag2text \
    --checkpoint /path/to/tag2text_swin_14m.pth
```

**Usage for Image Tagging (RAM):**
```bash
conda activate grand_env_3
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --model-type ram \
    --checkpoint /path/to/ram_swin_large_14m.pth
```

**Usage for GRIT:**
```bash
conda activate grand_env_3
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output
```

---

### 4. Environment: `grand_env_4` (EVA-02, POMP, EVA-CLIP)

**Purpose:** Object detection using EVA-02, open vocabulary detection using POMP, label assignment using EVA-CLIP

**Step 1: Create environment from provided file**
```bash
conda env create -f environments/grand_env_4.yml
```

**Step 2: Activate and setup EVA-02**
```bash
conda activate grand_env_4
# Follow instructions at: https://github.com/baaivision/EVA/tree/master/EVA-02/det#setup
```

**Step 3: Download model checkpoints**

For EVA-02:
```bash
cd checkpoints
# Download eva02_L_lvis_sys.pth from:
# https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys.pth

# Download eva02_L_lvis_sys_o365.pth from:
# https://huggingface.co/Yuxin-CV/EVA-02/blob/main/eva02/det/eva02_L_lvis_sys_o365.pth
cd ..
```

For POMP:
```bash
cd checkpoints
# Download from Google Drive:
# https://drive.google.com/file/d/1C8oU6cWkJdU3Q3IHaqTcbIToRLo9bMnu/view?usp=sharing
# https://drive.google.com/file/d/1TwrjcUYimkI_f9z9UZXCmLztdgv31Peu/view?usp=sharing
# Place Detic_LI_CLIP_R5021k_640b64_4x_ft4x_max-size_pomp.pth in checkpoints directory
cd ..
```

**Usage for EVA-02:**
```bash
conda activate grand_env_4
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --model_name 'eva-02-01'  # or 'eva-02-02'
```

**Usage for POMP:**
```bash
conda activate grand_env_4
python launch_pomp_multi_gpu_inference.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --tags_dir_path /path/to/tags \
    --gpu_ids "0,1"
```

---

### 5. Environment: `grand_env_5` (OV-SAM)

**Purpose:** Open vocabulary classification using OV-SAM

**Step 1: Setup environment**
```bash
# Follow installation instructions at:
# https://github.com/HarborYuan/ovsam?tab=readme-ov-file#%EF%B8%8F-installation

conda create -n grand_env_5 python=3.10
conda activate grand_env_5
# Install required dependencies as per OV-SAM documentation
```

**Step 2: Download model checkpoint**
```bash
cd checkpoints
# Download sam2clip_vith_rn50x16.pth from:
# https://huggingface.co/HarborYuan/ovsam_models/blob/main/sam2clip_vith_rn50x16.pth
cd ..
```

**Usage:**
```bash
conda activate grand_env_5
python launch_ov_sam_multi_gpu_inference.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --sam_annotations_dir /path/to/sam_annotations \
    --gpu_ids "0,1"
```

---

## Level 2 Inference Environments

### 6. Environment: `grand_env_3` (BLIP-2 Captioning) - Already Created

**Purpose:** Image captioning using BLIP-2

**Setup:** Environment already created in Level 1 (step 3)

**Usage:**
```bash
conda activate grand_env_3
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output
```

**Note:** Checkpoints are automatically downloaded by the model.

---

### 7. Environment: `grand_env_6` (LLaVA Captioning)

**Purpose:** Image captioning using LLaVA

**Step 1: Create environment from provided file**
```bash
conda env create -f environments/grand_env_6.yml
```

**Step 2: Activate environment**
```bash
conda activate grand_env_6
```

**Step 3: Use LLaVA checkpoint** (already downloaded in step 2 of grand_env_1)

**Usage:**
```bash
conda activate grand_env_6
python infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --gpu_ids "0,1" \
    --llava_model_path /path/to/llava-v1-0719-336px-lora-merge-vicuna-13b-v1.3
```

---

### 8. Environment: `grand_env_7` (MDETR)

**Purpose:** Phrase grounding using MDETR

**Step 1: Create environment from provided file**
```bash
conda env create -f environments/grand_env_7.yml
```

**Step 2: Activate and setup MDETR**
```bash
conda activate grand_env_7
# Follow MDETR installation instructions
```

**Usage:**
```bash
conda activate grand_env_7
export NUM_GPU=1
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env infer.py \
    --image_dir_path /path/to/images \
    --output_dir_path /path/to/output \
    --blip2_pred_path /path/to/blip2_predictions \
    --llava_pred_path /path/to/llava_predictions
```

---

### 9. Environment: `grand_env_8` (GPT4RoI)

**Purpose:** Attribute enrichment using GPT4RoI

**Step 1: Setup GPT4RoI environment**
```bash
cd level_2_inference/4_gpt4roi
git clone https://github.com/jshilong/GPT4RoI.git
cd GPT4RoI

# Follow installation instructions at:
# https://github.com/jshilong/GPT4RoI?tab=readme-ov-file#install
```

**Step 2: Download GPT4RoI weights**
```bash
# Follow instructions at:
# https://github.com/jshilong/GPT4RoI?tab=readme-ov-file#weights
```

**Step 3: Copy inference files**
```bash
cp ../ddp.py GPT4RoI/gpt4roi/
cp ../inference_utils.py GPT4RoI/gpt4roi/
cp ../infer.py GPT4RoI/gpt4roi/
```

**Usage:**
```bash
conda activate grand_env_8
export NUM_GPU=1
cd level_2_inference/4_gpt4roi/GPT4RoI
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPU \
    --master_port=1211 \
    --use_env gpt4roi/infer.py \
    --image_dir_path /path/to/images \
    --level_2_pred_path /path/to/level_2_predictions \
    --output_dir_path /path/to/output
```

---

## Level 3 & Utilities Environments

### 10. Environment: `grand_env_9` (Dense Captions)

**Purpose:** Generate level-3 dense captions

**Step 1: Create environment from provided file**
```bash
conda env create -f environments/grand_env_9.yml
```

**Step 2: Activate environment**
```bash
conda activate grand_env_9
```

**Usage:**
```bash
conda activate grand_env_9
python run.py \
    --image_dir_path /path/to/images \
    --level_2_dir_path /path/to/level_2_predictions \
    --output_dir_path /path/to/output \
    --gpu_ids "0,1" \
    --job_id '111'
```

---

### 11. Environment: `grand_env_utils` (Utility Scripts)

**Purpose:** Scene graph generation and processing utilities

**Step 1: Create environment file**
Create a file named `grand_env_utils.yml`:

```yaml
name: grand_env_utils
channels:
  - defaults
dependencies:
  - python=3.11.8
  - pip=23.3.1
  - pip:
    - torch==2.2.1
    - numpy==1.26.4
    - opencv-python==4.9.0.80
    - pillow==10.2.0
    - scipy==1.12.0
    - scikit-learn==1.4.1.post1
    - transformers==4.38.2
    - spacy==3.7.4
    - en-core-web-sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
    - nltk==3.8.1
    - lmdb==1.4.1
    - mmcv-full==1.7.2
    - pycocotools==2.0.7
    - pyyaml==6.0.1
    - tqdm==4.66.2
```

**Step 2: Create the environment**
```bash
conda env create -f grand_env_utils.yml
```

**Step 3: Activate environment**
```bash
conda activate grand_env_utils
```

**Usage examples:**

Merge Level 1 predictions:
```bash
conda activate grand_env_utils
python utils/merge_json_level_1_with_nms.py \
    --image_dir_path /path/to/images \
    --predictions_dir_path /path/to/predictions \
    --output_dir_path /path/to/output/level-1-raw
```

Prepare Level 1 scene graph:
```bash
conda activate grand_env_utils
python utils/prepare_level_1.py \
    --image_dir_path /path/to/images \
    --raw_dir_path /path/to/level-1-raw \
    --output_dir_path /path/to/level-1-processed
```

---

## Quick Setup: Create All Environments at Once

You can create all environments sequentially using these commands:

```bash
# Navigate to the environments directory
cd environments

# Create all environments
conda env create -f grand_env_1.yml
conda env create -f grand_env_2.yml
conda env create -f grand_env_3.yml
conda env create -f grand_env_4.yml
# For grand_env_5, follow OV-SAM installation guide
conda env create -f grand_env_6.yml
conda env create -f grand_env_7.yml
# For grand_env_8, follow GPT4RoI installation guide
conda env create -f grand_env_9.yml
conda env create -f grand_env_utils.yml
```

---

## Troubleshooting

### Common Issues

1. **pip dependencies fail during conda environment creation:**
   - Remove the failing packages from the .yml file
   - Create the environment without them
   - Activate the environment and install them manually using pip

2. **CUDA compatibility issues:**
   - Ensure your CUDA version matches the cudatoolkit specified in the environment
   - Update CUDA drivers if necessary
   - Modify cudatoolkit version in .yml files to match your system

3. **Memory issues during model inference:**
   - Reduce batch size
   - Use fewer GPUs
   - Process images in smaller batches

4. **Missing model checkpoints:**
   - Verify all checkpoint downloads are complete
   - Check file permissions
   - Ensure sufficient disk space

### Verification Commands

After creating each environment, verify it works:

```bash
conda activate <env_name>
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
conda deactivate
```

---

## Environment Summary Table

| Environment | Purpose | Models | Python | PyTorch |
|-------------|---------|--------|--------|---------|
| grand_env_1 | Landmark, Co-DETR, OWL-ViT | LLaVA, Co-DETR, OWL-ViT | 3.10 | 1.13+ |
| grand_env_2 | Depth Estimation | MiDaS | 3.10.8 | 1.13.0 |
| grand_env_3 | Tagging, GRIT, BLIP-2 | Tag2Text, RAM, GRIT, BLIP-2 | 3.10 | varies |
| grand_env_4 | EVA-02, POMP, EVA-CLIP | EVA-02, POMP | 3.10 | varies |
| grand_env_5 | OV-SAM | OV-SAM | 3.10 | varies |
| grand_env_6 | LLaVA Caption | LLaVA | 3.10 | varies |
| grand_env_7 | MDETR Grounding | MDETR | 3.10 | varies |
| grand_env_8 | Attribute Enrichment | GPT4RoI | 3.10 | varies |
| grand_env_9 | Dense Captions | Custom | 3.10 | varies |
| grand_env_utils | Utilities | Various | 3.11.8 | 2.2.1 |

---

## Additional Notes

1. **Environment Isolation:** Each environment is isolated to prevent dependency conflicts
2. **GPU Requirements:** Most models require CUDA-capable GPUs
3. **Disk Space:** Ensure adequate disk space for all checkpoints (~50GB+)
4. **Checkpoint Management:** Keep all checkpoints in a centralized directory
5. **Multi-GPU Support:** Most models support distributed training/inference

---

## Next Steps

After setting up all environments:

1. Download all required model checkpoints
2. Verify each environment by running test inference
3. Configure the `run_pipeline.sh` script with your paths
4. Run the complete pipeline or individual components

For detailed usage of each model, refer to the README files in their respective directories.
