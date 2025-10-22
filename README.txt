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

---

## Environment Setup Instructions

### Prerequisites
- Anaconda or Miniconda installed
- CUDA-compatible GPU(s)
- Git and Git LFS installed

---

## Level 1 Inference Environments

### 1. Environment: `grand_env_2` (MiDaS Depth Maps)

**Purpose:** Depth estimation using MiDaS model.

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
