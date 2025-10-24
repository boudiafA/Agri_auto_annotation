---

# GranD Project Setup and Environment Installation Guide

This guide provides comprehensive instructions for setting up the GranD project, including downloading the necessary data and code, installing the required conda environments, and running the processing pipeline.

## Step 1: Setup and Downloads

Before installing the environments, you need to download the main project code, the dataset, and the environment export files.

1.  **Main Code and Weights**
    The main folder containing all the code and model weights can be downloaded from:
    - **Link:** [https://drive.google.com/file/d/1aRJ00qMRYF6aqdePuAve_lO_3H0FtYBn/view?usp=sharing]

2.  **Dataset**
    Download the required dataset from the following link:
    - **Link:** [https://drive.google.com/file/d/1GQz7HQw6Mvlqw4GoRRxy-MwFxrDejp4d/view?usp=sharing](https://drive.google.com/file/d/1GQz7HQw6Mvlqw4GoRRxy-MwFxrDejp4d/view?usp=sharing)

3.  **Conda Environment Exports**
    The conda environment files are bundled in `env_exports.rar`. Download it from:
    - **Link:** [https://drive.google.com/file/d/1RyVw6k3DdFddwh9_rBvU3NoJsKkVaHGl/view?usp=sharing](https://drive.google.com/file/d/1RyVw6k3DdFddwh9_rBvU3NoJsKkVaHGl/view?usp=sharing)

    > **Important:** After downloading, extract the contents of `env_exports.rar`. This will create a folder named `env_exports` containing all the necessary `.tar.gz` and `.yml` files used in the next step. The installation scripts assume this folder is located at `~/env_exports/`.

---

## Step 2: Conda Environments Installation

This repository contains exported conda environments for the GranD project. Once you have downloaded and extracted `env_exports.rar`, follow the instructions below to install them on your system.

### Prerequisites

- Conda or Miniconda installed
- Sufficient disk space for the environments

### Environment List

The following environments are available:

**Archive-based (.tar.gz):**
- `fastvlm`
- `grand_env_2`
- `grand_env_3`
- `grand_env_6`
- `grand_env_7`
- `grand_env_8`
- `grand_env_9`
- `grand_env_utils`
- `llama31`
- `sam`

**YAML-based (.yml + -pip.txt):**
- `grand_env_1` (contains editable packages: sam-hq, xformers)
- `grand_env_4` (contains editable package: eva_02)
- `grand_env_5` (contains editable package: detectron2)

---

### Installation Instructions

#### Method 1: Installing from .tar.gz Archives (Fast & Easy)

For environments with `.tar.gz` files, use this method:

```bash
# Choose your environment name (e.g., fastvlm, sam, llama31, etc.)
ENV_NAME="fastvlm"

# Create directory for the environment
mkdir -p ~/envs/$ENV_NAME

# Extract the archive (assuming env_exports is in your home directory)
tar -xzf ~/env_exports/${ENV_NAME}.tar.gz -C ~/envs/$ENV_NAME

# Activate the environment
source ~/envs/$ENV_NAME/bin/activate

# Cleanup prefixes (important for relocatability)
conda-unpack
```

**Example for multiple environments:**
```bash
for ENV in fastvlm grand_env_2 grand_env_3 sam llama31; do
    mkdir -p ~/envs/$ENV
    tar -xzf ~/env_exports/${ENV}.tar.gz -C ~/envs/$ENV
    source ~/envs/$ENV/bin/activate
    conda-unpack
    conda deactivate
done
```

---

#### Method 2: Installing from YAML + Pip Requirements

For `grand_env_1`, `grand_env_4`, and `grand_env_5`, use this method:

##### Step A: Create conda environment from YAML

```bash
# Choose your environment (grand_env_1, grand_env_4, or grand_env_5)
ENV_NAME="grand_env_1"

# Create the environment (assuming env_exports is in your home directory)
conda env create -f ~/env_exports/${ENV_NAME}.yml
```

##### Step B: Install pip packages

```bash
# Activate the environment
conda activate $ENV_NAME

# Install pip packages
pip install -r ~/env_exports/${ENV_NAME}-pip.txt
```

##### Step C: Install editable packages

**For grand_env_1:**
```bash
conda activate grand_env_1
pip install -e /path/to/main/project/folder/environments/sam-hq
pip install -e /path/to/main/project/folder/environments/xformers
```

**For grand_env_4:**
```bash
conda activate grand_env_4
pip install -e /path/to/main/project/folder/level_1_inference/5_eva_02
```

**For grand_env_5:**
```bash
conda activate grand_env_5
pip install -e /path/to/main/project/folder/environments/detectron2
```

> **Note:** Replace `/path/to/main/project/folder/` with the actual path to the code you downloaded from **[LinkPlaceHolder]**. These source directories must exist on your system.

---

## Step 3: Running the Pipeline

After setting up the project directories and conda environments, you can run the main pipeline.

1.  **Navigate** to the main project directory you downloaded from **[LinkPlaceHolder]**.

2.  **Edit the script** `per_dataset_pipline_iNatAg_subset.sh` with a text editor.

3.  **Adjust the path variables** at the top of the script to point to your directories:
    ```bash
    # Adjust these paths to match your system configuration

    # Path to the root of the downloaded dataset
    DATASETS_ROOT="/path/to/your/downloaded/dataset_folder"

    # Directory where all outputs will be saved
    OUTPUTS_ROOT="/path/to/your/desired/outputs_folder"

    # Directory containing the model checkpoints and weights
    CKPT_DIR="/path/to/your/main/project/folder/weights"
    ```

4.  **Execute the script** from your terminal:
    ```bash
    bash per_dataset_pipline_iNatAg_subset.sh
    ```

---

## Quick Reference

### List all installed environments
```bash
conda env list
```

### Activate an environment
```bash
# For tar.gz-based environments
source ~/envs/ENV_NAME/bin/activate

# For YAML-based environments
conda activate ENV_NAME
```

### Deactivate current environment
```bash
conda deactivate
```

### Remove an environment
```bash
# For tar.gz-based
rm -rf ~/envs/ENV_NAME

# For YAML-based
conda env remove -n ENV_NAME
```

---

## Troubleshooting

### Issue: `conda-unpack` not found
Install conda-pack in your base environment:
```bash
conda install -n base conda-pack
```

### Issue: Editable packages not found
Ensure the source directories for editable packages exist at the paths you specified. These directories are located inside the main project folder you downloaded from **[LinkPlaceHolder]**.

### Issue: Package conflicts during YAML installation
Try creating the environment with `--force`:
```bash
conda env create -f ~/env_exports/${ENV_NAME}.yml --force
```

---

## Environment Storage Locations

- **Archive-based environments:** `~/envs/ENV_NAME/`
- **YAML-based environments:** `~/miniconda3/envs/ENV_NAME/` (or `~/anaconda3/envs/`)

---

## Notes

- Archive-based installations (`.tar.gz`) are faster and preserve exact package versions.
- YAML-based installations rebuild environments and may fetch newer compatible versions.
- Editable packages allow you to modify source code and see changes immediately without reinstalling.
