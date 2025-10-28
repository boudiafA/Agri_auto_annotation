
# GranD Project Setup and Environment Installation Guide

This guide provides comprehensive instructions for setting up the GranD project, including cloning the code, downloading the necessary data, installing the required conda environments, and running the processing pipeline.

## Step 1: Prerequisites

Before you begin, ensure you have the following installed on your system:
- **Git:** For cloning the source code repository.
- **Conda / Miniconda:** For managing the Python environments.

## Step 2: Setup and Downloads

This step covers getting the code, dataset, model checkpoints, and environment files. For clarity, we recommend creating a central workspace directory to hold all project-related files.

1.  **Clone the Code Repository**
    Open your terminal and clone the main project repository from GitHub:
    ```bash
    git clone https://github.com/your-username/grand-project.git
    cd grand-project
    ```
    *(Note: Replace `https://github.com/your-username/grand-project.git` with the actual repository URL.)*

2.  **Download the Model Checkpoints**
    The model weights and checkpoints are required for running the pipeline. Download and unzip them.
    - **Link:** [https://drive.google.com/file/d/12sgu6c9ZzBjwwhMjPEUniM_ASj4XlIzB/view?usp=sharing](https://drive.google.com/file/d/12sgu6c9ZzBjwwhMjPEUniM_ASj4XlIzB/view?usp=sharing)
3.  **Download the Dataset**
    Download and unzip the project dataset.
    - **Link:** [https://drive.google.com/file/d/1GQz7HQw6Mvlqw4GoRRxy-MwFxrDejp4d/view?usp=sharing](https://drive.google.com/file/d/1GQz7HQw6Mvlqw4GoRRxy-MwFxrDejp4d/view?usp=sharing)

4.  **Download Conda Environment Files**
    The environment definition files are bundled in `env_exports.rar`.
    - **Link:** [https://drive.google.com/file/d/1RyVw6k3DdFddwh9_rBvU3NoJsKkVaHGl/view?usp=sharing](https://drive.google.com/file/d/1RyVw6k3DdFddwh9_rBvU3NoJsKkVaHGl/view?usp=sharing)

    > **Important:** After downloading, extract the contents of `env_exports.rar`. This will create a folder named `env_exports` containing all the necessary `.tar.gz` and `.yml` files used in the next step. The installation scripts assume this folder is located at `~/env_exports/`.

### Recommended Directory Structure

After completing the downloads and cloning, we suggest organizing your workspace as follows. This structure will make it easier to configure the paths in the run scripts.

```
/path/to/your/workspace/
├── grand-project/         # (1. The cloned GitHub repository)
│   ├── per_dataset_pipline_iNatAg_subset.sh
│   └── ...
├── datasets/              # (3. The unzipped dataset folder)
│   └── ...
└── checkpoints/           # (2. The unzipped checkpoints folder)
```

---

## Step 3: Conda Environments Installation

Follow the instructions below to install the necessary conda environments using the files from `env_exports`.

### Prerequisites

- Conda or Miniconda is installed.
- The `env_exports.rar` file has been downloaded and extracted to `~/env_exports/`.

### Environment List

The following environments are required for the pipeline:

**Archive-based (.tar.gz):**
- `fastvlm`, `grand_env_2`, `grand_env_3`, `grand_env_6`, `grand_env_7`, `grand_env_8`, `grand_env_9`, `grand_env_utils`, `llama31`, `sam`

**YAML-based (.yml + -pip.txt):**
- `grand_env_1`, `grand_env_4`, `grand_env_5` (these contain editable packages)

---

### Installation Instructions

#### Method 1: Installing from .tar.gz Archives (Fast & Easy)

For environments packaged as `.tar.gz` files:

```bash
# Example for a single environment
ENV_NAME="fastvlm"
mkdir -p ~/envs/$ENV_NAME
tar -xzf ~/env_exports/${ENV_NAME}.tar.gz -C ~/envs/$ENV_NAME
source ~/envs/$ENV_NAME/bin/activate
conda-unpack # This step is crucial for making the environment relocatable
conda deactivate
```

**Script to install multiple environments:**
```bash
for ENV in fastvlm grand_env_2 grand_env_3 sam llama31; do # Add other envs as needed
    echo "Installing $ENV..."
    mkdir -p ~/envs/$ENV
    tar -xzf ~/env_exports/${ENV}.tar.gz -C ~/envs/$ENV
    source ~/envs/$ENV/bin/activate
    conda-unpack
    conda deactivate
done
```

---

#### Method 2: Installing from YAML + Pip Requirements

For `grand_env_1`, `grand_env_4`, and `grand_env_5`, follow these steps:

##### A: Create conda environment from YAML

```bash
# Example for grand_env_1
ENV_NAME="grand_env_1"
conda env create -f ~/env_exports/${ENV_NAME}.yml
```

##### B: Install pip packages

```bash
conda activate $ENV_NAME
pip install -r ~/env_exports/${ENV_NAME}-pip.txt
```

##### C: Install editable packages

These packages need to be linked to the source code within your cloned repository. **Replace `/path/to/your/workspace/` with the actual path on your system.**

**For `grand_env_1`:**
```bash
conda activate grand_env_1
pip install -e /path/to/your/workspace/grand-project/environments/sam-hq
pip install -e /path/to/your/workspace/grand-project/environments/xformers
```

**For `grand_env_4`:**
```bash
conda activate grand_env_4
pip install -e /path/to/your/workspace/grand-project/level_1_inference/5_eva_02
```

**For `grand_env_5`:**
```bash
conda activate grand_env_5
pip install -e /path/to/your/workspace/grand-project/environments/detectron2
```

---

## Step 4: Running the Pipeline

After setting up the directories and conda environments, you can configure and run the main pipeline.

1.  **Navigate** to the cloned project directory:
    ```bash
    cd /path/to/your/workspace/grand-project/
    ```

2.  **Edit the script** `per_dataset_pipline_iNatAg_subset.sh` with a text editor.

3.  **Adjust the path variables** at the top of the script. Using the recommended directory structure from Step 2, your paths would look like this:
    ```bash
    # === ADJUST THESE PATHS TO MATCH YOUR SYSTEM CONFIGURATION ===

    # Path to the root of the downloaded and unzipped dataset
    DATASETS_ROOT="/path/to/your/workspace/datasets"

    # Directory where all pipeline outputs will be saved
    OUTPUTS_ROOT="/path/to/your/workspace/grand-project/outputs" # An 'outputs' folder inside the repo is a good choice

    # Directory containing the downloaded model checkpoints
    CKPT_DIR="/path/to/your/workspace/checkpoints"
    
    # =============================================================
    ```

4.  **Execute the script** from your terminal:
    ```bash
    bash per_dataset_pipline_iNatAg_subset.sh
    ```

---

## Quick Reference

- **List all installed environments:** `conda env list`
- **Activate an environment:**
  - `source ~/envs/ENV_NAME/bin/activate` (for .tar.gz-based)
  - `conda activate ENV_NAME` (for .yml-based)
- **Deactivate environment:** `conda deactivate`
- **Remove an environment:**
  - `rm -rf ~/envs/ENV_NAME` (for .tar.gz-based)
  - `conda env remove -n ENV_NAME` (for .yml-based)

---

## Troubleshooting

- **`conda-unpack` not found:** Install it in your base environment: `conda install -n base conda-pack`.
- **Editable packages not found:** Double-check that the path `/path/to/your/workspace/grand-project/` is correct and points to your cloned repository.
- **Package conflicts during YAML installation:** Try creating the environment with the `--force` flag: `conda env create -f ~/env_exports/${ENV_NAME}.yml --force`.
