Download the dataset from "https://drive.google.com/file/d/1GQz7HQw6Mvlqw4GoRRxy-MwFxrDejp4d/view?usp=sharing"

The in "env_exports.rar" the link "https://drive.google.com/file/d/1RyVw6k3DdFddwh9_rBvU3NoJsKkVaHGl/view?usp=sharing"

The main folder which contain all the cods and weights is in "LinkPlaceHolder"

The script to run the pipline is "per_dataset_pipline_iNatAg_subset.sh", inside it, adjust DATASETS_ROOT, OUTPUTS_ROOT, CKPT_DIR to point to the correct path

Adjust the following "README.md" to add the above information.

Conda Environments Installation Guide

This repository contains exported conda environments for the GranD project. Follow the instructions below to install them on your system.

Prerequisites

Conda or Miniconda installed

Sufficient disk space for the environments

Environment List

The following environments are available:

Archive-based (.tar.gz):

fastvlm

grand_env_2

grand_env_3

grand_env_6

grand_env_7

grand_env_8

grand_env_9

grand_env_utils

llama31

sam

YAML-based (.yml + -pip.txt):

grand_env_1 (contains editable packages: sam-hq, xformers)

grand_env_4 (contains editable package: eva_02)

grand_env_5 (contains editable package: detectron2)

Installation Instructions
Method 1: Installing from .tar.gz Archives (Fast & Easy)

For environments with .tar.gz files, use this method:

code
Bash
download
content_copy
expand_less
# Choose your environment name (e.g., fastvlm, sam, llama31, etc.)
ENV_NAME="fastvlm"

# Create directory for the environment
mkdir -p ~/envs/$ENV_NAME

# Extract the archive
tar -xzf ~/env_exports/${ENV_NAME}.tar.gz -C ~/envs/$ENV_NAME

# Activate the environment
source ~/envs/$ENV_NAME/bin/activate

# Cleanup prefixes (important for relocatability)
conda-unpack

Example for multiple environments:

code
Bash
download
content_copy
expand_less
for ENV in fastvlm grand_env_2 grand_env_3 sam llama31; do
    mkdir -p ~/envs/$ENV
    tar -xzf ~/env_exports/${ENV}.tar.gz -C ~/envs/$ENV
    source ~/envs/$ENV/bin/activate
    conda-unpack
    conda deactivate
done
Method 2: Installing from YAML + Pip Requirements

For grand_env_1, grand_env_4, and grand_env_5, use this method:

Step 1: Create conda environment from YAML
code
Bash
download
content_copy
expand_less
# Choose your environment (grand_env_1, grand_env_4, or grand_env_5)
ENV_NAME="grand_env_1"

# Create the environment
conda env create -f ~/env_exports/${ENV_NAME}.yml
Step 2: Install pip packages
code
Bash
download
content_copy
expand_less
# Activate the environment
conda activate $ENV_NAME

# Install pip packages
pip install -r ~/env_exports/${ENV_NAME}-pip.txt
Step 3: Install editable packages

For grand_env_1:

code
Bash
download
content_copy
expand_less
conda activate grand_env_1
pip install -e /home/abood/groundingLMM/GranD/environments/sam-hq
pip install -e /home/abood/groundingLMM/GranD/environments/xformers

For grand_env_4:

code
Bash
download
content_copy
expand_less
conda activate grand_env_4
pip install -e /home/abood/groundingLMM/GranD/level_1_inference/5_eva_02

For grand_env_5:

code
Bash
download
content_copy
expand_less
conda activate grand_env_5
pip install -e /home/abood/groundingLMM/GranD/environments/detectron2

Note: Make sure the editable package directories exist on your system. If you're installing on a different machine, you'll need to clone/copy these source directories first.

Quick Reference
List all installed environments
code
Bash
download
content_copy
expand_less
conda env list
Activate an environment
code
Bash
download
content_copy
expand_less
# For tar.gz-based environments
source ~/envs/ENV_NAME/bin/activate

# For YAML-based environments
conda activate ENV_NAME
Deactivate current environment
code
Bash
download
content_copy
expand_less
conda deactivate
Remove an environment
code
Bash
download
content_copy
expand_less
# For tar.gz-based
rm -rf ~/envs/ENV_NAME

# For YAML-based
conda env remove -n ENV_NAME
Troubleshooting
Issue: conda-unpack not found

Install conda-pack in your base environment:

code
Bash
download
content_copy
expand_less
conda install -n base conda-pack
Issue: Editable packages not found

Ensure the source directories for editable packages exist:

/home/abood/groundingLMM/GranD/environments/sam-hq

/home/abood/groundingLMM/GranD/environments/xformers

/home/abood/groundingLMM/GranD/level_1_inference/5_eva_02

/home/abood/groundingLMM/GranD/environments/detectron2

If installing on a different machine, clone or copy these directories first.

Issue: Package conflicts during YAML installation

Try creating the environment with --force:

code
Bash
download
content_copy
expand_less
conda env create -f ~/env_exports/${ENV_NAME}.yml --force
Environment Storage Locations

Archive-based environments: ~/envs/ENV_NAME/

YAML-based environments: ~/miniconda3/envs/ENV_NAME/ (or ~/anaconda3/envs/)

Notes

Archive-based installations (.tar.gz) are faster and preserve exact package versions

YAML-based installations rebuild environments and may fetch newer compatible versions

Editable packages allow you to modify source code and see changes immediately without reinstalling
