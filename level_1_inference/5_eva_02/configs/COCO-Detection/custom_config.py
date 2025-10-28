# eva02_custom_config.py
from detectron2.config import LazyConfig
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
import os

# Register your custom dataset
def register_custom_datasets(data_dir):
    """
    Register your COCO format datasets
    Args:
        data_dir: Directory containing your annotation files and images
    """
    # Register training dataset
    register_coco_instances(
        "custom_train", 
        {}, 
        os.path.join(data_dir, "annotation_train.json"),  # path to your train annotations
        os.path.join(data_dir, "train_images")  # path to your train images directory
    )
    
    # Register test/validation dataset
    register_coco_instances(
        "custom_test", 
        {}, 
        os.path.join(data_dir, "annotation_test.json"),   # path to your test annotations
        os.path.join(data_dir, "test_images")   # path to your test images directory
    )

# Call the registration function
# Update this path to your actual data directory
DATA_DIR = r"E:\Desktop\GLaMM"
register_custom_datasets(DATA_DIR)

# Get the number of classes from your dataset
# You'll need to update this based on your actual number of classes
NUM_CLASSES = 68  # Update this to match your dataset

# Base configuration - modify based on available configs
from ..common.optim import AdamW as optimizer
from ..common.coco_schedule import lr_multiplier_1x as lr_multiplier  
from ..common.data.coco import dataloader
from ..common.train import train

# Model configuration
model = LazyConfig.new_dict()
model.backbone = LazyConfig.new_dict()
model.backbone._target_ = "path.to.eva02.backbone"  # Update with actual EVA-02 backbone path

# Update model parameters for your dataset
model.roi_heads = LazyConfig.new_dict()
model.roi_heads.num_classes = NUM_CLASSES

# Dataset configuration
dataloader.train.dataset.names = ["custom_train"]
dataloader.test.dataset.names = ["custom_test"]

# Training configuration
train.init_checkpoint = "../../eva02_L_lvis_sys.pth"  # or eva02_L_lvis_sys_o365.pth
train.output_dir = "./fine-tune_eva02_L_lvis_sys"
train.max_iter = 5000  # Adjust based on your dataset size
train.eval_period = 500  # Evaluate every 500 iterations
train.checkpointer.period = 1000  # Save checkpoint every 1000 iterations

# Optimizer settings
optimizer.lr = 1e-4  # Lower learning rate for fine-tuning
optimizer.weight_decay = 1e-4

# Enable mixed precision training (optional, for faster training)
train.amp.enabled = True

# Data augmentation settings (optional)
dataloader.train.mapper.augmentations = [
    dict(_target_="detectron2.data.transforms.ResizeShortestEdge",
         short_edge_length=[640, 672, 704, 736, 768, 800],
         max_size=1333,
         sample_style="choice"),
    dict(_target_="detectron2.data.transforms.RandomFlip"),
]

# Test configuration
dataloader.test.mapper.augmentations = [
    dict(_target_="detectron2.data.transforms.ResizeShortestEdge",
         short_edge_length=800,
         max_size=1333),
]

# Solver configuration for fine-tuning
train.warmup_factor = 0.1
train.warmup_iters = 1000