# eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8_bbox_only.py
# UPDATED FOR CUSTOM DETECTION-ONLY DATASET - FIXED VERSION

from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator  # Changed from LVISEvaluator for custom datasets
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads

# Import the base configuration - adjust path as needed
from ..eva2_o365_to_coco.eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8 import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

# Configure dataset for your custom detection dataset
# These will be updated by the training script
dataloader.train.dataset.names = "custom_dataset_train"
dataloader.test.dataset.names = "custom_dataset_test"

# Use standard sampler for most custom datasets
# Uncomment below if you have severe class imbalance
# dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
#     repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
#         dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001
#     )
# )

# Configure evaluator for COCO format (suitable for most custom datasets)
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
    # Remove LVIS-specific parameters
)

# Remove old single-stage ROI heads components that don't exist in cascade
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

# Configure the model for detection-only CASCADE
# CRITICAL: NO mask components mentioned at all
model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=1203,  # Will be updated by training script based on your dataset
    
    # CRITICAL: Do NOT include mask_head or mask_pooler AT ALL
    # Don't even mention them, not even as None
    
    # Configure box heads for cascade detection (3 stages)
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            num_classes="${...num_classes}",
            test_score_thresh=0.02,  # Good for fine-tuned models
            test_topk_per_image=300,  # LVIS setting, will be adjusted by script
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
            use_fed_loss=True,  # Will be disabled by script for custom datasets
            get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
                dataloader.train.dataset.names, 0.5
            ),
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# CRITICAL: Disable mask prediction globally
model.mask_on = False

# Dataloader configuration
dataloader.test.num_workers = 0  # Will be set by training script
dataloader.train.total_batch_size = 64  # Will be adjusted by training script for GPU memory

# Training configuration - using the O365 training schedule initially
train.max_iter = 70000  # Will be adjusted by training script
train.eval_period = 5000  # Will be adjusted by training script
train.checkpointer.period = 5000  # Will be adjusted by training script

# Learning rate schedule - will be adjusted by training script for shorter fine-tuning
# lr_multiplier and optimizer configurations are inherited from base config

# Configure mappers for detection-only training
# These settings ensure no mask data is expected
try:
    # Train mapper
    dataloader.train.mapper.use_instance_mask = False
    dataloader.train.mapper.use_keypoint = False
    dataloader.train.mapper.recompute_boxes = False
    
    # Test mapper  
    dataloader.test.mapper.use_instance_mask = False
    dataloader.test.mapper.use_keypoint = False
    dataloader.test.mapper.recompute_boxes = False
except AttributeError:
    # Mappers will be configured by the training script if not available here
    pass

# Optional: Adjust base learning rate for fine-tuning
# optimizer.lr = 1e-4  # Will be set by training script