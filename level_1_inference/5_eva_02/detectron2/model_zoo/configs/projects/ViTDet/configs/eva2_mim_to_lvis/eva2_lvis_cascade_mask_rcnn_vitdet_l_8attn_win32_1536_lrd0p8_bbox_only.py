# eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8_bbox_only.py
# UPDATED FOR CUSTOM DETECTION-ONLY DATASET - FIXED VERSION

from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator  # Changed from LVISEvaluator for custom datasets
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads

# Import the base model configuration - adjust this import path as needed
# If you don't have this exact import, find the equivalent MIM-to-COCO config
from ..eva2_mim_to_coco.eva2_coco_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8 import (
    dataloader,
    lr_multiplier,
    model,
    optimizer,
    train,
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
    # Remove LVIS-specific parameters like max_dets_per_image
)

# Remove old single-stage ROI heads components that don't exist in cascade
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]

# Configure the model for detection-only CASCADE
# CRITICAL: NO mask components mentioned at all
model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=80,  # Will be updated by training script based on your dataset
    
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
            test_score_thresh=0.05,  # Reasonable threshold for custom datasets
            test_topk_per_image=100,  # Reasonable max detections
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
            use_fed_loss=False,  # Disabled for custom datasets initially, training script may enable
            # Fed loss weights - will be disabled by training script for custom datasets
            # get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
            #     dataloader.train.dataset.names, 0.5
            # ),
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

# Training configuration - adjust for fine-tuning
optimizer.lr = 1e-4  # Will be adjusted by training script

train.max_iter = 40000  # Will be adjusted by training script
train.eval_period = 2500  # Will be adjusted by training script  
train.checkpointer.period = 2500  # Will be adjusted by training script

# Update LR schedule for training length - will be adjusted by training script
lr_multiplier.scheduler.milestones = [
    train.max_iter * 8 // 10, 
    train.max_iter * 9 // 10
]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 1000 / train.max_iter

# Dataloader configuration
dataloader.test.num_workers = 0  # Will be set by training script
dataloader.train.total_batch_size = 128  # Will be adjusted by training script based on GPU memory

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