"""
EVA-02 Configuration for Bounding Box Detection Only (No Masks)
Supports both eva-02-01 and eva-02-02 models with mask detection disabled.

This config is designed for datasets that only have bounding box annotations.
"""

from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import FastRCNNOutputLayers, FastRCNNConvFCHead, CascadeROIHeads

from detectron2.config import LazyCall as L
from detectron2.modeling.roi_heads import CascadeROIHeads
from detectron2.modeling.poolers import ROIAlign
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNConvFCHead, FastRCNNOutputLayers
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import ShapeSpec

# Import base configurations for both models
def get_base_config(model_variant="eva-02-01"):
    """
    Get base configuration for specified EVA-02 model variant.
    
    Args:
        model_variant: Either "eva-02-01" or "eva-02-02"
    """
    if model_variant == "eva-02-01":
        # Import eva-02-01 base config (O365 to COCO)
        try:
            from ..eva2_o365_to_coco.eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8 import (
                dataloader,
                model,
                train,
                lr_multiplier,
                optimizer,
            )
        except ImportError:
            # Fallback - create basic configuration
            from detectron2.config import LazyCall as L
            from detectron2.data import build_detection_train_loader, build_detection_test_loader
            from detectron2.evaluation import COCOEvaluator
            from detectron2.solver import WarmupParamScheduler
            from fvcore.common.param_scheduler import MultiStepParamScheduler
            
            dataloader = L(lambda: None)()
            dataloader.train = L(build_detection_train_loader)(
                dataset=L(lambda: None)(),
                mapper=L(lambda: None)(),
                sampler=None,
                total_batch_size=16,
                aspect_ratio_grouping=True,
                num_workers=4,
            )
            dataloader.test = L(build_detection_test_loader)(
                dataset=L(lambda: None)(),
                mapper=L(lambda: None)(),
                num_workers=4,
            )
            dataloader.evaluator = L(COCOEvaluator)(
                dataset_name="${..test.dataset.names}",
            )
            
            model = L(lambda: None)()
            train = L(lambda: None)()
            train.device = "cuda"
            train.amp = L(lambda: None)()
            train.amp.enabled = False
            train.ddp = L(lambda: None)()
            train.ddp.broadcast_buffers = False
            train.ddp.find_unused_parameters = False
            train.ddp.fp16_compression = False
            train.checkpointer = L(lambda: None)()
            train.checkpointer.period = 5000
            train.checkpointer.max_to_keep = 100
            train.eval_period = 5000
            train.log_period = 20
            train.output_dir = "./output"
            train.max_iter = 90000
            train.seed = 42
            train.model_ema = L(lambda: None)()
            train.model_ema.enabled = False
            train.model_ema.decay = 0.999
            train.model_ema.use_ema_weights_for_eval_only = False
            
            lr_multiplier = L(WarmupParamScheduler)(
                scheduler=L(MultiStepParamScheduler)(
                    values=[1.0, 0.1, 0.01],
                    milestones=[60000, 80000],
                    num_updates=90000,
                ),
                warmup_length=1000 / 90000,
                warmup_method="linear",
                warmup_factor=0.001,
            )
            
            optimizer = L(lambda: None)()
            optimizer.lr = 0.0001
            
    elif model_variant == "eva-02-02":
        # Import eva-02-02 base config (MIM to COCO)  
        try:
            from ..eva2_mim_to_coco.eva2_coco_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8 import (
                dataloader,
                lr_multiplier,
                model,
                optimizer,
                train,
            )
        except ImportError:
            # Fallback - same as above
            from detectron2.config import LazyCall as L
            from detectron2.data import build_detection_train_loader, build_detection_test_loader
            from detectron2.evaluation import COCOEvaluator
            from detectron2.solver import WarmupParamScheduler
            from fvcore.common.param_scheduler import MultiStepParamScheduler
            
            dataloader = L(lambda: None)()
            dataloader.train = L(build_detection_train_loader)(
                dataset=L(lambda: None)(),
                mapper=L(lambda: None)(),
                sampler=None,
                total_batch_size=16,
                aspect_ratio_grouping=True,
                num_workers=4,
            )
            dataloader.test = L(build_detection_test_loader)(
                dataset=L(lambda: None)(),
                mapper=L(lambda: None)(),
                num_workers=4,
            )
            dataloader.evaluator = L(COCOEvaluator)(
                dataset_name="${..test.dataset.names}",
            )
            
            model = L(lambda: None)()
            train = L(lambda: None)()
            train.device = "cuda"
            train.amp = L(lambda: None)()
            train.amp.enabled = False
            train.ddp = L(lambda: None)()
            train.ddp.broadcast_buffers = False
            train.ddp.find_unused_parameters = False
            train.ddp.fp16_compression = False
            train.checkpointer = L(lambda: None)()
            train.checkpointer.period = 5000
            train.checkpointer.max_to_keep = 100
            train.eval_period = 5000
            train.log_period = 20
            train.output_dir = "./output"
            train.max_iter = 90000
            train.seed = 42
            train.model_ema = L(lambda: None)()
            train.model_ema.enabled = False
            train.model_ema.decay = 0.999
            train.model_ema.use_ema_weights_for_eval_only = False
            
            lr_multiplier = L(WarmupParamScheduler)(
                scheduler=L(MultiStepParamScheduler)(
                    values=[1.0, 0.1, 0.01],
                    milestones=[60000, 80000],
                    num_updates=90000,
                ),
                warmup_length=1000 / 90000,
                warmup_method="linear",
                warmup_factor=0.001,
            )
            
            optimizer = L(lambda: None)()
            optimizer.lr = 0.0001
    else:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    return dataloader, model, train, lr_multiplier, optimizer


# Configuration function that can be called with model variant
def create_bbox_only_config(model_variant="eva-02-01", num_classes=80):
    """
    Create bbox-only configuration for EVA-02 models.
    
    Args:
        model_variant: Either "eva-02-01" or "eva-02-02"
        num_classes: Number of object classes in your dataset
    """
    
    # Get base configuration
    dataloader, model, train, lr_multiplier, optimizer = get_base_config(model_variant)
    
    # Configure dataloader for bbox-only detection
    # CRITICAL: Disable mask-related data loading
    if hasattr(dataloader.train, 'mapper'):
        dataloader.train.mapper.use_instance_mask = False
        dataloader.train.mapper.recompute_boxes = False
        if hasattr(dataloader.train.mapper, 'use_keypoint'):
            dataloader.train.mapper.use_keypoint = False
        if hasattr(dataloader.train.mapper, 'keypoint_hflip_indices'):
            dataloader.train.mapper.keypoint_hflip_indices = None
    
    if hasattr(dataloader.test, 'mapper'):
        dataloader.test.mapper.use_instance_mask = False  
        dataloader.test.mapper.recompute_boxes = False
        if hasattr(dataloader.test.mapper, 'use_keypoint'):
            dataloader.test.mapper.use_keypoint = False
        if hasattr(dataloader.test.mapper, 'keypoint_hflip_indices'):
            dataloader.test.mapper.keypoint_hflip_indices = None
    
    # Set up evaluator for COCO-format evaluation (bbox only)
    dataloader.evaluator = L(COCOEvaluator)(
        dataset_name="${..test.dataset.names}",
        tasks=("bbox",),  # Only evaluate bounding boxes, not segmentation
    )
    
    # Configure model for bbox-only detection (remove mask components)
    # Update ROI heads to remove mask prediction
    model.roi_heads.update(
        _target_=CascadeROIHeads,
        num_classes=num_classes,
        # REMOVED: mask_in_features, mask_pooler, mask_head (no masks!)
        # Only keep box-related components
        box_heads=[
            L(FastRCNNConvFCHead)(
                input_shape=ShapeSpec(channels=256, height=7, width=7),
                conv_dims=[256, 256, 256, 256],
                fc_dims=[1024],
                conv_norm="LN",
            )
            for _ in range(3)  # 3 cascade stages
        ],
        box_predictors=[
            L(FastRCNNOutputLayers)(
                input_shape=ShapeSpec(channels=1024),
                box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
                num_classes=num_classes,
                test_score_thresh=0.05,
                test_topk_per_image=100,
                cls_agnostic_bbox_reg=True,
                use_sigmoid_ce=True,
                use_fed_loss=False,  # Disable fed loss for custom datasets
            )
            for (w1, w2) in [(10, 5), (20, 10), (30, 15)]  # Cascade weights
        ],
        proposal_matchers=[
            L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
            for th in [0.5, 0.6, 0.7]  # Cascade IoU thresholds
        ],
    )


    # # For simplicity, define once
    # box_head_shape = ShapeSpec(channels=256, height=7, width=7)

    # # Inject roi_heads directly
    # model.roi_heads = L(CascadeROIHeads)(
    #     num_classes=num_classes,  # 👈 replace with your detection class count
    #     box_heads=[
    #         L(FastRCNNConvFCHead)(
    #             input_shape=box_head_shape,
    #             conv_dims=[256, 256, 256, 256],
    #             fc_dims=[1024],
    #             conv_norm="LN",
    #         )
    #         for _ in range(3)
    #     ],
    #     box_predictors=[
    #         L(FastRCNNOutputLayers)(
    #             input_shape=ShapeSpec(channels=1024),
    #             box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
    #             num_classes="${..num_classes}",
    #             test_score_thresh=0.02,
    #             test_topk_per_image=300,
    #             cls_agnostic_bbox_reg=True,
    #             use_sigmoid_ce=True,
    #             use_fed_loss=False,  # 👈 usually disabled for custom dataset
    #             get_fed_loss_cls_weights=None,
    #         )
    #         for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    #     ],
    #     proposal_matchers=[
    #         L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
    #         for th in [0.5, 0.6, 0.7]
    #     ],
    #     mask_on=False,  # 👈 explicitly disable mask head
    # )

    
    # Additional model configuration to ensure masks are disabled
    if hasattr(model, 'mask_on'):
        model.mask_on = False
    if hasattr(model.roi_heads, 'mask_on'):
        model.roi_heads.mask_on = False
    
    # Training configuration optimized for bbox detection
    train.max_iter = 10000  # Reasonable default for fine-tuning
    train.eval_period = 1000
    train.checkpointer.period = 1000
    train.log_period = 50
    
    # Learning rate and optimization
    optimizer.lr = 0.0001  # Good default for fine-tuning
    
    # Update lr_multiplier for shorter training
    if hasattr(lr_multiplier, 'scheduler') and hasattr(lr_multiplier.scheduler, 'milestones'):
        lr_multiplier.scheduler.milestones = [
            int(train.max_iter * 0.7),  # Reduce LR at 70% of training
            int(train.max_iter * 0.9),  # Reduce LR again at 90% of training
        ]
        lr_multiplier.scheduler.num_updates = train.max_iter
        lr_multiplier.warmup_length = 500 / train.max_iter  # Warmup for first 500 iterations
    
    # Batch size configuration
    dataloader.train.total_batch_size = 8  # Reasonable default
    dataloader.train.num_workers = 4
    dataloader.test.num_workers = 4
    
    return {
        'dataloader': dataloader,
        'model': model, 
        'train': train,
        'lr_multiplier': lr_multiplier,
        'optimizer': optimizer,
    }


# Default configuration for eva-02-01 (can be overridden)
config_data = create_bbox_only_config(model_variant="eva-02-01", num_classes=80)
dataloader = config_data['dataloader']
model = config_data['model']
train = config_data['train']
lr_multiplier = config_data['lr_multiplier']
optimizer = config_data['optimizer']