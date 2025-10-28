# # (grand_env_4) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_1_inference/5_eva_02$ python eva2_FineTune.py
# # EVA-02 Fine-tuning Script for Detection Only (BBOX) - FIXED VERSION
# # Fixed version with proper detection-only configuration

import argparse
import json
import os
import torch
import torch.nn.parallel
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from detectron2.data import DatasetMapper
from detectron2.config import LazyCall as L
from detectron2.engine.hooks import HookBase
import logging
import time
from tqdm import tqdm
import gc
import random
import tempfile
import psutil
from contextlib import contextmanager

# Import DDP utilities from your original code
from ddp import *

# Config paths - same as inference code
eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
                                     "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
eva02_L_lvis_sys_config_path = ("projects/ViTDet/configs/eva2_mim_to_lvis/"
                                "eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py")

# Checkpoint paths - same as inference code
eva02_L_lvis_sys_o365_ckpt_path = ("eva02_L_lvis_sys_o365.pth")
eva02_L_lvis_sys_ckpt_path = ("eva02_L_lvis_sys.pth")

# ========================================================================================
# ENHANCED MEMORY MANAGEMENT FUNCTIONS WITH TQDM
# ========================================================================================

def get_gpu_memory_info():
    """Get current GPU memory usage information"""
    if not torch.cuda.is_available():
        return "CUDA not available"
    
    device = torch.cuda.current_device()
    allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
    cached = torch.cuda.memory_reserved(device) / 1024**3      # GB
    
    try:
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB"
    except:
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"

def get_cpu_memory_info():
    """Get current CPU memory usage information"""
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / 1024**3  # GB
    system_memory = psutil.virtual_memory()
    return f"CPU Memory - Process: {cpu_memory:.2f}GB, System: {system_memory.percent:.1f}% used"

def comprehensive_memory_cleanup(verbose=True):
    """Comprehensive memory cleanup function"""
    logger = logging.getLogger(__name__)
    
    if verbose:
        logger.info(f"Before cleanup: {get_gpu_memory_info()}")
        logger.info(f"Before cleanup: {get_cpu_memory_info()}")
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Additional cleanup for persistent memory
        try:
            torch.cuda.ipc_collect()
        except:
            pass  # Not available in all PyTorch versions
    
    # Additional Python garbage collection
    gc.collect()
    
    if verbose:
        logger.info(f"After cleanup: {get_gpu_memory_info()}")
        logger.info(f"After cleanup: {get_cpu_memory_info()}")

def periodic_memory_cleanup(iteration, cleanup_interval=1000, verbose=False):
    """Perform memory cleanup at regular intervals during training"""
    if iteration % cleanup_interval == 0:
        logger = logging.getLogger(__name__)
        if verbose:
            logger.info(f"Iteration {iteration}: Performing periodic memory cleanup")
        comprehensive_memory_cleanup(verbose=verbose)

@contextmanager
def memory_managed_operation(operation_name="Operation", cleanup_after=True):
    """Context manager for memory-managed operations"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {operation_name}")
    logger.info(f"Before {operation_name}: {get_gpu_memory_info()}")
    
    try:
        yield
    except Exception as e:
        logger.error(f"Error during {operation_name}: {e}")
        comprehensive_memory_cleanup(verbose=True)
        raise
    finally:
        if cleanup_after:
            logger.info(f"Cleaning up after {operation_name}")
            comprehensive_memory_cleanup(verbose=False)

def safe_model_deletion(model, model_name="model"):
    """Safely delete a model and cleanup associated memory"""
    logger = logging.getLogger(__name__)
    
    if model is not None:
        logger.info(f"Deleting {model_name}")
        
        # Move model to CPU first to free GPU memory
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
        except:
            pass  # In case model is already on CPU or cannot be moved
        
        # Delete the model
        del model
        
        # Force cleanup
        comprehensive_memory_cleanup(verbose=False)
        logger.info(f"{model_name} deleted and memory cleaned")

def estimate_memory_usage(train_batch_size, eval_batch_size, model_name="eva-02"):
    """
    Estimate memory usage for training and evaluation
    
    Args:
        train_batch_size (int): Training batch size
        eval_batch_size (int): Evaluation batch size
        model_name (str): Model name for size estimation
    
    Returns:
        dict: Memory usage estimates
    """
    # Rough estimates for EVA-02 Large model (in GB)
    base_model_memory = 3.5  # Model weights and buffers
    
    # Memory per sample (rough estimates)
    if model_name.startswith("eva-02"):
        memory_per_train_sample = 1.2  # Includes gradients, optimizer states, activations
        memory_per_eval_sample = 0.4   # Only forward pass activations
    else:
        memory_per_train_sample = 0.8   # Conservative estimate
        memory_per_eval_sample = 0.3
    
    train_memory = base_model_memory + (train_batch_size * memory_per_train_sample)
    eval_memory = base_model_memory + (eval_batch_size * memory_per_eval_sample)
    
    return {
        "base_model": base_model_memory,
        "training_total": train_memory,
        "evaluation_total": eval_memory,
        "training_batch_overhead": train_batch_size * memory_per_train_sample,
        "evaluation_batch_overhead": eval_batch_size * memory_per_eval_sample
    }

def get_optimal_eval_batch_size(train_batch_size, available_memory_gb=None):
    """
    Calculate optimal evaluation batch size based on training batch size and available memory
    
    Args:
        train_batch_size (int): Training batch size per GPU
        available_memory_gb (float): Available GPU memory in GB (optional)
    
    Returns:
        int: Recommended evaluation batch size
    """
    # Base multiplier: evaluation typically uses 60-70% less memory per sample
    # because no gradients, optimizer states, or intermediate activations for backprop
    base_multiplier = 2.5
    
    if available_memory_gb is not None:
        # Adjust based on available memory
        if available_memory_gb < 8:
            base_multiplier = 1.5  # Conservative for low memory
        elif available_memory_gb > 16:
            base_multiplier = 3.5  # More aggressive for high memory
        elif available_memory_gb > 24:
            base_multiplier = 4.0  # Very aggressive for very high memory
    
    # Calculate optimal batch size
    if train_batch_size == 1:
        return max(2, min(6, int(train_batch_size * base_multiplier)))
    elif train_batch_size <= 4:
        return max(train_batch_size, min(12, int(train_batch_size * base_multiplier)))
    else:
        return max(train_batch_size, int(train_batch_size * 1.5))

def memory_efficient_evaluation(model, test_loader, evaluator, logger, cleanup_interval=50, timeout_minutes=30):
    """
    Perform memory-efficient evaluation with tqdm progress bar
    
    Args:
        model: The model to evaluate
        test_loader: Test data loader
        evaluator: COCO evaluator
        logger: Logger instance
        cleanup_interval: How often to perform memory cleanup during evaluation
        timeout_minutes: Maximum time to wait for evaluation (in minutes)
    
    Returns:
        dict: Evaluation results
    """
    logger.info("🚀 Starting memory-efficient evaluation")
    logger.info(f"⏱️  Timeout: {timeout_minutes} minutes")
    logger.info(f"💾 {get_gpu_memory_info()}")
    
    model.eval()
    
    # Estimate total batches for progress reporting
    try:
        total_batches = len(test_loader)
        logger.info(f"📊 Total evaluation batches: {total_batches}")
    except:
        total_batches = None
        logger.info("❓ Cannot determine total batch count")
    
    # Check if test_loader is empty
    if total_batches == 0:
        logger.error("❌ Test loader is empty! No data to evaluate.")
        return {}
    
    # Perform evaluation with tqdm progress bar
    with torch.no_grad():
        evaluator.reset()
        logger.info("🔄 Evaluator reset, starting inference...")
        
        total_samples = 0
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60
        
        # Create progress bar
        pbar = tqdm(
            test_loader,
            desc="🔍 Evaluating",
            total=total_batches,
            unit="batch",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Samples: {postfix}]'
        )
        
        try:
            for idx, inputs in enumerate(pbar):
                # Check for timeout
                elapsed = time.time() - start_time
                if elapsed > timeout_seconds:
                    pbar.close()
                    logger.error(f"⏰ Evaluation timeout after {elapsed:.1f} seconds!")
                    break
                
                # Update progress bar postfix with current stats
                pbar.set_postfix_str(f"{total_samples}")
                
                # Validate inputs
                if not inputs:
                    pbar.set_description("⚠️  Empty batch, skipping")
                    continue
                    
                # Run inference
                try:
                    outputs = model(inputs)
                    if not outputs:
                        pbar.set_description("⚠️  Empty outputs")
                        continue
                        
                    evaluator.process(inputs, outputs)
                    total_samples += len(inputs)
                    
                    # Update description with success
                    pbar.set_description("🔍 Evaluating")
                    
                except Exception as e:
                    pbar.set_description(f"❌ Error in batch {idx}")
                    logger.error(f"Error processing batch {idx}: {e}")
                    continue
                
                # Periodic memory cleanup during evaluation
                if (idx + 1) % cleanup_interval == 0:
                    # Light cleanup during evaluation
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Update progress bar with memory info
                    allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                    pbar.set_description(f"🔍 Evaluating (GPU: {allocated:.1f}GB)")
            
        finally:
            pbar.close()
        
        eval_time = time.time() - start_time
        logger.info(f"✅ Inference completed! Processed {total_samples} samples in {eval_time:.2f}s")
        
        if total_samples == 0:
            logger.error("❌ No samples were processed during evaluation!")
            return {}
            
        # Computing final results with progress indication
        logger.info("🧮 Computing evaluation metrics...")
        
        # Show a simple spinner while computing metrics
        with tqdm(desc="🧮 Computing metrics", bar_format='{desc}', leave=False) as pbar:
            try:
                results = evaluator.evaluate()
                pbar.set_description("✅ Metrics computed!")
                if not results:
                    logger.warning("⚠️  Evaluator returned empty results!")
                    return {}
                logger.info("🎯 Evaluation metrics computation completed!")
                return results
            except Exception as e:
                pbar.set_description("❌ Metrics computation failed!")
                logger.error(f"Error computing evaluation metrics: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {}

class MemoryMonitoringHook(HookBase):
    """Hook to monitor memory usage during training with tqdm-style updates"""
    
    def __init__(self, log_interval=100, cleanup_interval=1000, verbose_cleanup=False):
        self.log_interval = log_interval
        self.cleanup_interval = cleanup_interval
        self.verbose_cleanup = verbose_cleanup
        self.logger = logging.getLogger(__name__)
        self.last_memory_log = 0
    
    def before_step(self):
        iteration = self.trainer.iter
        
        # Log memory usage periodically with emojis
        if iteration % self.log_interval == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            self.logger.info(f"🔄 Iteration {iteration} | 💾 GPU: {allocated:.1f}GB")
        
        # Periodic cleanup
        if iteration % self.cleanup_interval == 0 and iteration > 0:
            self.logger.info(f"🧹 Iteration {iteration}: Performing periodic cleanup")
            comprehensive_memory_cleanup(verbose=self.verbose_cleanup)
    
    def after_step(self):
        pass

# ========================================================================================
# ENHANCED DATASET FUNCTIONS WITH TQDM
# ========================================================================================

def create_subset_annotations(annotation_file, output_file, subset_ratio=0.001):
    """Create a subset of COCO annotations file with specified ratio of data"""
    
    with memory_managed_operation(f"Creating subset annotations ({subset_ratio*100}%)", cleanup_after=True):
        print(f"🎯 Creating subset annotations with {subset_ratio*100}% of original data...")
        
        # Load original annotations with progress
        with tqdm(desc="📖 Loading annotations", bar_format='{desc}', leave=False) as pbar:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            pbar.set_description("✅ Annotations loaded!")
        
        # Get original counts
        original_images = len(data['images'])
        original_annotations = len(data['annotations'])
        
        # Calculate subset size (minimum 1 image)
        subset_size = max(1, int(original_images * subset_ratio))
        
        print(f"📊 Original dataset: {original_images} images, {original_annotations} annotations")
        print(f"🎯 Subset dataset: {subset_size} images (target)")
        
        # Randomly sample images with progress
        random.seed(42)  # For reproducibility
        with tqdm(desc="🎲 Sampling images", total=subset_size, leave=False) as pbar:
            sampled_images = random.sample(data['images'], subset_size)
            pbar.update(subset_size)
        
        sampled_image_ids = {img['id'] for img in sampled_images}
        
        # Filter annotations for sampled images with progress
        sampled_annotations = []
        with tqdm(data['annotations'], desc="🔍 Filtering annotations", leave=False) as pbar:
            for ann in pbar:
                if ann['image_id'] in sampled_image_ids:
                    sampled_annotations.append(ann)
        
        # Create subset data
        subset_data = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data['categories'],  # Keep all categories
            'images': sampled_images,
            'annotations': sampled_annotations
        }
        
        # Add info about subset
        subset_data['info']['description'] = f"Subset ({subset_ratio*100}%) of original dataset for testing"
        
        # Save subset annotations with progress
        with tqdm(desc="💾 Saving subset", bar_format='{desc}', leave=False) as pbar:
            with open(output_file, 'w') as f:
                json.dump(subset_data, f)
            pbar.set_description("✅ Subset saved!")
        
        print(f"✅ Subset created: {len(sampled_images)} images, {len(sampled_annotations)} annotations")
        print(f"📁 Subset saved to: {output_file}")
        
        return output_file

def register_datasets(dataset_path, images_path, use_test_subset=False):
    """Register COCO format datasets with memory management and progress"""
    
    with memory_managed_operation("Dataset Registration", cleanup_after=True):
        # Register training dataset
        train_json = os.path.join(dataset_path, "annotation_train.json")
        if not os.path.exists(train_json):
            raise FileNotFoundError(f"❌ Training annotation file not found: {train_json}")
        
        # Create subset if test mode is enabled
        if use_test_subset:
            # Create temporary directory for subset annotations 
            temp_dir = tempfile.mkdtemp(prefix="eva_test_subset_")
            
            # Create subset training annotations
            subset_train_json = os.path.join(temp_dir, "annotation_train_subset.json")
            create_subset_annotations(train_json, subset_train_json, subset_ratio=0.001)
            
            # Register subset training dataset
            register_coco_instances("custom_train", {}, subset_train_json, images_path)
            print(f"✅ Registered TEST training dataset (0.1%): {subset_train_json}")
            
            # Handle test dataset
            test_json = os.path.join(dataset_path, "annotation_test.json")
            if os.path.exists(test_json):
                # Create subset test annotations
                subset_test_json = os.path.join(temp_dir, "annotation_test_subset.json")
                create_subset_annotations(test_json, subset_test_json, subset_ratio=0.001)
                
                register_coco_instances("custom_test", {}, subset_test_json, images_path)
                print(f"✅ Registered TEST test dataset (0.1%): {subset_test_json}")
                return "custom_train", "custom_test"
            else:
                print(f"⚠️  Warning: Test annotation file not found: {test_json}")
                return "custom_train", None
                
        else:
            # Regular full dataset registration
            register_coco_instances("custom_train", {}, train_json, images_path)
            print(f"✅ Registered FULL training dataset: {train_json}")
            
            # Register test dataset
            test_json = os.path.join(dataset_path, "annotation_test.json")
            if os.path.exists(test_json):
                register_coco_instances("custom_test", {}, test_json, images_path)
                print(f"✅ Registered FULL test dataset: {test_json}")
                return "custom_train", "custom_test"
            else:
                print(f"⚠️  Warning: Test annotation file not found: {test_json}")
                return "custom_train", None

# ========================================================================================
# PARSE ARGS AND OTHER FUNCTIONS (keeping them the same)
# ========================================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="EVA-02 Fine-tuning for Detection")
    
    # Model selection
    parser.add_argument("--model_name", default="eva-02-02", choices=["eva-02-01", "eva-02-02"],
                        help="Either eva-02-01 or eva-02-02")
    
    # Dataset paths
    parser.add_argument("--dataset_path", default="/mnt/e/Desktop/GLaMM/detection_dataset_coco",
                        help="Path to dataset directory containing annotation_train.json and annotation_test.json")
    parser.add_argument("--images_path", default="/mnt/e/Desktop/GLaMM/detection_dataset_coco",
                        help="Path to images directory")
    
    # Training parameters
    parser.add_argument("--output_dir", default="./eva_2_FT_2",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--num_classes", type=int, default=30,
                        help="Number of classes in your dataset")
    parser.add_argument("--max_iter", type=int, default=300000,
                        help="Maximum number of training iterations")
    parser.add_argument("--eval_period", type=int, default=10000,
                        help="Evaluation period")
    parser.add_argument("--checkpoint_period", type=int, default=10000,
                        help="Checkpoint saving period")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--warmup_iters", type=int, default=1000,
                        help="Warmup iterations")
    
    # Worker configuration
    parser.add_argument("--train_num_workers", type=int, default=2,
                        help="Number of workers for training dataloader")
    parser.add_argument("--test_num_workers", type=int, default=10,
                        help="Number of workers for test dataloader")
    
    # Memory management options
    parser.add_argument("--memory_cleanup_interval", type=int, default=100,
                        help="Interval for periodic memory cleanup during training")
    parser.add_argument("--memory_log_interval", type=int, default=10000000,
                        help="Interval for memory usage logging")
    parser.add_argument("--aggressive_cleanup", action="store_true",
                        help="Enable more aggressive memory cleanup (slower but more memory-efficient)")
    
    # Evaluation options
    parser.add_argument("--eval_before_training", action="store_true", default=True,
                        help="Perform evaluation before training to establish baseline")
    parser.add_argument("--skip_initial_eval", action="store_true",
                        help="Skip initial evaluation (overrides --eval_before_training)")
    parser.add_argument("--eval_only", action="store_true",
                        help="Only perform evaluation, do not train")
    
    # Test run option
    parser.add_argument("--test", action="store_true",
                        help="Run on 0.1% of the dataset as a test run")
    
    # Resume training
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--resume_checkpoint", type=str, default="",
                        help="Specific checkpoint to resume from")
    
    # Batch size parameters
    parser.add_argument("--batch_size_per_gpu", type=int, default=1,
                        help="Training batch size per GPU")
    parser.add_argument("--eval_batch_size_per_gpu", type=int, default=6,
                        help="Evaluation batch size per GPU (if None, will auto-calculate based on training batch size)")
    parser.add_argument("--auto_eval_batch_size", action="store_true", default=True,
                        help="Automatically determine optimal evaluation batch size")
    
    # DDP Related parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    args = parser.parse_args()
    
    # Handle conflicting flags
    if args.skip_initial_eval:
        args.eval_before_training = False
    
    # Adjust cleanup intervals for aggressive mode
    if args.aggressive_cleanup:
        args.memory_cleanup_interval = min(500, args.memory_cleanup_interval)
        args.memory_log_interval = min(50, args.memory_log_interval)
    
    # Auto-calculate evaluation batch size if not specified
    if args.eval_batch_size_per_gpu is None and args.auto_eval_batch_size:
        # Use the optimal calculation function
        train_batch = int(args.batch_size_per_gpu)
        
        # Try to get available GPU memory for optimization
        available_memory = None
        if torch.cuda.is_available():
            try:
                device = torch.cuda.current_device()
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
                # Assume we can use about 80% of total memory for evaluation
                available_memory = total_memory * 0.8
            except:
                available_memory = None
        
        args.eval_batch_size_per_gpu = get_optimal_eval_batch_size(train_batch, available_memory)
        
        print(f"🎯 Auto-calculated evaluation batch size: {args.eval_batch_size_per_gpu} (training: {train_batch})")
        if available_memory:
            print(f"💾 Based on available GPU memory: {available_memory:.1f}GB")
    elif args.eval_batch_size_per_gpu is None:
        # Default to same as training if auto-calculation is disabled
        args.eval_batch_size_per_gpu = int(args.batch_size_per_gpu)
    
    # Ensure eval batch size is at least as large as training batch size
    if args.eval_batch_size_per_gpu < args.batch_size_per_gpu:
        print(f"⚠️  Warning: Evaluation batch size ({args.eval_batch_size_per_gpu}) is smaller than training batch size ({args.batch_size_per_gpu})")
        print(f"This is unusual and may not be optimal. Consider using a larger evaluation batch size.")
    
    return args

def setup_config(args, train_dataset_name, test_dataset_name):
    """Setup configuration for training with memory considerations"""
    
    with memory_managed_operation("Configuration Setup", cleanup_after=True):
        # Load base config
        if args.model_name == "eva-02-01":
            config_path = eva02_L_lvis_sys_o365_config_path
            checkpoint_path = eva02_L_lvis_sys_o365_ckpt_path
        else:
            config_path = eva02_L_lvis_sys_config_path
            checkpoint_path = eva02_L_lvis_sys_ckpt_path
        
        cfg = LazyConfig.load(config_path)
        
        # CRITICAL FIX: Configure for detection-only training
        # 1. Disable mask training in the model
        cfg.model.roi_heads.num_classes = args.num_classes
        
        # 2. Remove mask head completely from the config
        if hasattr(cfg.model.roi_heads, 'mask_head'):
            delattr(cfg.model.roi_heads, 'mask_head')
        if hasattr(cfg.model.roi_heads, 'mask_in_features'):
            delattr(cfg.model.roi_heads, 'mask_in_features')
        if hasattr(cfg.model.roi_heads, 'mask_pooler'):
            delattr(cfg.model.roi_heads, 'mask_pooler')
        
        # 3. Disable federated loss for custom dataset
        # For cascade models, we need to update all box predictors
        if hasattr(cfg.model.roi_heads, 'box_predictors'):
            # This is a cascade model with multiple predictors
            for i in range(len(cfg.model.roi_heads.box_predictors)):
                cfg.model.roi_heads.box_predictors[i].use_fed_loss = False
                cfg.model.roi_heads.box_predictors[i].use_sigmoid_ce = False
                cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.05
                cfg.model.roi_heads.box_predictors[i].test_topk_per_image = 1000
                # Remove the fed loss weight function
                if hasattr(cfg.model.roi_heads.box_predictors[i], 'get_fed_loss_cls_weights'):
                    delattr(cfg.model.roi_heads.box_predictors[i], 'get_fed_loss_cls_weights')
        elif hasattr(cfg.model.roi_heads, 'box_predictor'):
            # This is a standard model with single predictor
            cfg.model.roi_heads.box_predictor.use_fed_loss = False
            cfg.model.roi_heads.box_predictor.use_sigmoid_ce = False
            cfg.model.roi_heads.box_predictor.test_score_thresh = 0.05
            cfg.model.roi_heads.box_predictor.test_topk_per_image = 1000
            # Remove the fed loss weight function
            if hasattr(cfg.model.roi_heads.box_predictor, 'get_fed_loss_cls_weights'):
                delattr(cfg.model.roi_heads.box_predictor, 'get_fed_loss_cls_weights')
        
        # 4. CRITICAL FIX: Configure dataset names properly
        cfg.dataloader.train.dataset.names = train_dataset_name
        if test_dataset_name:
            cfg.dataloader.test.dataset.names = test_dataset_name
            
        # Adjust number of workers based on memory constraints and user input
        cfg.dataloader.train.num_workers = args.train_num_workers
        cfg.dataloader.test.num_workers = args.test_num_workers

        # Configure the dataset mapper for detection-only (no masks)
        cfg.dataloader.train.mapper = L(DatasetMapper)(
            is_train=True,
            augmentations=[
                L(T.ResizeShortestEdge)(
                    short_edge_length=[640, 672, 704, 736, 768, 800],
                    sample_style="choice",
                    max_size=1333,
                ),
                L(T.RandomFlip)(horizontal=True),
                L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.2),
                L(T.RandomContrast)(intensity_min=0.8, intensity_max=1.2),
                L(T.RandomSaturation)(intensity_min=0.8, intensity_max=1.2),
            ],
            image_format="BGR",
            use_instance_mask=False,  # CRITICAL: Disable mask usage
            use_keypoint=False,
            recompute_boxes=False,
        )
        
        cfg.dataloader.test.mapper = L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
            ],
            image_format="BGR",
            use_instance_mask=False,  # CRITICAL: Disable mask usage
            use_keypoint=False,
        )
        
        # Training parameters
        cfg.train.max_iter = args.max_iter
        
        # Adjust training parameters for test mode
        if args.test:
            # Reduce iterations for test mode
            cfg.train.max_iter = min(100, args.max_iter)  # Max 100 iterations for test
            args.eval_period = min(50, args.eval_period)  # Evaluate every 50 iterations
            args.checkpoint_period = min(50, args.checkpoint_period)  # Save every 50 iterations
            print(f"🧪 TEST MODE: Reduced max_iter to {cfg.train.max_iter}")
        
        # Fix: Use correct optimizer parameter path
        if hasattr(cfg, 'optimizer') and hasattr(cfg.optimizer, 'lr'):
            cfg.optimizer.lr = args.learning_rate
        elif hasattr(cfg, 'optimizer') and hasattr(cfg.optimizer, 'base_lr'):
            cfg.optimizer.base_lr = args.learning_rate
            
        # Warmup iterations - might be in lr_multiplier
        if hasattr(cfg, 'lr_multiplier') and hasattr(cfg.lr_multiplier, 'warmup_length'):
            cfg.lr_multiplier.warmup_length = args.warmup_iters / cfg.train.max_iter
        
        # Checkpoint period
        if hasattr(cfg.train, 'checkpointer'):
            cfg.train.checkpointer.period = args.checkpoint_period
        
        # FIXED: Set batch sizes differently for train and test
        cfg.dataloader.train.total_batch_size = args.batch_size_per_gpu * args.world_size
        # Remove total_batch_size from test config - use samples_per_gpu instead
        if hasattr(cfg.dataloader.test, 'total_batch_size'):
            delattr(cfg.dataloader.test, 'total_batch_size')
        # Set samples per GPU for test dataloader
        cfg.dataloader.test.samples_per_gpu = args.eval_batch_size_per_gpu
        
        # Output directory
        if args.test:
            # Create separate output directory for test runs
            test_output_dir = os.path.join(args.output_dir, "test_run")
            os.makedirs(test_output_dir, exist_ok=True)
            cfg.train.output_dir = test_output_dir
            print(f"🧪 TEST MODE: Output directory set to {test_output_dir}")
        else:
            cfg.train.output_dir = args.output_dir
        
        # Checkpoint path for initialization
        cfg.train.init_checkpoint = checkpoint_path
        
        # Evaluation
        cfg.train.eval_period = args.eval_period
        
        # Update evaluator for custom dataset - ONLY for bbox evaluation
        if hasattr(cfg, 'dataloader') and hasattr(cfg.dataloader, 'evaluator'):
            cfg.dataloader.evaluator = L(COCOEvaluator)(
                dataset_name=test_dataset_name if test_dataset_name else train_dataset_name,
                tasks=["bbox"],  # CRITICAL: Only evaluate bbox, not segm
                output_dir=cfg.train.output_dir,
            )
        
        return cfg

def perform_initial_evaluation(cfg, model_weights_path, test_dataset_name, logger, args):
    """Perform initial evaluation before training with enhanced memory management and progress bars"""
    
    with memory_managed_operation("Initial Evaluation", cleanup_after=True):
        logger.info("=" * 80)
        logger.info("🎯 PERFORMING INITIAL EVALUATION (PRE-TRAINING BASELINE)")
        logger.info("=" * 80)
        
        # Create model for evaluation with progress indication
        with tqdm(desc="🏗️  Creating model", bar_format='{desc}', leave=False) as pbar:
            model = instantiate(cfg.model)
            model.to(cfg.train.device)
            model.eval()
            pbar.set_description("✅ Model created")
        
        # Load pre-trained weights with progress indication
        with tqdm(desc="📥 Loading weights", bar_format='{desc}', leave=False) as pbar:
            checkpointer = DetectionCheckpointer(model, cfg.train.output_dir)
            checkpointer.load(model_weights_path)
            pbar.set_description("✅ Weights loaded")
        
        logger.info(f"✅ Loaded pre-trained weights from: {model_weights_path}")
        logger.info(f"🎯 Evaluating on dataset: {test_dataset_name}")
        logger.info(f"💾 {get_gpu_memory_info()}")
        
        # Verify dataset exists and has data
        dataset_dicts = DatasetCatalog.get(test_dataset_name)
        logger.info(f"📊 Dataset verification: {len(dataset_dicts)} samples found")
        
        if len(dataset_dicts) == 0:
            logger.error("❌ Dataset is empty! Cannot perform evaluation.")
            return {}
        
        # Create data loader for evaluation
        with tqdm(desc="🔄 Creating data loader", bar_format='{desc}', leave=False) as pbar:
            test_loader = build_detection_test_loader(
                dataset=dataset_dicts,
                mapper=instantiate(cfg.dataloader.test.mapper),
                num_workers=cfg.dataloader.test.num_workers,
                batch_size=args.eval_batch_size_per_gpu
            )
            pbar.set_description("✅ Data loader ready")
        
        logger.info(f"🎯 Test loader created with batch size: {args.eval_batch_size_per_gpu}")
        logger.info(f"📊 Test dataset size: {len(dataset_dicts)} samples")
        
        # Quick test: Process one batch to check if everything works
        with tqdm(desc="🧪 Testing model", bar_format='{desc}', leave=False) as pbar:
            try:
                test_iter = iter(test_loader)
                sample_batch = next(test_iter)
                
                with torch.no_grad():
                    sample_outputs = model(sample_batch)
                
                pbar.set_description("✅ Model test passed")
                del sample_outputs, sample_batch, test_iter
                torch.cuda.empty_cache()
            except Exception as e:
                pbar.set_description("❌ Model test failed")
                logger.error(f"Model test failed: {e}")
                return {}
        
        # Create evaluator
        logger.info("🧮 Creating evaluator...")
        evaluator = COCOEvaluator(
            test_dataset_name, 
            tasks=["bbox"], 
            output_dir=os.path.join(cfg.train.output_dir, "initial_eval")
        )
        
        logger.info("🚀 Starting full evaluation...")
        
        # Run memory-efficient evaluation
        start_time = time.time()
        results = memory_efficient_evaluation(model, test_loader, evaluator, logger, cleanup_interval=20)
        eval_time = time.time() - start_time
        
        logger.info(f"✅ Initial evaluation completed in {eval_time:.2f} seconds")
        logger.info("📊 Initial evaluation results (PRE-TRAINING BASELINE):")
        
        # Print results in a more readable format
        if results:
            print_csv_format(results)
        else:
            logger.warning("⚠️  No evaluation results returned!")
        
        # Save initial evaluation results
        initial_eval_dir = os.path.join(cfg.train.output_dir, "initial_eval")
        os.makedirs(initial_eval_dir, exist_ok=True)
        
        # Save results as JSON
        results_file = os.path.join(initial_eval_dir, "initial_evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Initial evaluation results saved to: {results_file}")
        
        # Print key metrics summary
        if results and 'bbox' in results:
            bbox_results = results['bbox']
            logger.info("=" * 60)
            logger.info("🎯 INITIAL BASELINE SUMMARY:")
            logger.info(f"  📊 Bbox AP     : {bbox_results.get('AP', 'N/A'):.3f}")
            logger.info(f"  🎯 Bbox AP50   : {bbox_results.get('AP50', 'N/A'):.3f}")
            logger.info(f"  🔍 Bbox AP75   : {bbox_results.get('AP75', 'N/A'):.3f}")
            logger.info("=" * 60)
        else:
            logger.warning("⚠️  No bbox results found in evaluation!")
        
        # Cleanup evaluation components
        safe_model_deletion(model, "evaluation_model")
        del test_loader
        del evaluator
        
        logger.info("✅ Initial evaluation completed. Ready to start training...")
        return results

class EVA02Trainer(DefaultTrainer):
    """Custom trainer for EVA-02 fine-tuning with memory management"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """Build COCO evaluator for detection only"""
        return COCOEvaluator(dataset_name, tasks=["bbox"], output_dir=cfg.train.output_dir)
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build training data loader"""
        return instantiate(cfg.dataloader.train)
    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Build test data loader - FIXED"""
        # Use the standard Detectron2 function for building test loader
        return build_detection_test_loader(
            dataset=DatasetCatalog.get(dataset_name),
            mapper=instantiate(cfg.dataloader.test.mapper),
            num_workers=cfg.dataloader.test.num_workers,
            batch_size=getattr(cfg.dataloader.test, 'samples_per_gpu', 2)
        )
    
    def build_hooks(self):
        """Build training hooks with memory management"""
        hooks_list = super().build_hooks()
        
        # Add evaluation hook if test dataset is available
        if hasattr(self.cfg.dataloader.test, 'dataset') and hasattr(self.cfg.dataloader.test.dataset, 'names'):
            hooks_list.insert(
                -1,
                hooks.EvalHook(
                    self.cfg.train.eval_period,
                    lambda: self.test(self.cfg, self.model, evaluators=[
                        self.build_evaluator(self.cfg, self.cfg.dataloader.test.dataset.names)
                    ])
                )
            )
        
        return hooks_list

def train_model(args):
    """Main training function with enhanced memory management and progress indicators"""
    
    # Initialize start_time early to prevent UnboundLocalError
    start_time = time.time()
    
    # Setup logging
    output_dir = os.path.join(args.output_dir, "test_run") if args.test else args.output_dir
    setup_logger(output=output_dir, distributed_rank=args.local_rank)
    logger = logging.getLogger(__name__)
    
    # Set logger level to INFO to show evaluation progress
    logger.setLevel(logging.INFO)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Log memory management settings with estimates
    logger.info("=" * 60)
    logger.info("🔧 MEMORY AND BATCH SIZE CONFIGURATION:")
    logger.info(f"  🧹 Memory cleanup interval: {args.memory_cleanup_interval} iterations")
    logger.info(f"  📝 Memory log interval: {args.memory_log_interval} iterations")
    logger.info(f"  🚀 Aggressive cleanup: {args.aggressive_cleanup}")
    logger.info(f"  🎯 Training batch size per GPU: {args.batch_size_per_gpu}")
    logger.info(f"  📊 Evaluation batch size per GPU: {args.eval_batch_size_per_gpu}")
    logger.info(f"  👥 Training workers: {args.train_num_workers}")
    logger.info(f"  🧪 Test workers: {args.test_num_workers}")
    batch_ratio = args.eval_batch_size_per_gpu / args.batch_size_per_gpu
    logger.info(f"  📈 Eval/Train batch ratio: {batch_ratio:.1f}x")
    
    # Memory usage estimates
    memory_est = estimate_memory_usage(args.batch_size_per_gpu, args.eval_batch_size_per_gpu, args.model_name)
    logger.info("  💾 Estimated Memory Usage:")
    logger.info(f"    🏗️  Model base: {memory_est['base_model']:.1f}GB")
    logger.info(f"    🎯 Training total: {memory_est['training_total']:.1f}GB")
    logger.info(f"    📊 Evaluation total: {memory_est['evaluation_total']:.1f}GB")
    logger.info(f"    💰 Memory savings during eval: {memory_est['training_total'] - memory_est['evaluation_total']:.1f}GB")
    logger.info("=" * 60)
    
    # Print mode status
    if args.test:
        logger.info("=" * 60)
        logger.info("🧪 RUNNING IN TEST MODE (0.1% of dataset)")
        logger.info("=" * 60)
    
    if args.eval_only:
        logger.info("=" * 60)
        logger.info("📊 RUNNING IN EVALUATION-ONLY MODE")
        logger.info("=" * 60)
    
    # Initial memory status
    logger.info(f"🚀 Initial memory status: {get_gpu_memory_info()}")
    logger.info(f"💻 Initial memory status: {get_cpu_memory_info()}")
    
    # Register datasets
    train_dataset, test_dataset = register_datasets(args.dataset_path, args.images_path, use_test_subset=args.test)
    
    # Setup configuration
    cfg = setup_config(args, train_dataset, test_dataset)
    
    # Debug: Print some key configuration values
    logger.info(f"🎯 Number of classes: {cfg.model.roi_heads.num_classes}")
    logger.info(f"🏋️  Training dataset: {cfg.dataloader.train.dataset.names}")
    if test_dataset:
        logger.info(f"🧪 Test dataset: {cfg.dataloader.test.dataset.names}")
    logger.info(f"🔄 Max iterations: {cfg.train.max_iter}")
    logger.info(f"📦 Training batch size: {cfg.dataloader.train.total_batch_size} (per GPU: {args.batch_size_per_gpu})")
    logger.info(f"📊 Evaluation batch size per GPU: {getattr(cfg.dataloader.test, 'samples_per_gpu', args.eval_batch_size_per_gpu)}")
    logger.info(f"🎯 Detection-only mode: masks disabled")
    
    if args.test:
        logger.info(f"🧪 TEST MODE: Running on 0.1% of dataset")
        logger.info(f"🧪 TEST MODE: Max iterations reduced to {cfg.train.max_iter}")
    
    # Check if federated loss is disabled
    if hasattr(cfg.model.roi_heads, 'box_predictors'):
        logger.info(f"✅ Federated loss disabled for cascade model")
    elif hasattr(cfg.model.roi_heads, 'box_predictor'):
        logger.info(f"✅ Federated loss disabled for standard model")
    
    # PERFORM INITIAL EVALUATION BEFORE TRAINING
    initial_results = None
    if args.eval_before_training and test_dataset and not args.resume:
        try:
            initial_results = perform_initial_evaluation(
                cfg, 
                cfg.train.init_checkpoint, 
                test_dataset, 
                logger,
                args  # Pass args parameter
            )
        except Exception as e:
            logger.warning(f"⚠️  Initial evaluation failed: {e}")
            logger.warning("🔄 Continuing with training...")
    elif args.eval_only and test_dataset:
        # Only perform evaluation and exit
        try:
            initial_results = perform_initial_evaluation(
                cfg, 
                cfg.train.init_checkpoint, 
                test_dataset, 
                logger,
                args  # Pass args parameter
            )
            logger.info("✅ Evaluation-only mode completed successfully.")
            return initial_results
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise
    elif args.resume:
        logger.info("🔄 Resuming training - skipping initial evaluation")
    elif not test_dataset:
        logger.warning("⚠️  No test dataset available - skipping initial evaluation")
    else:
        logger.info("⏭️  Initial evaluation disabled")
    
    # If eval_only mode, we're done
    if args.eval_only:
        return initial_results
    
    # SETUP MODEL FOR TRAINING WITH MEMORY MANAGEMENT
    model = None
    trainer = None
    
    try:
        with memory_managed_operation("Model Setup", cleanup_after=False):
            # Create model with progress indication
            with tqdm(desc="🏗️  Creating model", bar_format='{desc}', leave=False) as pbar:
                model = instantiate(cfg.model)
                model.to(cfg.train.device)
                pbar.set_description("✅ Model created")
            
            logger.info(f"🎯 Model configured for detection-only training")
            logger.info(f"💾 Memory after model creation: {get_gpu_memory_info()}")
            
            # Setup optimizer
            cfg.optimizer.params.model = model
            optimizer = instantiate(cfg.optimizer)
            
            # Setup data loader
            train_loader = instantiate(cfg.dataloader.train)
            
            # Setup checkpointer
            checkpointer = DetectionCheckpointer(
                model, cfg.train.output_dir, optimizer=optimizer
            )
            
            # Resume or load checkpoint with progress indication
            with tqdm(desc="📥 Loading checkpoint", bar_format='{desc}', leave=False) as pbar:
                if args.resume:
                    if args.resume_checkpoint:
                        checkpointer.resume_or_load(args.resume_checkpoint, resume=True)
                    else:
                        checkpointer.resume_or_load(None, resume=True)  # <--- FIXED
                else:
                    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=False)
                pbar.set_description("✅ Checkpoint loaded")

            
            logger.info(f"💾 Memory after checkpoint loading: {get_gpu_memory_info()}")
            
            # Setup trainer
            from detectron2.engine import SimpleTrainer
            trainer = SimpleTrainer(model, train_loader, optimizer)
            
            # Create memory monitoring hook
            memory_hook = MemoryMonitoringHook(
                log_interval=args.memory_log_interval,
                cleanup_interval=args.memory_cleanup_interval,
                verbose_cleanup=args.aggressive_cleanup
            )
            
            # Training hooks with memory monitoring - FIXED: Pass individual hooks
            trainer.register_hooks([
                hooks.IterationTimer(),
                hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
                hooks.PeriodicCheckpointer(checkpointer, period=args.checkpoint_period),
                hooks.PeriodicWriter(
                    [CommonMetricPrinter(cfg.train.max_iter), JSONWriter(os.path.join(cfg.train.output_dir, "metrics.json"))],
                    period=20,
                ),
            ])
            
            # Register memory hook separately
            trainer.register_hooks([memory_hook])
            
            # Add evaluation hook if test dataset exists
            if test_dataset:
                evaluator = COCOEvaluator(test_dataset, tasks=["bbox"], output_dir=cfg.train.output_dir)
                
                # Create evaluation function that works with the new test loader
                def eval_function():
                    test_loader = build_detection_test_loader(
                        dataset=DatasetCatalog.get(test_dataset),
                        mapper=instantiate(cfg.dataloader.test.mapper),
                        num_workers=cfg.dataloader.test.num_workers,
                        batch_size=getattr(cfg.dataloader.test, 'samples_per_gpu', 2)
                    )
                    return inference_on_dataset(model, test_loader, evaluator)
                
                trainer.register_hooks([
                    hooks.EvalHook(args.eval_period, eval_function)
                ])
        
        # Start training
        mode_str = "🧪 TEST detection-only training (0.1% dataset)" if args.test else "🚀 FULL detection-only training"
        logger.info(f"Starting {mode_str}...")
        
        if initial_results and 'bbox' in initial_results:
            logger.info(f"🎯 Training will start from baseline AP: {initial_results['bbox'].get('AP', 'N/A'):.3f}")
        
        logger.info(f"💾 Memory before training: {get_gpu_memory_info()}")
        start_time = time.time()  # Reset start time just before training
        
        # Training with enhanced error handling
        print("\n" + "="*80)
        print("🚀 STARTING TRAINING - Progress will be shown by Detectron2 trainer")
        print("="*80)
        trainer.train(0, cfg.train.max_iter)
        
    except KeyboardInterrupt:
        logger.info("⏹️  Training interrupted by user")
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error(f"💾 Memory at error: {get_gpu_memory_info()}")
        comprehensive_memory_cleanup(verbose=True)
        raise
    finally:
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
        # Final evaluation if test dataset is available
        if test_dataset and model is not None:
            try:
                logger.info("🏁 Running final evaluation with optimized batch size...")
                model.eval()
                evaluator = COCOEvaluator(test_dataset, tasks=["bbox"], output_dir=cfg.train.output_dir)
                
                # Create test loader for final evaluation
                logger.info("🔄 Creating final evaluation test loader...")
                test_loader = build_detection_test_loader(
                    dataset=DatasetCatalog.get(test_dataset),
                    mapper=instantiate(cfg.dataloader.test.mapper),
                    num_workers=cfg.dataloader.test.num_workers,
                    batch_size=args.eval_batch_size_per_gpu
                )
                
                logger.info(f"📊 Final evaluation batch size: {args.eval_batch_size_per_gpu}")
                final_results = memory_efficient_evaluation(model, test_loader, evaluator, logger, cleanup_interval=30)
                
                logger.info("🏆 Final evaluation results:")
                if final_results:
                    print_csv_format(final_results)
                else:
                    logger.warning("⚠️  No final evaluation results returned!")
                
                # Cleanup evaluation components
                del evaluator
                del test_loader
                
                # Compare with initial results if available
                if initial_results and final_results and 'bbox' in initial_results and 'bbox' in final_results:
                    initial_ap = initial_results['bbox'].get('AP', 0.0)
                    final_ap = final_results['bbox'].get('AP', 0.0)
                    improvement = final_ap - initial_ap
                    
                    logger.info("=" * 80)
                    logger.info("🎯 TRAINING RESULTS SUMMARY:")
                    logger.info(f"  🎯 Initial AP (baseline): {initial_ap:.3f}")
                    logger.info(f"  🏆 Final AP (trained)   : {final_ap:.3f}")
                    if improvement > 0:
                        logger.info(f"  📈 Improvement          : +{improvement:.3f} 🎉")
                    else:
                        logger.info(f"  📉 Change               : {improvement:+.3f}")
                    logger.info("=" * 80)
                elif final_results and 'bbox' in final_results:
                    final_ap = final_results['bbox'].get('AP', 0.0)
                    logger.info("=" * 80)
                    logger.info("🎯 TRAINING RESULTS SUMMARY:")
                    logger.info(f"  🏆 Final AP (trained): {final_ap:.3f}")
                    logger.info("=" * 80)
            except Exception as e:
                logger.warning(f"⚠️  Final evaluation failed: {e}")
                import traceback
                logger.warning(f"Traceback: {traceback.format_exc()}")
        
        if args.test:
            logger.info("=" * 60)
            logger.info("🎉 TEST RUN COMPLETED SUCCESSFULLY")
            logger.info("✅ You can now run the full training without --test flag")
            logger.info("=" * 60)
        
        # COMPREHENSIVE CLEANUP
        logger.info("🧹 Performing comprehensive cleanup...")
        
        # Safe deletion of training components
        if trainer is not None:
            del trainer
        
        if model is not None:
            safe_model_deletion(model, "training_model")
        
        # Final memory cleanup
        comprehensive_memory_cleanup(verbose=True)
        
        logger.info("✅ Training and cleanup completed successfully")

def main():
    """Main function"""
    args = parse_args()
    
    # Initialize distributed training
    init_distributed_mode(args)
    
    # Set device
    torch.cuda.set_device(args.local_rank)
    
    # Log initial system information
    print(f"🚀 Starting EVA-02 fine-tuning with enhanced memory management")
    print(f"💾 Initial GPU memory: {get_gpu_memory_info()}")
    print(f"💻 Initial CPU memory: {get_cpu_memory_info()}")
    
    # Train model
    train_model(args)

def distributed_main():
    """Entry point for distributed training"""
    args = parse_args()
    
    print(f"⚙️  Command Line Args: {args}")
    
    # Launch distributed training
    launch(
        main,
        args.world_size,
        num_machines=1,
        machine_rank=0,
        dist_url=args.dist_url,
        args=(args,),
    )

if __name__ == "__main__":
    main()

# EVA-02 Fine-tuning Script for Detection Only (BBOX) - FIXED CHECKPOINT VERSION
# Fixed version with proper checkpoint management for complete resuming

# import argparse
# import json
# import os
# import torch
# import torch.nn.parallel
# import detectron2.data.transforms as T
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import LazyConfig, instantiate
# from detectron2.data import build_detection_train_loader, build_detection_test_loader
# from detectron2.data import DatasetCatalog, MetadataCatalog
# from detectron2.data.datasets import register_coco_instances
# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset, print_csv_format
# from detectron2.utils.logger import setup_logger
# from detectron2.utils.events import CommonMetricPrinter, JSONWriter
# from detectron2.solver import build_lr_scheduler, build_optimizer
# from detectron2.utils.events import EventStorage
# from detectron2.data import DatasetMapper
# from detectron2.config import LazyCall as L
# from detectron2.engine.hooks import HookBase
# import logging
# import time
# from tqdm import tqdm
# import gc
# import random
# import tempfile
# import psutil
# from contextlib import contextmanager

# # Import DDP utilities from your original code
# from ddp import *

# # Config paths - same as inference code
# eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
#                                      "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
# eva02_L_lvis_sys_config_path = ("projects/ViTDet/configs/eva2_mim_to_lvis/"
#                                 "eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py")

# # Checkpoint paths - same as inference code
# eva02_L_lvis_sys_o365_ckpt_path = ("eva02_L_lvis_sys_o365.pth")
# eva02_L_lvis_sys_ckpt_path = ("eva02_L_lvis_sys.pth")

# # ========================================================================================
# # ENHANCED MEMORY MANAGEMENT FUNCTIONS WITH TQDM
# # ========================================================================================

# def get_gpu_memory_info():
#     """Get current GPU memory usage information"""
#     if not torch.cuda.is_available():
#         return "CUDA not available"
    
#     device = torch.cuda.current_device()
#     allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
#     cached = torch.cuda.memory_reserved(device) / 1024**3      # GB
    
#     try:
#         total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
#         return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB"
#     except:
#         return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"

# def get_cpu_memory_info():
#     """Get current CPU memory usage information"""
#     process = psutil.Process(os.getpid())
#     cpu_memory = process.memory_info().rss / 1024**3  # GB
#     system_memory = psutil.virtual_memory()
#     return f"CPU Memory - Process: {cpu_memory:.2f}GB, System: {system_memory.percent:.1f}% used"

# def comprehensive_memory_cleanup(verbose=True):
#     """Comprehensive memory cleanup function"""
#     logger = logging.getLogger(__name__)
    
#     if verbose:
#         logger.info(f"Before cleanup: {get_gpu_memory_info()}")
#         logger.info(f"Before cleanup: {get_cpu_memory_info()}")
    
#     # Force garbage collection
#     gc.collect()
    
#     # Clear CUDA cache if available
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()
        
#         # Additional cleanup for persistent memory
#         try:
#             torch.cuda.ipc_collect()
#         except:
#             pass  # Not available in all PyTorch versions
    
#     # Additional Python garbage collection
#     gc.collect()
    
#     if verbose:
#         logger.info(f"After cleanup: {get_gpu_memory_info()}")
#         logger.info(f"After cleanup: {get_cpu_memory_info()}")

# def periodic_memory_cleanup(iteration, cleanup_interval=1000, verbose=False):
#     """Perform memory cleanup at regular intervals during training"""
#     if iteration % cleanup_interval == 0:
#         logger = logging.getLogger(__name__)
#         if verbose:
#             logger.info(f"Iteration {iteration}: Performing periodic memory cleanup")
#         comprehensive_memory_cleanup(verbose=verbose)

# @contextmanager
# def memory_managed_operation(operation_name="Operation", cleanup_after=True):
#     """Context manager for memory-managed operations"""
#     logger = logging.getLogger(__name__)
    
#     logger.info(f"Starting {operation_name}")
#     logger.info(f"Before {operation_name}: {get_gpu_memory_info()}")
    
#     try:
#         yield
#     except Exception as e:
#         logger.error(f"Error during {operation_name}: {e}")
#         comprehensive_memory_cleanup(verbose=True)
#         raise
#     finally:
#         if cleanup_after:
#             logger.info(f"Cleaning up after {operation_name}")
#             comprehensive_memory_cleanup(verbose=False)

# def safe_model_deletion(model, model_name="model"):
#     """Safely delete a model and cleanup associated memory"""
#     logger = logging.getLogger(__name__)
    
#     if model is not None:
#         logger.info(f"Deleting {model_name}")
        
#         # Move model to CPU first to free GPU memory
#         try:
#             if hasattr(model, 'cpu'):
#                 model.cpu()
#         except:
#             pass  # In case model is already on CPU or cannot be moved
        
#         # Delete the model
#         del model
        
#         # Force cleanup
#         comprehensive_memory_cleanup(verbose=False)
#         logger.info(f"{model_name} deleted and memory cleaned")

# def estimate_memory_usage(train_batch_size, eval_batch_size, model_name="eva-02"):
#     """
#     Estimate memory usage for training and evaluation
    
#     Args:
#         train_batch_size (int): Training batch size
#         eval_batch_size (int): Evaluation batch size
#         model_name (str): Model name for size estimation
    
#     Returns:
#         dict: Memory usage estimates
#     """
#     # Rough estimates for EVA-02 Large model (in GB)
#     base_model_memory = 3.5  # Model weights and buffers
    
#     # Memory per sample (rough estimates)
#     if model_name.startswith("eva-02"):
#         memory_per_train_sample = 1.2  # Includes gradients, optimizer states, activations
#         memory_per_eval_sample = 0.4   # Only forward pass activations
#     else:
#         memory_per_train_sample = 0.8   # Conservative estimate
#         memory_per_eval_sample = 0.3
    
#     train_memory = base_model_memory + (train_batch_size * memory_per_train_sample)
#     eval_memory = base_model_memory + (eval_batch_size * memory_per_eval_sample)
    
#     return {
#         "base_model": base_model_memory,
#         "training_total": train_memory,
#         "evaluation_total": eval_memory,
#         "training_batch_overhead": train_batch_size * memory_per_train_sample,
#         "evaluation_batch_overhead": eval_batch_size * memory_per_eval_sample
#     }

# def get_optimal_eval_batch_size(train_batch_size, available_memory_gb=None):
#     """
#     Calculate optimal evaluation batch size based on training batch size and available memory
    
#     Args:
#         train_batch_size (int): Training batch size per GPU
#         available_memory_gb (float): Available GPU memory in GB (optional)
    
#     Returns:
#         int: Recommended evaluation batch size
#     """
#     # Base multiplier: evaluation typically uses 60-70% less memory per sample
#     # because no gradients, optimizer states, or intermediate activations for backprop
#     base_multiplier = 2.5
    
#     if available_memory_gb is not None:
#         # Adjust based on available memory
#         if available_memory_gb < 8:
#             base_multiplier = 1.5  # Conservative for low memory
#         elif available_memory_gb > 16:
#             base_multiplier = 3.5  # More aggressive for high memory
#         elif available_memory_gb > 24:
#             base_multiplier = 4.0  # Very aggressive for very high memory
    
#     # Calculate optimal batch size
#     if train_batch_size == 1:
#         return max(2, min(6, int(train_batch_size * base_multiplier)))
#     elif train_batch_size <= 4:
#         return max(train_batch_size, min(12, int(train_batch_size * base_multiplier)))
#     else:
#         return max(train_batch_size, int(train_batch_size * 1.5))

# def memory_efficient_evaluation(model, test_loader, evaluator, logger, cleanup_interval=50, timeout_minutes=30):
#     """
#     Perform memory-efficient evaluation with tqdm progress bar
    
#     Args:
#         model: The model to evaluate
#         test_loader: Test data loader
#         evaluator: COCO evaluator
#         logger: Logger instance
#         cleanup_interval: How often to perform memory cleanup during evaluation
#         timeout_minutes: Maximum time to wait for evaluation (in minutes)
    
#     Returns:
#         dict: Evaluation results
#     """
#     logger.info("🚀 Starting memory-efficient evaluation")
#     logger.info(f"⏱️ Timeout: {timeout_minutes} minutes")
#     logger.info(f"💾 {get_gpu_memory_info()}")
    
#     model.eval()
    
#     # Estimate total batches for progress reporting
#     try:
#         total_batches = len(test_loader)
#         logger.info(f"📊 Total evaluation batches: {total_batches}")
#     except:
#         total_batches = None
#         logger.info("❓ Cannot determine total batch count")
    
#     # Check if test_loader is empty
#     if total_batches == 0:
#         logger.error("❌ Test loader is empty! No data to evaluate.")
#         return {}
    
#     # Perform evaluation with tqdm progress bar
#     with torch.no_grad():
#         evaluator.reset()
#         logger.info("🔄 Evaluator reset, starting inference...")
        
#         total_samples = 0
#         start_time = time.time()
#         timeout_seconds = timeout_minutes * 60
        
#         # Create progress bar
#         pbar = tqdm(
#             test_loader,
#             desc="🔍 Evaluating",
#             total=total_batches,
#             unit="batch",
#             ncols=120,
#             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, Samples: {postfix}]'
#         )
        
#         try:
#             for idx, inputs in enumerate(pbar):
#                 # Check for timeout
#                 elapsed = time.time() - start_time
#                 if elapsed > timeout_seconds:
#                     pbar.close()
#                     logger.error(f"⏰ Evaluation timeout after {elapsed:.1f} seconds!")
#                     break
                
#                 # Update progress bar postfix with current stats
#                 pbar.set_postfix_str(f"{total_samples}")
                
#                 # Validate inputs
#                 if not inputs:
#                     pbar.set_description("⚠️ Empty batch, skipping")
#                     continue
                    
#                 # Run inference
#                 try:
#                     outputs = model(inputs)
#                     if not outputs:
#                         pbar.set_description("⚠️ Empty outputs")
#                         continue
                        
#                     evaluator.process(inputs, outputs)
#                     total_samples += len(inputs)
                    
#                     # Update description with success
#                     pbar.set_description("🔍 Evaluating")
                    
#                 except Exception as e:
#                     pbar.set_description(f"❌ Error in batch {idx}")
#                     logger.error(f"Error processing batch {idx}: {e}")
#                     continue
                
#                 # Periodic memory cleanup during evaluation
#                 if (idx + 1) % cleanup_interval == 0:
#                     # Light cleanup during evaluation
#                     gc.collect()
#                     if torch.cuda.is_available():
#                         torch.cuda.empty_cache()
                    
#                     # Update progress bar with memory info
#                     allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
#                     pbar.set_description(f"🔍 Evaluating (GPU: {allocated:.1f}GB)")
            
#         finally:
#             pbar.close()
        
#         eval_time = time.time() - start_time
#         logger.info(f"✅ Inference completed! Processed {total_samples} samples in {eval_time:.2f}s")
        
#         if total_samples == 0:
#             logger.error("❌ No samples were processed during evaluation!")
#             return {}
            
#         # Computing final results with progress indication
#         logger.info("🧮 Computing evaluation metrics...")
        
#         # Show a simple spinner while computing metrics
#         with tqdm(desc="🧮 Computing metrics", bar_format='{desc}', leave=False) as pbar:
#             try:
#                 results = evaluator.evaluate()
#                 pbar.set_description("✅ Metrics computed!")
#                 if not results:
#                     logger.warning("⚠️ Evaluator returned empty results!")
#                     return {}
#                 logger.info("🎯 Evaluation metrics computation completed!")
#                 return results
#             except Exception as e:
#                 pbar.set_description("❌ Metrics computation failed!")
#                 logger.error(f"Error computing evaluation metrics: {e}")
#                 import traceback
#                 logger.error(f"Traceback: {traceback.format_exc()}")
#                 return {}

# class MemoryMonitoringHook(HookBase):
#     """Hook to monitor memory usage during training with tqdm-style updates"""
    
#     def __init__(self, log_interval=100, cleanup_interval=1000, verbose_cleanup=False):
#         self.log_interval = log_interval
#         self.cleanup_interval = cleanup_interval
#         self.verbose_cleanup = verbose_cleanup
#         self.logger = logging.getLogger(__name__)
#         self.last_memory_log = 0
    
#     def before_step(self):
#         iteration = self.trainer.iter
        
#         # Log memory usage periodically with emojis
#         if iteration % self.log_interval == 0:
#             allocated = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
#             self.logger.info(f"🔄 Iteration {iteration} | 💾 GPU: {allocated:.1f}GB")
        
#         # Periodic cleanup
#         if iteration % self.cleanup_interval == 0 and iteration > 0:
#             self.logger.info(f"🧹 Iteration {iteration}: Performing periodic cleanup")
#             comprehensive_memory_cleanup(verbose=self.verbose_cleanup)
    
#     def after_step(self):
#         pass

# # ========================================================================================
# # ENHANCED DATASET FUNCTIONS WITH TQDM
# # ========================================================================================

# def create_subset_annotations(annotation_file, output_file, subset_ratio=0.001):
#     """Create a subset of COCO annotations file with specified ratio of data"""
    
#     with memory_managed_operation(f"Creating subset annotations ({subset_ratio*100}%)", cleanup_after=True):
#         print(f"🎯 Creating subset annotations with {subset_ratio*100}% of original data...")
        
#         # Load original annotations with progress
#         with tqdm(desc="📖 Loading annotations", bar_format='{desc}', leave=False) as pbar:
#             with open(annotation_file, 'r') as f:
#                 data = json.load(f)
#             pbar.set_description("✅ Annotations loaded!")
        
#         # Get original counts
#         original_images = len(data['images'])
#         original_annotations = len(data['annotations'])
        
#         # Calculate subset size (minimum 1 image)
#         subset_size = max(1, int(original_images * subset_ratio))
        
#         print(f"📊 Original dataset: {original_images} images, {original_annotations} annotations")
#         print(f"🎯 Subset dataset: {subset_size} images (target)")
        
#         # Randomly sample images with progress
#         random.seed(42)  # For reproducibility
#         with tqdm(desc="🎲 Sampling images", total=subset_size, leave=False) as pbar:
#             sampled_images = random.sample(data['images'], subset_size)
#             pbar.update(subset_size)
        
#         sampled_image_ids = {img['id'] for img in sampled_images}
        
#         # Filter annotations for sampled images with progress
#         sampled_annotations = []
#         with tqdm(data['annotations'], desc="🔍 Filtering annotations", leave=False) as pbar:
#             for ann in pbar:
#                 if ann['image_id'] in sampled_image_ids:
#                     sampled_annotations.append(ann)
        
#         # Create subset data
#         subset_data = {
#             'info': data.get('info', {}),
#             'licenses': data.get('licenses', []),
#             'categories': data['categories'],  # Keep all categories
#             'images': sampled_images,
#             'annotations': sampled_annotations
#         }
        
#         # Add info about subset
#         subset_data['info']['description'] = f"Subset ({subset_ratio*100}%) of original dataset for testing"
        
#         # Save subset annotations with progress
#         with tqdm(desc="💾 Saving subset", bar_format='{desc}', leave=False) as pbar:
#             with open(output_file, 'w') as f:
#                 json.dump(subset_data, f)
#             pbar.set_description("✅ Subset saved!")
        
#         print(f"✅ Subset created: {len(sampled_images)} images, {len(sampled_annotations)} annotations")
#         print(f"🔍 Subset saved to: {output_file}")
        
#         return output_file

# def register_datasets(dataset_path, images_path, use_test_subset=False):
#     """Register COCO format datasets with memory management and progress"""
    
#     with memory_managed_operation("Dataset Registration", cleanup_after=True):
#         # Register training dataset
#         train_json = os.path.join(dataset_path, "annotation_train.json")
#         if not os.path.exists(train_json):
#             raise FileNotFoundError(f"❌ Training annotation file not found: {train_json}")
        
#         # Create subset if test mode is enabled
#         if use_test_subset:
#             # Create temporary directory for subset annotations 
#             temp_dir = tempfile.mkdtemp(prefix="eva_test_subset_")
            
#             # Create subset training annotations
#             subset_train_json = os.path.join(temp_dir, "annotation_train_subset.json")
#             create_subset_annotations(train_json, subset_train_json, subset_ratio=0.001)
            
#             # Register subset training dataset
#             register_coco_instances("custom_train", {}, subset_train_json, images_path)
#             print(f"✅ Registered TEST training dataset (0.1%): {subset_train_json}")
            
#             # Handle test dataset
#             test_json = os.path.join(dataset_path, "annotation_test.json")
#             if os.path.exists(test_json):
#                 # Create subset test annotations
#                 subset_test_json = os.path.join(temp_dir, "annotation_test_subset.json")
#                 create_subset_annotations(test_json, subset_test_json, subset_ratio=0.001)
                
#                 register_coco_instances("custom_test", {}, subset_test_json, images_path)
#                 print(f"✅ Registered TEST test dataset (0.1%): {subset_test_json}")
#                 return "custom_train", "custom_test"
#             else:
#                 print(f"⚠️ Warning: Test annotation file not found: {test_json}")
#                 return "custom_train", None
                
#         else:
#             # Regular full dataset registration
#             register_coco_instances("custom_train", {}, train_json, images_path)
#             print(f"✅ Registered FULL training dataset: {train_json}")
            
#             # Register test dataset
#             test_json = os.path.join(dataset_path, "annotation_test.json")
#             if os.path.exists(test_json):
#                 register_coco_instances("custom_test", {}, test_json, images_path)
#                 print(f"✅ Registered FULL test dataset: {test_json}")
#                 return "custom_train", "custom_test"
#             else:
#                 print(f"⚠️ Warning: Test annotation file not found: {test_json}")
#                 return "custom_train", None

# # ========================================================================================
# # PARSE ARGS AND OTHER FUNCTIONS
# # ========================================================================================

# def parse_args():
#     parser = argparse.ArgumentParser(description="EVA-02 Fine-tuning for Detection")
    
#     # Model selection
#     parser.add_argument("--model_name", default="eva-02-02", choices=["eva-02-01", "eva-02-02"],
#                         help="Either eva-02-01 or eva-02-02")
    
#     # Dataset paths
#     parser.add_argument("--dataset_path", default="/mnt/e/Desktop/GLaMM/detection_dataset_coco",
#                         help="Path to dataset directory containing annotation_train.json and annotation_test.json")
#     parser.add_argument("--images_path", default="/mnt/e/Desktop/GLaMM/detection_dataset_coco",
#                         help="Path to images directory")
    
#     # Training parameters
#     parser.add_argument("--output_dir", default="./eva_2_FT",
#                         help="Directory to save model checkpoints and logs")
#     parser.add_argument("--num_classes", type=int, default=30,
#                         help="Number of classes in your dataset")
#     parser.add_argument("--max_iter", type=int, default=300000,
#                         help="Maximum number of training iterations")
#     parser.add_argument("--eval_period", type=int, default=10000,
#                         help="Evaluation period")
#     parser.add_argument("--checkpoint_period", type=int, default=10000,
#                         help="Checkpoint saving period")
#     parser.add_argument("--learning_rate", type=float, default=0.0001,
#                         help="Learning rate")
#     parser.add_argument("--warmup_iters", type=int, default=1000,
#                         help="Warmup iterations")
    
#     # Worker configuration
#     parser.add_argument("--train_num_workers", type=int, default=2,
#                         help="Number of workers for training dataloader")
#     parser.add_argument("--test_num_workers", type=int, default=10,
#                         help="Number of workers for test dataloader")
    
#     # Memory management options
#     parser.add_argument("--memory_cleanup_interval", type=int, default=100,
#                         help="Interval for periodic memory cleanup during training")
#     parser.add_argument("--memory_log_interval", type=int, default=10000000,
#                         help="Interval for memory usage logging")
#     parser.add_argument("--aggressive_cleanup", action="store_true",
#                         help="Enable more aggressive memory cleanup (slower but more memory-efficient)")
    
#     # Evaluation options
#     parser.add_argument("--eval_before_training", action="store_true", default=True,
#                         help="Perform evaluation before training to establish baseline")
#     parser.add_argument("--skip_initial_eval", action="store_true",
#                         help="Skip initial evaluation (overrides --eval_before_training)")
#     parser.add_argument("--eval_only", action="store_true",
#                         help="Only perform evaluation, do not train")
    
#     # Test run option
#     parser.add_argument("--test", action="store_true",
#                         help="Run on 0.1% of the dataset as a test run")
    
#     # Resume training
#     parser.add_argument("--resume", action="store_true",
#                         help="Resume training from latest checkpoint")
#     parser.add_argument("--resume_checkpoint", type=str, default="",
#                         help="Specific checkpoint to resume from")
    
#     # Batch size parameters
#     parser.add_argument("--batch_size_per_gpu", type=int, default=1,
#                         help="Training batch size per GPU")
#     parser.add_argument("--eval_batch_size_per_gpu", type=int, default=1,
#                         help="Evaluation batch size per GPU (if None, will auto-calculate based on training batch size)")
#     parser.add_argument("--auto_eval_batch_size", action="store_true", default=True,
#                         help="Automatically determine optimal evaluation batch size")
    
#     # DDP Related parameters
#     parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
#     parser.add_argument('--local_rank', default=0, type=int)
#     parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
#     args = parser.parse_args()
    
#     # Handle conflicting flags
#     if args.skip_initial_eval:
#         args.eval_before_training = False
    
#     # Adjust cleanup intervals for aggressive mode
#     if args.aggressive_cleanup:
#         args.memory_cleanup_interval = min(500, args.memory_cleanup_interval)
#         args.memory_log_interval = min(50, args.memory_log_interval)
    
#     # Auto-calculate evaluation batch size if not specified
#     if args.eval_batch_size_per_gpu is None and args.auto_eval_batch_size:
#         # Use the optimal calculation function
#         train_batch = int(args.batch_size_per_gpu)
        
#         # Try to get available GPU memory for optimization
#         available_memory = None
#         if torch.cuda.is_available():
#             try:
#                 device = torch.cuda.current_device()
#                 total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
#                 # Assume we can use about 80% of total memory for evaluation
#                 available_memory = total_memory * 0.8
#             except:
#                 available_memory = None
        
#         args.eval_batch_size_per_gpu = get_optimal_eval_batch_size(train_batch, available_memory)
        
#         print(f"🎯 Auto-calculated evaluation batch size: {args.eval_batch_size_per_gpu} (training: {train_batch})")
#         if available_memory:
#             print(f"💾 Based on available GPU memory: {available_memory:.1f}GB")
#     elif args.eval_batch_size_per_gpu is None:
#         # Default to same as training if auto-calculation is disabled
#         args.eval_batch_size_per_gpu = int(args.batch_size_per_gpu)
    
#     # Ensure eval batch size is at least as large as training batch size
#     if args.eval_batch_size_per_gpu < args.batch_size_per_gpu:
#         print(f"⚠️ Warning: Evaluation batch size ({args.eval_batch_size_per_gpu}) is smaller than training batch size ({args.batch_size_per_gpu})")
#         print(f"This is unusual and may not be optimal. Consider using a larger evaluation batch size.")
    
#     return args

# def setup_config(args, train_dataset_name, test_dataset_name):
#     """Setup configuration for training with memory considerations"""
    
#     with memory_managed_operation("Configuration Setup", cleanup_after=True):
#         # Load base config
#         if args.model_name == "eva-02-01":
#             config_path = eva02_L_lvis_sys_o365_config_path
#             checkpoint_path = eva02_L_lvis_sys_o365_ckpt_path
#         else:
#             config_path = eva02_L_lvis_sys_config_path
#             checkpoint_path = eva02_L_lvis_sys_ckpt_path
        
#         cfg = LazyConfig.load(config_path)
        
#         # CRITICAL FIX: Configure for detection-only training
#         # 1. Disable mask training in the model
#         cfg.model.roi_heads.num_classes = args.num_classes
        
#         # 2. Remove mask head completely from the config
#         if hasattr(cfg.model.roi_heads, 'mask_head'):
#             delattr(cfg.model.roi_heads, 'mask_head')
#         if hasattr(cfg.model.roi_heads, 'mask_in_features'):
#             delattr(cfg.model.roi_heads, 'mask_in_features')
#         if hasattr(cfg.model.roi_heads, 'mask_pooler'):
#             delattr(cfg.model.roi_heads, 'mask_pooler')
        
#         # 3. Disable federated loss for custom dataset
#         # For cascade models, we need to update all box predictors
#         if hasattr(cfg.model.roi_heads, 'box_predictors'):
#             # This is a cascade model with multiple predictors
#             for i in range(len(cfg.model.roi_heads.box_predictors)):
#                 cfg.model.roi_heads.box_predictors[i].use_fed_loss = False
#                 cfg.model.roi_heads.box_predictors[i].use_sigmoid_ce = False
#                 cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.05
#                 cfg.model.roi_heads.box_predictors[i].test_topk_per_image = 1000
#                 # Remove the fed loss weight function
#                 if hasattr(cfg.model.roi_heads.box_predictors[i], 'get_fed_loss_cls_weights'):
#                     delattr(cfg.model.roi_heads.box_predictors[i], 'get_fed_loss_cls_weights')
#         elif hasattr(cfg.model.roi_heads, 'box_predictor'):
#             # This is a standard model with single predictor
#             cfg.model.roi_heads.box_predictor.use_fed_loss = False
#             cfg.model.roi_heads.box_predictor.use_sigmoid_ce = False
#             cfg.model.roi_heads.box_predictor.test_score_thresh = 0.05
#             cfg.model.roi_heads.box_predictor.test_topk_per_image = 1000
#             # Remove the fed loss weight function
#             if hasattr(cfg.model.roi_heads.box_predictor, 'get_fed_loss_cls_weights'):
#                 delattr(cfg.model.roi_heads.box_predictor, 'get_fed_loss_cls_weights')
        
#         # 4. CRITICAL FIX: Configure dataset names properly
#         cfg.dataloader.train.dataset.names = train_dataset_name
#         if test_dataset_name:
#             cfg.dataloader.test.dataset.names = test_dataset_name
            
#         # Adjust number of workers based on memory constraints and user input
#         cfg.dataloader.train.num_workers = args.train_num_workers
#         cfg.dataloader.test.num_workers = args.test_num_workers

#         # Configure the dataset mapper for detection-only (no masks)
#         cfg.dataloader.train.mapper = L(DatasetMapper)(
#             is_train=True,
#             augmentations=[
#                 L(T.ResizeShortestEdge)(
#                     short_edge_length=[640, 672, 704, 736, 768, 800],
#                     sample_style="choice",
#                     max_size=1333,
#                 ),
#                 L(T.RandomFlip)(horizontal=True),
#                 L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.2),
#                 L(T.RandomContrast)(intensity_min=0.8, intensity_max=1.2),
#                 L(T.RandomSaturation)(intensity_min=0.8, intensity_max=1.2),
#             ],
#             image_format="BGR",
#             use_instance_mask=False,  # CRITICAL: Disable mask usage
#             use_keypoint=False,
#             recompute_boxes=False,
#         )
        
#         cfg.dataloader.test.mapper = L(DatasetMapper)(
#             is_train=False,
#             augmentations=[
#                 L(T.ResizeShortestEdge)(short_edge_length=800, max_size=1333),
#             ],
#             image_format="BGR",
#             use_instance_mask=False,  # CRITICAL: Disable mask usage
#             use_keypoint=False,
#         )
        
#         # Training parameters
#         cfg.train.max_iter = args.max_iter
        
#         # Adjust training parameters for test mode
#         if args.test:
#             # Reduce iterations for test mode
#             cfg.train.max_iter = min(100, args.max_iter)  # Max 100 iterations for test
#             args.eval_period = min(50, args.eval_period)  # Evaluate every 50 iterations
#             args.checkpoint_period = min(50, args.checkpoint_period)  # Save every 50 iterations
#             print(f"🧪 TEST MODE: Reduced max_iter to {cfg.train.max_iter}")
        
#         # Fix: Use correct optimizer parameter path
#         if hasattr(cfg, 'optimizer') and hasattr(cfg.optimizer, 'lr'):
#             cfg.optimizer.lr = args.learning_rate
#         elif hasattr(cfg, 'optimizer') and hasattr(cfg.optimizer, 'base_lr'):
#             cfg.optimizer.base_lr = args.learning_rate
            
#         # Warmup iterations - might be in lr_multiplier
#         if hasattr(cfg, 'lr_multiplier') and hasattr(cfg.lr_multiplier, 'warmup_length'):
#             cfg.lr_multiplier.warmup_length = args.warmup_iters / cfg.train.max_iter
        
#         # Checkpoint period
#         if hasattr(cfg.train, 'checkpointer'):
#             cfg.train.checkpointer.period = args.checkpoint_period
        
#         # FIXED: Set batch sizes differently for train and test
#         cfg.dataloader.train.total_batch_size = args.batch_size_per_gpu * args.world_size
#         # Remove total_batch_size from test config - use samples_per_gpu instead
#         if hasattr(cfg.dataloader.test, 'total_batch_size'):
#             delattr(cfg.dataloader.test, 'total_batch_size')
#         # Set samples per GPU for test dataloader
#         cfg.dataloader.test.samples_per_gpu = args.eval_batch_size_per_gpu
        
#         # Output directory
#         if args.test:
#             # Create separate output directory for test runs
#             test_output_dir = os.path.join(args.output_dir, "test_run")
#             os.makedirs(test_output_dir, exist_ok=True)
#             cfg.train.output_dir = test_output_dir
#             print(f"🧪 TEST MODE: Output directory set to {test_output_dir}")
#         else:
#             cfg.train.output_dir = args.output_dir
        
#         # Checkpoint path for initialization
#         cfg.train.init_checkpoint = checkpoint_path
        
#         # Evaluation
#         cfg.train.eval_period = args.eval_period
        
#         # Update evaluator for custom dataset - ONLY for bbox evaluation
#         if hasattr(cfg, 'dataloader') and hasattr(cfg.dataloader, 'evaluator'):
#             cfg.dataloader.evaluator = L(COCOEvaluator)(
#                 dataset_name=test_dataset_name if test_dataset_name else train_dataset_name,
#                 tasks=["bbox"],  # CRITICAL: Only evaluate bbox, not segm
#                 output_dir=cfg.train.output_dir,
#             )
        
#         return cfg

# def perform_initial_evaluation(cfg, model_weights_path, test_dataset_name, logger, args):
#     """Perform initial evaluation before training with enhanced memory management and progress bars"""
    
#     with memory_managed_operation("Initial Evaluation", cleanup_after=True):
#         logger.info("=" * 80)
#         logger.info("🎯 PERFORMING INITIAL EVALUATION (PRE-TRAINING BASELINE)")
#         logger.info("=" * 80)
        
#         # Create model for evaluation with progress indication
#         with tqdm(desc="🏗️ Creating model", bar_format='{desc}', leave=False) as pbar:
#             model = instantiate(cfg.model)
#             model.to(cfg.train.device)
#             model.eval()
#             pbar.set_description("✅ Model created")
        
#         # Load pre-trained weights with progress indication
#         with tqdm(desc="📥 Loading weights", bar_format='{desc}', leave=False) as pbar:
#             checkpointer = DetectionCheckpointer(model, cfg.train.output_dir)
#             checkpointer.load(model_weights_path)
#             pbar.set_description("✅ Weights loaded")
        
#         logger.info(f"✅ Loaded pre-trained weights from: {model_weights_path}")
#         logger.info(f"🎯 Evaluating on dataset: {test_dataset_name}")
#         logger.info(f"💾 {get_gpu_memory_info()}")
        
#         # Verify dataset exists and has data
#         dataset_dicts = DatasetCatalog.get(test_dataset_name)
#         logger.info(f"📊 Dataset verification: {len(dataset_dicts)} samples found")
        
#         if len(dataset_dicts) == 0:
#             logger.error("❌ Dataset is empty! Cannot perform evaluation.")
#             return {}
        
#         # Create data loader for evaluation
#         with tqdm(desc="🔄 Creating data loader", bar_format='{desc}', leave=False) as pbar:
#             test_loader = build_detection_test_loader(
#                 dataset=dataset_dicts,
#                 mapper=instantiate(cfg.dataloader.test.mapper),
#                 num_workers=cfg.dataloader.test.num_workers,
#                 batch_size=args.eval_batch_size_per_gpu
#             )
#             pbar.set_description("✅ Data loader ready")
        
#         logger.info(f"🎯 Test loader created with batch size: {args.eval_batch_size_per_gpu}")
#         logger.info(f"📊 Test dataset size: {len(dataset_dicts)} samples")
        
#         # Quick test: Process one batch to check if everything works
#         with tqdm(desc="🧪 Testing model", bar_format='{desc}', leave=False) as pbar:
#             try:
#                 test_iter = iter(test_loader)
#                 sample_batch = next(test_iter)
                
#                 with torch.no_grad():
#                     sample_outputs = model(sample_batch)
                
#                 pbar.set_description("✅ Model test passed")
#                 del sample_outputs, sample_batch, test_iter
#                 torch.cuda.empty_cache()
#             except Exception as e:
#                 pbar.set_description("❌ Model test failed")
#                 logger.error(f"Model test failed: {e}")
#                 return {}
        
#         # Create evaluator
#         logger.info("🧮 Creating evaluator...")
#         evaluator = COCOEvaluator(
#             test_dataset_name, 
#             tasks=["bbox"], 
#             output_dir=os.path.join(cfg.train.output_dir, "initial_eval")
#         )
        
#         logger.info("🚀 Starting full evaluation...")
        
#         # Run memory-efficient evaluation
#         start_time = time.time()
#         results = memory_efficient_evaluation(model, test_loader, evaluator, logger, cleanup_interval=20)
#         eval_time = time.time() - start_time
        
#         logger.info(f"✅ Initial evaluation completed in {eval_time:.2f} seconds")
#         logger.info("📊 Initial evaluation results (PRE-TRAINING BASELINE):")
        
#         # Print results in a more readable format
#         if results:
#             print_csv_format(results)
#         else:
#             logger.warning("⚠️ No evaluation results returned!")
        
#         # Save initial evaluation results
#         initial_eval_dir = os.path.join(cfg.train.output_dir, "initial_eval")
#         os.makedirs(initial_eval_dir, exist_ok=True)
        
#         # Save results as JSON
#         results_file = os.path.join(initial_eval_dir, "initial_evaluation_results.json")
#         with open(results_file, 'w') as f:
#             json.dump(results, f, indent=2)
#         logger.info(f"💾 Initial evaluation results saved to: {results_file}")
        
#         # Print key metrics summary
#         if results and 'bbox' in results:
#             bbox_results = results['bbox']
#             logger.info("=" * 60)
#             logger.info("🎯 INITIAL BASELINE SUMMARY:")
#             logger.info(f"  📊 Bbox AP     : {bbox_results.get('AP', 'N/A'):.3f}")
#             logger.info(f"  🎯 Bbox AP50   : {bbox_results.get('AP50', 'N/A'):.3f}")
#             logger.info(f"  📍 Bbox AP75   : {bbox_results.get('AP75', 'N/A'):.3f}")
#             logger.info("=" * 60)
#         else:
#             logger.warning("⚠️ No bbox results found in evaluation!")
        
#         # Cleanup evaluation components
#         safe_model_deletion(model, "evaluation_model")
#         del test_loader
#         del evaluator
        
#         logger.info("✅ Initial evaluation completed. Ready to start training...")
#         return results

# # ========================================================================================
# # 🔧 FIXED: PROPER DEFAULTTRAINER IMPLEMENTATION
# # ========================================================================================

# class EVA02Trainer(DefaultTrainer):
#     """
#     ✅ FIXED: Custom trainer for EVA-02 fine-tuning with COMPLETE checkpoint management
    
#     This trainer properly handles:
#     - ✅ Model weights
#     - ✅ Optimizer state  
#     - ✅ Scheduler state
#     - ✅ Iteration counter
#     - ✅ Training metadata
#     - ✅ Random number generator states
#     """
    
#     def __init__(self, cfg, args):
#         self.args = args  # Store args for later use
#         super().__init__(cfg)
    
#     @classmethod
#     def build_evaluator(cls, cfg, dataset_name):
#         """Build COCO evaluator for detection only"""
#         return COCOEvaluator(dataset_name, tasks=["bbox"], output_dir=cfg.train.output_dir)
    
#     @classmethod
#     def build_train_loader(cls, cfg):
#         """Build training data loader"""
#         return instantiate(cfg.dataloader.train)
    
#     @classmethod
#     def build_test_loader(cls, cfg, dataset_name):
#         """Build test data loader - FIXED"""
#         return build_detection_test_loader(
#             dataset=DatasetCatalog.get(dataset_name),
#             mapper=instantiate(cfg.dataloader.test.mapper),
#             num_workers=cfg.dataloader.test.num_workers,
#             batch_size=getattr(cfg.dataloader.test, 'samples_per_gpu', 2)
#         )
    
#     @classmethod
#     def build_optimizer(cls, cfg, model):
#         """Build optimizer - FIXED: Properly set model reference"""
#         cfg.optimizer.params.model = model
#         return instantiate(cfg.optimizer)
    
#     @classmethod 
#     def build_lr_scheduler(cls, cfg, optimizer):
#         """Build learning rate scheduler - FIXED: Ensures scheduler state is saved"""
#         return instantiate(cfg.lr_multiplier)
    
#     def build_hooks(self):
#         """
#         Build training hooks with memory management - FIXED
#         DefaultTrainer automatically includes proper checkpoint saving hooks
#         """
#         # Get default hooks from parent (includes proper checkpoint management)
#         hooks_list = super().build_hooks()
        
#         # Add our custom memory monitoring hook
#         memory_hook = MemoryMonitoringHook(
#             log_interval=self.args.memory_log_interval,
#             cleanup_interval=self.args.memory_cleanup_interval,
#             verbose_cleanup=self.args.aggressive_cleanup
#         )
        
#         # Insert memory hook before the last hook (usually TensorboardXWriter)
#         hooks_list.insert(-1, memory_hook)
        
#         return hooks_list
    
#     def resume_or_load(self, resume=True):
#         """
#         ✅ FIXED: Proper resume/load with complete checkpoint state
#         DefaultTrainer handles all checkpoint state automatically
#         """
#         if resume and self.args.resume_checkpoint:
#             # Resume from specific checkpoint
#             checkpoint_path = self.args.resume_checkpoint
#             self.logger.info(f"🔄 Resuming from specific checkpoint: {checkpoint_path}")
#             super().resume_or_load(checkpoint_path, resume=True)
#         elif resume:
#             # Resume from latest checkpoint in output directory
#             self.logger.info(f"🔄 Resuming from latest checkpoint in: {self.cfg.train.output_dir}")
#             super().resume_or_load(resume=True)
#         else:
#             # Load pretrained weights for fine-tuning (not resuming)
#             init_checkpoint = self.cfg.train.init_checkpoint
#             self.logger.info(f"📥 Loading pretrained weights from: {init_checkpoint}")
#             super().resume_or_load(init_checkpoint, resume=False)
    
#     def train(self):
#         """
#         ✅ FIXED: Enhanced training with proper state management
#         """
#         self.logger.info("🚀 Starting training with complete checkpoint state management")
#         self.logger.info(f"💾 {get_gpu_memory_info()}")
        
#         # Call parent train method - handles all checkpoint state automatically
#         return super().train()

# def train_model(args):
#     """Main training function with enhanced memory management and FIXED checkpoint handling"""
    
#     # Initialize start_time early to prevent UnboundLocalError
#     start_time = time.time()
    
#     # Setup logging
#     output_dir = os.path.join(args.output_dir, "test_run") if args.test else args.output_dir
#     setup_logger(output=output_dir, distributed_rank=args.local_rank)
#     logger = logging.getLogger(__name__)
    
#     # Set logger level to INFO to show evaluation progress
#     logger.setLevel(logging.INFO)
    
#     # Create output directory
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Log memory management settings with estimates
#     logger.info("=" * 60)
#     logger.info("🔧 MEMORY AND BATCH SIZE CONFIGURATION:")
#     logger.info(f"  🧹 Memory cleanup interval: {args.memory_cleanup_interval} iterations")
#     logger.info(f"  📍 Memory log interval: {args.memory_log_interval} iterations")
#     logger.info(f"  🚀 Aggressive cleanup: {args.aggressive_cleanup}")
#     logger.info(f"  🎯 Training batch size per GPU: {args.batch_size_per_gpu}")
#     logger.info(f"  📊 Evaluation batch size per GPU: {args.eval_batch_size_per_gpu}")
#     logger.info(f"  👥 Training workers: {args.train_num_workers}")
#     logger.info(f"  🧪 Test workers: {args.test_num_workers}")
#     batch_ratio = args.eval_batch_size_per_gpu / args.batch_size_per_gpu
#     logger.info(f"  📈 Eval/Train batch ratio: {batch_ratio:.1f}x")
    
#     # Memory usage estimates
#     memory_est = estimate_memory_usage(args.batch_size_per_gpu, args.eval_batch_size_per_gpu, args.model_name)
#     logger.info("  💾 Estimated Memory Usage:")
#     logger.info(f"    🏗️ Model base: {memory_est['base_model']:.1f}GB")
#     logger.info(f"    🎯 Training total: {memory_est['training_total']:.1f}GB")
#     logger.info(f"    📊 Evaluation total: {memory_est['evaluation_total']:.1f}GB")
#     logger.info(f"    💰 Memory savings during eval: {memory_est['training_total'] - memory_est['evaluation_total']:.1f}GB")
#     logger.info("=" * 60)
    
#     # Print mode status
#     if args.test:
#         logger.info("=" * 60)
#         logger.info("🧪 RUNNING IN TEST MODE (0.1% of dataset)")
#         logger.info("=" * 60)
    
#     if args.eval_only:
#         logger.info("=" * 60)
#         logger.info("📊 RUNNING IN EVALUATION-ONLY MODE")
#         logger.info("=" * 60)
    
#     # ✅ CHECKPOINT STATUS LOGGING
#     logger.info("=" * 60)
#     logger.info("🔧 CHECKPOINT MANAGEMENT STATUS:")
#     logger.info("  ✅ Model weights: SAVED")
#     logger.info("  ✅ Optimizer state: SAVED") 
#     logger.info("  ✅ Scheduler state: SAVED")
#     logger.info("  ✅ Iteration counter: SAVED")
#     logger.info("  ✅ Training metadata: SAVED")
#     logger.info("  ✅ Random number states: SAVED")
#     logger.info("  🎯 Using DefaultTrainer for complete state management")
#     if args.resume:
#         logger.info("  🔄 RESUME MODE: Will restore all training state")
#     else:
#         logger.info("  📥 INITIAL MODE: Will load pretrained weights only")
#     logger.info("=" * 60)
    
#     # Initial memory status
#     logger.info(f"🚀 Initial memory status: {get_gpu_memory_info()}")
#     logger.info(f"💻 Initial memory status: {get_cpu_memory_info()}")
    
#     # Register datasets
#     train_dataset, test_dataset = register_datasets(args.dataset_path, args.images_path, use_test_subset=args.test)
    
#     # Setup configuration
#     cfg = setup_config(args, train_dataset, test_dataset)
    
#     # Debug: Print some key configuration values
#     logger.info(f"🎯 Number of classes: {cfg.model.roi_heads.num_classes}")
#     logger.info(f"🏋️ Training dataset: {cfg.dataloader.train.dataset.names}")
#     if test_dataset:
#         logger.info(f"🧪 Test dataset: {cfg.dataloader.test.dataset.names}")
#     logger.info(f"🔄 Max iterations: {cfg.train.max_iter}")
#     logger.info(f"📦 Training batch size: {cfg.dataloader.train.total_batch_size} (per GPU: {args.batch_size_per_gpu})")
#     logger.info(f"📊 Evaluation batch size per GPU: {getattr(cfg.dataloader.test, 'samples_per_gpu', args.eval_batch_size_per_gpu)}")
#     logger.info(f"🎯 Detection-only mode: masks disabled")
    
#     if args.test:
#         logger.info(f"🧪 TEST MODE: Running on 0.1% of dataset")
#         logger.info(f"🧪 TEST MODE: Max iterations reduced to {cfg.train.max_iter}")
    
#     # Check if federated loss is disabled
#     if hasattr(cfg.model.roi_heads, 'box_predictors'):
#         logger.info(f"✅ Federated loss disabled for cascade model")
#     elif hasattr(cfg.model.roi_heads, 'box_predictor'):
#         logger.info(f"✅ Federated loss disabled for standard model")
    
#     # PERFORM INITIAL EVALUATION BEFORE TRAINING
#     initial_results = None
#     if args.eval_before_training and test_dataset and not args.resume:
#         try:
#             initial_results = perform_initial_evaluation(
#                 cfg, 
#                 cfg.train.init_checkpoint, 
#                 test_dataset, 
#                 logger,
#                 args  # Pass args parameter
#             )
#         except Exception as e:
#             logger.warning(f"⚠️ Initial evaluation failed: {e}")
#             logger.warning("🔄 Continuing with training...")
#     elif args.eval_only and test_dataset:
#         # Only perform evaluation and exit
#         try:
#             initial_results = perform_initial_evaluation(
#                 cfg, 
#                 cfg.train.init_checkpoint, 
#                 test_dataset, 
#                 logger,
#                 args  # Pass args parameter
#             )
#             logger.info("✅ Evaluation-only mode completed successfully.")
#             return initial_results
#         except Exception as e:
#             logger.error(f"❌ Evaluation failed: {e}")
#             raise
#     elif args.resume:
#         logger.info("🔄 Resuming training - skipping initial evaluation")
#     elif not test_dataset:
#         logger.warning("⚠️ No test dataset available - skipping initial evaluation")
#     else:
#         logger.info("⭐ Initial evaluation disabled")
    
#     # If eval_only mode, we're done
#     if args.eval_only:
#         return initial_results
    
#     # ✅ FIXED: SETUP TRAINER WITH COMPLETE CHECKPOINT MANAGEMENT
#     trainer = None
    
#     try:
#         with memory_managed_operation("Trainer Setup", cleanup_after=False):
#             logger.info("🏗️ Creating EVA02Trainer with complete checkpoint management...")
            
#             # ✅ FIXED: Create trainer using DefaultTrainer (handles all checkpoint state)
#             trainer = EVA02Trainer(cfg, args)
            
#             logger.info("✅ Trainer created with complete state management")
#             logger.info(f"💾 Memory after trainer creation: {get_gpu_memory_info()}")
            
#             # ✅ FIXED: Proper resume/load handling
#             if args.resume:
#                 logger.info("🔄 RESUMING TRAINING - All state will be restored")
#                 trainer.resume_or_load(resume=True)
#                 current_iter = trainer.start_iter
#                 logger.info(f"📍 Resumed from iteration: {current_iter}")
#                 logger.info(f"🎯 Will train until iteration: {trainer.max_iter}")
#             else:
#                 logger.info("📥 LOADING PRETRAINED WEIGHTS for fine-tuning")
#                 trainer.resume_or_load(resume=False)
#                 logger.info(f"🎯 Will train from iteration 0 to {trainer.max_iter}")
            
#             logger.info(f"💾 Memory after checkpoint loading: {get_gpu_memory_info()}")
        
#         # Start training
#         mode_str = "🧪 TEST detection-only training (0.1% dataset)" if args.test else "🚀 FULL detection-only training"
#         logger.info(f"Starting {mode_str}...")
        
#         if initial_results and 'bbox' in initial_results:
#             logger.info(f"🎯 Training will start from baseline AP: {initial_results['bbox'].get('AP', 'N/A'):.3f}")
        
#         logger.info(f"💾 Memory before training: {get_gpu_memory_info()}")
#         start_time = time.time()  # Reset start time just before training
        
#         # ✅ FIXED: Training with complete checkpoint state management
#         print("\n" + "="*80)
#         print("🚀 STARTING TRAINING - All checkpoint state will be saved automatically")
#         print("   ✅ Model weights, optimizer, scheduler, iteration counter, metadata")
#         print("="*80)
        
#         # DefaultTrainer.train() handles everything automatically
#         trainer.train()
        
#     except KeyboardInterrupt:
#         logger.info("⏹️ Training interrupted by user")
#     except Exception as e:
#         logger.error(f"❌ Training failed with error: {e}")
#         logger.error(f"💾 Memory at error: {get_gpu_memory_info()}")
#         comprehensive_memory_cleanup(verbose=True)
#         raise
#     finally:
#         end_time = time.time()
#         training_time = end_time - start_time
#         logger.info(f"✅ Training completed in {training_time:.2f} seconds")
        
#         # Final evaluation if test dataset is available
#         if test_dataset and trainer is not None:
#             try:
#                 logger.info("🏁 Running final evaluation...")
                
#                 # Use the trainer's built-in test method for consistency
#                 final_results = trainer.test()
                
#                 logger.info("🏆 Final evaluation results:")
#                 if final_results:
#                     # final_results from trainer.test() is typically a dict of evaluator results
#                     for eval_name, results in final_results.items():
#                         if results:
#                             print_csv_format(results)
#                 else:
#                     logger.warning("⚠️ No final evaluation results returned!")
                
#                 # Compare with initial results if available
#                 if initial_results and final_results:
#                     # Extract bbox results from final evaluation
#                     final_bbox_results = None
#                     for eval_name, results in final_results.items():
#                         if results and 'bbox' in results:
#                             final_bbox_results = results['bbox']
#                             break
                    
#                     if final_bbox_results and 'bbox' in initial_results:
#                         initial_ap = initial_results['bbox'].get('AP', 0.0)
#                         final_ap = final_bbox_results.get('AP', 0.0)
#                         improvement = final_ap - initial_ap
                        
#                         logger.info("=" * 80)
#                         logger.info("🎯 TRAINING RESULTS SUMMARY:")
#                         logger.info(f"  🎯 Initial AP (baseline): {initial_ap:.3f}")
#                         logger.info(f"  🏆 Final AP (trained)   : {final_ap:.3f}")
#                         if improvement > 0:
#                             logger.info(f"  📈 Improvement          : +{improvement:.3f} 🎉")
#                         else:
#                             logger.info(f"  📉 Change               : {improvement:+.3f}")
#                         logger.info("=" * 80)
#                 elif final_results:
#                     # Try to extract any bbox AP for summary
#                     for eval_name, results in final_results.items():
#                         if results and 'bbox' in results:
#                             final_ap = results['bbox'].get('AP', 0.0)
#                             logger.info("=" * 80)
#                             logger.info("🎯 TRAINING RESULTS SUMMARY:")
#                             logger.info(f"  🏆 Final AP (trained): {final_ap:.3f}")
#                             logger.info("=" * 80)
#                             break
#             except Exception as e:
#                 logger.warning(f"⚠️ Final evaluation failed: {e}")
#                 import traceback
#                 logger.warning(f"Traceback: {traceback.format_exc()}")
        
#         if args.test:
#             logger.info("=" * 60)
#             logger.info("🎉 TEST RUN COMPLETED SUCCESSFULLY")
#             logger.info("✅ You can now run the full training without --test flag")
#             logger.info("✅ Checkpoints contain complete training state for resuming")
#             logger.info("=" * 60)
        
#         # COMPREHENSIVE CLEANUP
#         logger.info("🧹 Performing comprehensive cleanup...")
        
#         # Safe deletion of training components
#         if trainer is not None:
#             del trainer
        
#         # Final memory cleanup
#         comprehensive_memory_cleanup(verbose=True)
        
#         logger.info("✅ Training and cleanup completed successfully")

# def main():
#     """Main function"""
#     args = parse_args()
    
#     # Initialize distributed training
#     init_distributed_mode(args)
    
#     # Set device
#     torch.cuda.set_device(args.local_rank)
    
#     # Log initial system information
#     print(f"🚀 Starting EVA-02 fine-tuning with enhanced memory management")
#     print(f"✅ CHECKPOINT FIX: Complete training state will be saved and restored")
#     print(f"💾 Initial GPU memory: {get_gpu_memory_info()}")
#     print(f"💻 Initial CPU memory: {get_cpu_memory_info()}")
    
#     # Train model
#     train_model(args)

# def distributed_main():
#     """Entry point for distributed training"""
#     args = parse_args()
    
#     print(f"⚙️ Command Line Args: {args}")
    
#     # Launch distributed training
#     launch(
#         main,
#         args.world_size,
#         num_machines=1,
#         machine_rank=0,
#         dist_url=args.dist_url,
#         args=(args,),
#     )

# if __name__ == "__main__":
#     main()