# (grand_env_4) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_1_inference/5_eva_02$
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --model_name 'eva-02-01' --local_rank 0
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --model_name 'eva-02-02' --local_rank 0
import argparse
import json
import numpy as np
import torch
import torch.nn.parallel
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import _create_text_labels
from torch.utils.data import DataLoader, DistributedSampler
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from copy import copy
from ddp import *
from rle_format import mask_to_rle_pytorch, coco_encode_rle
from tqdm import tqdm
import time
import os
import gc

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
                                     "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
eva02_L_lvis_sys_config_path = ("projects/ViTDet/configs/eva2_mim_to_lvis/"
                                "eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py")

eva02_L_lvis_sys_o365_ckpt_path = ("eva02_L_lvis_sys_o365.pth")
eva02_L_lvis_sys_ckpt_path = ("eva02_L_lvis_sys.pth")

opts_1 = f"train.init_checkpoint={eva02_L_lvis_sys_o365_ckpt_path}"
opts_2 = f"train.init_checkpoint={eva02_L_lvis_sys_ckpt_path}"


def wsl2_memory_optimization():
    """Aggressive memory cleanup specifically for WSL2 environments"""
    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
    
    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Try to force memory compaction (WSL2 specific)
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except:
        pass  # Not available on all systems


def check_memory_usage():
    """Check current memory usage and print if it's getting high"""
    if not PSUTIL_AVAILABLE:
        return 0
        
    try:
        memory = psutil.virtual_memory()
        percent = memory.percent
        return percent
    except Exception:
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description="EVA-02 Object Detection")

    parser.add_argument("--model_name", required=True,
                        help="Either eva-02-01 or eva-02-02. Specifying eva-02-01 runs eva02_L_lvis_sys_o365_ckpt "
                             "while eva-02-02 runs eva02_L_lvis_sys")
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--opts", required=False, default="")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def setup(config_file, opts):
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, [opts])
    return cfg


def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = (preds.scores > confidence_threshold).cpu()
        predictions = copy(predictions)  # don't modify the original
        predictions["instances"] = preds[keep_idxs]
    return predictions


def process_single_image(image_name, predictions, args, model_name, metadata):
    """
    Process a single image immediately - NO multiprocessing, NO queuing
    Ensures only one image is processed at a time
    """
    try:
        bboxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy().tolist() if predictions.has("scores") else None
        pred_classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        pred_masks = predictions.pred_masks
        
        labels = _create_text_labels(pred_classes, None, metadata.get("thing_classes", None))
        
        # Convert masks to RLE format
        if pred_masks is not None:
            uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks)
        else:
            uncompressed_mask_rles = []

        all_image_predictions = []
        if bboxes is not None:
            for j, box in enumerate(bboxes):
                prediction = {}
                prediction['bbox'] = [round(float(b), 2) for b in box]
                if j < len(uncompressed_mask_rles):
                    prediction['sam_mask'] = coco_encode_rle(uncompressed_mask_rles[j])
                prediction['score'] = round(scores[j], 2) if scores else 0.0
                prediction['label'] = labels[j] if j < len(labels) else ""
                all_image_predictions.append(prediction)

        # Create output data
        all_data = {}
        all_data[image_name] = {}
        all_data[image_name][model_name] = {}
        all_data[image_name][model_name] = [{k: json_serializable(v) for k, v in prediction.items()} for
                                            prediction in all_image_predictions]

        # Write all_data to a JSON file immediately
        with open(f"{args.output_dir_path}/{model_name}/{image_name[:-4]}.json", 'w') as f:
            json.dump(all_data, f)
            
        # EXPLICIT CLEANUP - only keep one image worth of data in memory
        del bboxes, scores, pred_classes, pred_masks, labels, uncompressed_mask_rles, all_image_predictions, all_data
        
    except Exception as e:
        print(f"Error processing {image_name}: {e}")


def post_processing(q, args, model_name, metadata):
    while True:
        data = q.get()
        if data == "STOP":
            break

        image_name, pred_data = data

        bboxes = pred_data['pred_boxes']
        scores = pred_data['scores'].tolist() if pred_data['scores'] is not None else None
        pred_classes = pred_data['pred_classes'].tolist() if pred_data['pred_classes'] is not None else None
        pred_masks = pred_data['pred_masks']
        
        labels = _create_text_labels(pred_classes, None, metadata.get("thing_classes", None))
        
        # Convert masks to RLE format
        if pred_masks is not None:
            # Convert numpy masks back to torch for RLE processing
            pred_masks_torch = torch.from_numpy(pred_masks)
            uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_torch)
        else:
            uncompressed_mask_rles = []

        all_image_predictions = []
        if bboxes is not None:
            for j, box in enumerate(bboxes):
                prediction = {}
                prediction['bbox'] = [round(float(b), 2) for b in box]
                if j < len(uncompressed_mask_rles):
                    prediction['sam_mask'] = coco_encode_rle(uncompressed_mask_rles[j])
                prediction['score'] = round(scores[j], 2) if scores else 0.0
                prediction['label'] = labels[j] if j < len(labels) else ""
                all_image_predictions.append(prediction)

        # all_image_predictions = compute_depth(all_image_predictions, image_name[:-4])
        all_data = {}

        all_data[image_name] = {}
        all_data[image_name][model_name] = {}
        all_data[image_name][model_name] = [{k: json_serializable(v) for k, v in prediction.items()} for
                                            prediction in all_image_predictions]

        # Write all_data to a JSON file (file_wise)
        with open(f"{args.output_dir_path}/{model_name}/{image_name[:-4]}.json", 'w') as f:
            json.dump(all_data, f)


def run_inference(args, dataloader, confidence_threshold=0.05):
    # NO multiprocessing - process everything sequentially
    metadata = MetadataCatalog.get("lvis_v1_val")
    model_name = args.model_name
    os.makedirs(f"{args.output_dir_path}/{model_name}", exist_ok=True)

    # Initiate cfg
    cfg = setup(eva02_L_lvis_sys_o365_config_path if args.model_name == "eva-02-01" else eva02_L_lvis_sys_config_path,
                opts_1 if args.model_name == "eva-02-01" else opts_2)
    # Initiate the model (create, load checkpoints, put in eval mode)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.train.init_checkpoint)

    model.eval()
    
    if args.local_rank > 0:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    else:
        model = model.to(args.local_rank)

    start_time = time.time()
    processed_count = 0
    
    with torch.no_grad():
        for (image_name, image, image_height, image_width) in tqdm(dataloader):
            # AGGRESSIVE CLEANUP BEFORE EACH IMAGE
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if len(image) == 0:
                continue
            image_name = image_name[0]
            
            # Remove batch dimension - dataloader adds batch dim, but detectron2 expects (C, H, W)
            image_tensor = image[0].squeeze(0) if image[0].dim() == 4 else image[0]
            
            inputs = [{'image': image_tensor, "height": int(image_height), "width": int(image_width)}]
            predictions = model(inputs)[0]

            predictions = filter_predictions_with_confidence(predictions, confidence_threshold)
            predictions = predictions["instances"].to("cpu")

            # Process immediately - NO queuing, NO multiprocessing
            process_single_image(image_name, predictions, args, model_name, metadata)
            
            # EXPLICIT CLEANUP AFTER EACH IMAGE - Only one image in memory at a time
            del image_tensor, inputs, predictions
            
            processed_count += 1
            if processed_count % 3 == 0:
                memory_percent = check_memory_usage()
                if memory_percent > 75:
                    wsl2_memory_optimization()
                else:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    gc.collect()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('\033[92m' + "---- EVA-1 Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


class CustomImageDataset:
    """
    Custom dataset that handles image loading with only the required transform
    """
    def __init__(self, image_dir_path, output_dir_path, transforms=None, model_name=None):
        self.image_dir_path = image_dir_path
        self.output_dir_path = output_dir_path
        self.transforms = transforms
        self.model_name = model_name
        
        # Get all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            self.image_files.extend([f for f in os.listdir(image_dir_path) 
                                   if f.lower().endswith(ext)])
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        SEQUENTIAL PROCESSING: Only one image in memory at a time
        """
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir_path, image_name)
        
        # Check if output file already exists
        output_file = f"{self.output_dir_path}/{self.model_name}/{image_name[:-4]}.json"
        if os.path.exists(output_file):
            return image_name, [], 0, 0
        
        try:
            # Load image using detectron2's read_image
            image_array = read_image(image_path, format="RGB")
            original_height, original_width = image_array.shape[:2]
            
            # Apply only the required transform
            if self.transforms is not None:
                transform = self.transforms.get_transform(image_array)
                image_array = transform.apply_image(image_array)
            
            # Convert to tensor and immediately delete numpy array
            image_tensor = torch.as_tensor(image_array.astype("float32")).permute(2, 0, 1)
            del image_array  # CRITICAL: Delete numpy copy immediately
            
            return image_name, [image_tensor], original_height, original_width
            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            return image_name, [], 0, 0


def main():
    args = parse_args()
    init_distributed_mode(args)

    model_name = args.model_name
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path

    os.makedirs(output_dir_path, exist_ok=True)

    # Create the required transform only
    transforms = T.ResizeShortestEdge([800, 800], 1333)
    
    # Create dataset
    image_dataset = CustomImageDataset(
        image_dir_path, 
        output_dir_path,
        transforms=transforms, 
        model_name=model_name
    )
    
    # SEQUENTIAL PROCESSING ONLY - No parallel processing at all
    if args.local_rank > 0:
        sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
    else:
        sampler = None

    # CRITICAL: batch_size=1, num_workers=0 for sequential processing
    image_dataloader = DataLoader(
        image_dataset, 
        batch_size=1,           # Only 1 image at a time 
        num_workers=0,          # No parallel data loading
        sampler=sampler, 
        shuffle=False           # Sequential order
    )

    run_inference(args, dataloader=image_dataloader)


if __name__ == "__main__":
    main()