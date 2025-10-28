# (grand_env_4) python infer_per_iNatAg.py --input_dir "/mnt/e/Desktop/AgML/datasets_sorted/iNatAg_subset" --output_dir "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/iNatAg_subset" --model_name 'eva-02-01'

import argparse
import json
import numpy as np
import torch
import torch.nn.parallel
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data.detection_utils import read_image
from detectron2.utils.visualizer import _create_text_labels
from torch.utils.data import DataLoader, DistributedSampler, Dataset
from detectron2.data import MetadataCatalog
import detectron2.data.transforms as T
from copy import copy
from rle_format import mask_to_rle_pytorch, coco_encode_rle
from tqdm import tqdm
from multiprocessing import Process, Queue, Pool
import time
import os
import glob

eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
                                     "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
eva02_L_lvis_sys_config_path = ("projects/ViTDet/configs/eva2_mim_to_lvis/"
                                "eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py")

eva02_L_lvis_sys_o365_ckpt_path = ("./eva_1_FT_0/model_0029999.pth")
eva02_L_lvis_sys_ckpt_path = ("./eva_2_FT_2/model_0099999.pth")

opts_1 = f"train.init_checkpoint={eva02_L_lvis_sys_o365_ckpt_path}"
opts_2 = f"train.init_checkpoint={eva02_L_lvis_sys_ckpt_path}"


def scan_single_dataset(args):
    """Scan a single dataset folder - designed to run in parallel."""
    dataset_name, root_dir, output_dir, model_name, image_extensions = args
    
    dataset_path = os.path.join(root_dir, dataset_name)
    dataset_images = []
    
    # Get all images in this dataset folder
    for file_name in os.listdir(dataset_path):
        file_path = os.path.join(dataset_path, file_name)
        
        # Check if it's an image file
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_name)
            if ext.lower() in image_extensions:
                # Check if output JSON already exists
                output_json_path = os.path.join(
                    output_dir, 
                    dataset_name, 
                    model_name, 
                    os.path.splitext(file_name)[0] + ".json"
                )
                
                if not os.path.exists(output_json_path):
                    dataset_images.append((dataset_name, file_path, file_name))
    
    return dataset_images


class MultiDatasetImageDataset(Dataset):
    """Dataset that handles multiple dataset folders with images."""
    
    def __init__(self, root_dir, image_reader, output_dir, transforms=None, model_name="eva-02-01"):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.image_reader = image_reader
        self.transforms = transforms
        self.model_name = model_name
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Scan all dataset folders and collect images
        self.image_data = []  # List of (dataset_name, image_path, image_name)
        self._scan_datasets()
        
    def _scan_datasets(self):
        """Scan all dataset folders and collect image information - PARALLEL VERSION."""
        # Get all subdirectories in root_dir
        dataset_folders = [d for d in os.listdir(self.root_dir) 
                          if os.path.isdir(os.path.join(self.root_dir, d))]
        
        print(f"\n{'='*60}")
        print(f"Found {len(dataset_folders)} dataset folders to scan")
        print(f"Using parallel scanning with {os.cpu_count()} CPU cores")
        print(f"{'='*60}\n")
        
        # Prepare arguments for parallel processing
        scan_args = [
            (dataset_name, self.root_dir, self.output_dir, self.model_name, self.image_extensions)
            for dataset_name in sorted(dataset_folders)
        ]
        
        # Use multiprocessing Pool to scan datasets in parallel
        with Pool(processes=os.cpu_count()) as pool:
            results = list(tqdm(
                pool.imap(scan_single_dataset, scan_args),
                total=len(dataset_folders),
                desc="Scanning datasets"
            ))
        
        # Flatten results from all datasets
        for dataset_images in results:
            self.image_data.extend(dataset_images)
        
        print(f"\n{'='*60}")
        print(f"Scan complete!")
        print(f"Found {len(self.image_data)} images to process across {len(dataset_folders)} datasets")
        print(f"{'='*60}\n")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        dataset_name, image_path, image_name = self.image_data[idx]
        
        try:
            # Read image
            image = self.image_reader(image_path, format="BGR")
            original_height, original_width = image.shape[:2]
            
            # Apply transforms
            if self.transforms is not None:
                image = self.transforms.get_transform(image).apply_image(image)
            
            # Convert to tensor
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            
            return dataset_name, image_name, image, original_height, original_width
            
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return dataset_name, image_name, None, 0, 0


def parse_args():
    parser = argparse.ArgumentParser(description="EVA-02 Multi-Dataset Object Detection")

    parser.add_argument("--model_name", required=True,
                        help="Either eva-02-01 or eva-02-02")
    parser.add_argument("--input_dir", required=True,
                        help="Root directory containing dataset folders")
    parser.add_argument("--output_dir", required=True,
                        help="Root output directory (will mirror dataset structure)")
    parser.add_argument("--opts", required=False, default="")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=8, type=int)
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    return args


def json_serializable(data):
    if isinstance(data, np.float32):
        return round(float(data), 2)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data


def setup(config_file, opts):
    cfg = LazyConfig.load(config_file)
    cfg = LazyConfig.apply_overrides(cfg, [opts])
    
    # Disable federated loss for custom dataset
    num_classes = 30
    
    # Update all box_predictor stages
    if hasattr(cfg.model.roi_heads, 'box_predictor'):
        for i in range(len(cfg.model.roi_heads.box_predictor)):
            cfg.model.roi_heads.box_predictor[i].num_classes = num_classes
            cfg.model.roi_heads.box_predictor[i].get_fed_loss_cls_weights = lambda: None
    
    return cfg


def filter_predictions_with_confidence(predictions, confidence_threshold=0.5):
    if "instances" in predictions:
        preds = predictions["instances"]
        keep_idxs = (preds.scores > confidence_threshold).cpu()
        predictions = copy(predictions)
        predictions["instances"] = preds[keep_idxs]
    return predictions


def post_processing(q, args, model_name, metadata):
    """Worker process for post-processing predictions."""
    while True:
        data = q.get()
        if data == "STOP":
            break

        dataset_name, image_name, predictions = data

        bboxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy().tolist() if predictions.has("scores") else None
        labels = _create_text_labels(
            predictions.pred_classes.tolist() if predictions.has("pred_classes") else None,
            None, 
            metadata.get("thing_classes", None)
        )
        uncompressed_mask_rles = mask_to_rle_pytorch(predictions.pred_masks)

        all_image_predictions = []
        for j, box in enumerate(bboxes):
            prediction = {}
            prediction['bbox'] = [round(float(b), 2) for b in box]
            prediction['sam_mask'] = coco_encode_rle(uncompressed_mask_rles[j])
            prediction['score'] = round(scores[j], 2)
            prediction['label'] = labels[j]
            all_image_predictions.append(prediction)

        all_data = {}
        all_data[image_name] = {}
        all_data[image_name][model_name] = [
            {k: json_serializable(v) for k, v in prediction.items()} 
            for prediction in all_image_predictions
        ]

        # Create output directory for this dataset
        output_dataset_dir = os.path.join(args.output_dir, dataset_name, model_name)
        os.makedirs(output_dataset_dir, exist_ok=True)

        # Write JSON file
        output_path = os.path.join(output_dataset_dir, os.path.splitext(image_name)[0] + ".json")
        with open(output_path, 'w') as f:
            json.dump(all_data, f)


def run_inference(args, dataloader, confidence_threshold=0.05, num_post_processing_workers=8):
    q = Queue()
    workers = []

    # Create metadata
    metadata = MetadataCatalog.get("Agri_val")
    metadata.thing_classes = [
        "almond", "apple", "avocado", "banana", "broccoli", "cabbage", "capsicum", "cotton",
        "cucumber", "flower", "grape", "kiwi", "leaf", "lemon", "lettuce", "maize_tassel",
        "mango", "orange", "pineapple", "plant", "potato", "pumpkin", "rice_panicles", "rockmelon",
        "sorghum_head", "soybean_pod", "strawberry", "tomato", "weed", "wheat_head"
    ]

    model_name = args.model_name

    # Start worker processes
    for _ in range(num_post_processing_workers):
        p = Process(target=post_processing, args=(q, args, model_name, metadata))
        p.start()
        workers.append(p)

    # Initialize config
    cfg = setup(
        eva02_L_lvis_sys_o365_config_path if args.model_name == "eva-02-01" else eva02_L_lvis_sys_config_path,
        opts_1 if args.model_name == "eva-02-01" else opts_2
    )

    # Initialize model
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
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing images"):
            dataset_names, image_names, images, image_heights, image_widths = batch
            
            for i in range(len(images)):
                if images[i] is None:  # Skip if image failed to load
                    continue
                    
                inputs = [{
                    "image": images[i],
                    "height": int(image_heights[i]),
                    "width": int(image_widths[i]),
                }]
                
                preds = model(inputs)[0]
                preds = filter_predictions_with_confidence(preds, confidence_threshold)
                preds = preds["instances"].to("cpu")
                
                # Add dataset_name to the queue data
                q.put((dataset_names[i], image_names[i], preds))

    # Send stop signal to workers
    for _ in range(num_post_processing_workers):
        q.put("STOP")

    # Wait for all workers to finish
    for p in workers:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('\033[92m' + f"---- {model_name} Time taken: {elapsed_time:.2f} seconds ----" + '\033[0m')


def detection_collate_fn(batch):
    """Custom collate function for batching."""
    dataset_names, image_names, images, heights, widths = zip(*batch)
    return list(dataset_names), list(image_names), list(images), list(heights), list(widths)


def init_distributed_mode(args):
    """Initialize distributed training if needed."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.local_rank)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()


def main():
    args = parse_args()
    init_distributed_mode(args)

    model_name = args.model_name
    input_dir = args.input_dir
    output_dir = args.output_dir
    batch_size_per_gpu = args.batch_size_per_gpu

    os.makedirs(output_dir, exist_ok=True)

    # Create dataset
    transforms = T.ResizeShortestEdge([800, 800], 1333)
    image_dataset = MultiDatasetImageDataset(
        input_dir, 
        read_image, 
        output_dir,
        transforms=transforms, 
        model_name=model_name
    )
    
    if hasattr(args, 'distributed') and args.distributed:
        sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
    else:
        sampler = None
    
    image_dataloader = DataLoader(
        image_dataset, 
        batch_size=batch_size_per_gpu,
        num_workers=2, 
        sampler=sampler, 
        shuffle=(sampler is None), 
        collate_fn=detection_collate_fn
    )

    run_inference(args, dataloader=image_dataloader)


if __name__ == "__main__":
    main()