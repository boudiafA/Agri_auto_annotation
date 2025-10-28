# (grand_env_3) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_1_inference/7_pomp$
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --checkpoint "path/to/checkpoint.pth" --local_rank "0"
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, DistributedSampler
from ddp import *
from models.grit_src.image_dense_captions import setup_grit
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
import os

# Add CenterNet2 to the Python path
centernet_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../environments/CenterNet2'))
sys.path.insert(0, centernet_path)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--checkpoint", required=False, default="grit_b_densecap_objectdet.pth",
                        help="Path to the model checkpoint file")

    parser.add_argument("--opts", required=False, default="")
    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()

    return args

def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return float(data)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data

def dense_pred_dict(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    scores = predictions["instances"].scores if predictions["instances"].has("scores") else None
    object_description = predictions["instances"].pred_object_descriptions.data

    prediction_list = []
    for i in range(len(object_description)):
        bbox = [round(float(a), 2) for a in boxes[i].tensor.cpu().detach().numpy()[0]]
        score = round(float(scores[i]), 2) if scores is not None else None

        prediction_dict = {
            'bbox': bbox,
            'score': score,
            'description': object_description[i],
        }
        prediction_list.append(prediction_dict)

    return prediction_list

def run_inference(args, dataloader):
    # Initialize model with checkpoint path
    dense_caption_demo = setup_grit(device=device, checkpoint_path=args.checkpoint)
    dense_caption_model = dense_caption_demo.predictor.model
    
    # OPTIMIZATION: Model optimizations for faster inference
    dense_caption_model.eval()  # Set to evaluation mode
    if device == "cuda":
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.cuda.empty_cache()  # Clear any existing cache
    
    if args.local_rank == -1 or args.world_size == 1:
        # Single GPU / CPU
        model = dense_caption_demo.predictor.model.to(device)
    else:
        # Distributed Data Parallel
        model = torch.nn.parallel.DistributedDataParallel(dense_caption_demo.predictor.model, device_ids=[args.local_rank])

    start_time = time.time()
    all_data = {}
    model_name = 'grit'
    processed_image_dir = os.path.join(args.output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)
    
    # OPTIMIZATION: Load processed images list once instead of calling os.listdir() repeatedly
    processed_images = set(os.listdir(processed_image_dir))
    
    # OPTIMIZATION: Batch process to reduce I/O overhead while maintaining structure
    batch_results = {}
    images_processed_in_batch = 0
    
    for batch_data in tqdm(dataloader):
        # Unpack batch data
        image_names, images, heights, widths = batch_data
        
        # CRITICAL FIX: Move entire batch to GPU at once (most efficient)
        if device == "cuda":
            images = images.to(device, non_blocking=True)
        
        # Process each image in the batch
        for i in range(len(image_names)):
            image_name = image_names[i]
            output_file_name = f'{os.path.splitext(image_name)[0]}.json'
            
            # Skip if already processed (using pre-loaded set for O(1) lookup)
            if output_file_name in processed_images:
                continue
            
            inputs = {"image": images[i], "height": int(heights[i]), "width": int(widths[i])}
            
            with torch.no_grad():
                # OPTIMIZATION: Use autocast for mixed precision (faster inference)
                if device == "cuda":
                    with torch.cuda.amp.autocast():
                        predictions = model([inputs])[0]
                else:
                    predictions = model([inputs])[0]
                
                all_data[image_name] = {}
                all_data[image_name][model_name] = dense_pred_dict(predictions)

            # OPTIMIZATION: Batch file writes to reduce I/O overhead
            batch_results[image_name] = all_data[image_name]
            images_processed_in_batch += 1
            
            # Clear all_data for next iteration (maintaining original structure)
            all_data = {}
        
        # Write batch every 20 images to reduce I/O overhead
        write_frequency = max(20, int(args.batch_size_per_gpu) * 5)
        
        if images_processed_in_batch >= write_frequency:
            # Write all accumulated results
            for img_name, img_data in batch_results.items():
                output_fname = f'{img_name[:-4]}.json'
                output_path = os.path.join(processed_image_dir, output_fname)
                # Maintain exact same file structure as original
                with open(output_path, 'w') as f:
                    json.dump({img_name: img_data}, f)
            
            # Clear batch and reset counter
            batch_results = {}
            images_processed_in_batch = 0
            
            # OPTIMIZATION: Periodic GPU memory cleanup
            if device == "cuda":
                torch.cuda.empty_cache()
    
    # OPTIMIZATION: Write any remaining results in the final batch
    if batch_results:
        for img_name, img_data in batch_results.items():
            output_fname = f'{img_name[:-4]}.json'
            output_path = os.path.join(processed_image_dir, output_fname)
            with open(output_path, 'w') as f:
                json.dump({img_name: img_data}, f)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- Grit Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')

if __name__ == "__main__":
    args = parse_args()
    init_distributed_mode(args)
    
    # OPTIMIZATION: Set number of threads for better CPU utilization
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(8, os.cpu_count()))
    
    image_dir_path = args.image_dir_path

    # set up output paths
    output_dir_path = args.output_dir_path
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    batch_size_per_gpu = args.batch_size_per_gpu

    # Create dataset
    image_dataset = CustomImageDataset(image_dir_path)

    if args.local_rank == -1 or args.world_size == 1:
        # Single GPU / non-distributed
        distributed_sampler = None
    else:
        # Distributed mode
        from torch.utils.data import DistributedSampler
        distributed_sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)

    # OPTIMIZATION: Enhanced DataLoader for better throughput
    image_dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size_per_gpu,
        num_workers=min(8, os.cpu_count() // 2),  # Optimal worker count
        sampler=distributed_sampler,
        shuffle=(distributed_sampler is None),
        pin_memory=(device == "cuda"),  # Pin memory only if using GPU
        persistent_workers=True if min(8, os.cpu_count() // 2) > 0 else False,  # Reuse workers
        prefetch_factor=2  # Prefetch data for better pipeline
    )

    run_inference(args, dataloader=image_dataloader)