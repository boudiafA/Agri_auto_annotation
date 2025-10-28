# (grand_env_2) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_1_inference/2_depth_maps
# python infer.py --image_dir_path '/home/abood/groundingLMM/GranD/images' --output_dir_path '/home/abood/groundingLMM/GranD/output' --model_weights '/home/abood/groundingLMM/GranD/checkpoints/dpt_beit_large_512.pt' --local_rank 0

import torch
from tqdm import tqdm
import argparse
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from midas.model_loader import default_models, load_model
from ddp import *
import utils
import time
import os

first_execution = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images.")
    parser.add_argument("--output_dir_path", required=True,
                        help="Path to the output directory to store the predictions.")

    parser.add_argument('-m', '--model_weights', default=None,
                        help='Path to the trained weights of model')
    parser.add_argument('-t', '--model_type', default='dpt_beit_large_512', help='Model type')
    parser.add_argument('-s', '--side', action='store_true', default=True,
                        help='Output images contain RGB and depth images side by side')

    parser.add_argument("--opts", required=False, default="")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=4)
    parser.add_argument('--world_size', default=4, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    return args


def process_batch(model, batch_samples, target_sizes):
    """
    Run the inference and interpolate for a batch of images.

    Args:
        model: the model used for inference
        batch_samples: batch of images fed into the neural network
        target_sizes: list of target sizes (width, height) for each image in the batch

    Returns:
        list of predictions, one for each image in the batch
    """
    global first_execution

    if first_execution:
        batch_size, channels, height, width = batch_samples.shape
        print(f"    Batch size: {batch_size}, Input resized to {width}x{height} before entering the encoder")
        first_execution = False

    # Make sure the batch is on the same device as the model
    batch_samples = batch_samples.to(next(model.parameters()).device)

    # Forward pass for the entire batch
    batch_predictions = model.forward(batch_samples)
    
    # Process each prediction in the batch
    predictions = []
    for i in range(batch_predictions.shape[0]):
        prediction = batch_predictions[i:i+1]  # Keep batch dimension for interpolation
        target_size = target_sizes[i]  # Should now be a (width, height) tuple
        
        # Extract width and height
        width, height = target_size[0], target_size[1]
        
        # Convert from (width, height) to (height, width) for interpolate
        target_size_hw = (int(height), int(width))
        
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=target_size_hw,  # (height, width) format for interpolate
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        predictions.append(prediction)
    
    return predictions


def run(output_path, model_path, model_type="dpt_beit_large_512", dataloader=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False, height=None,
                                                square=False)
    
    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])

    # Create output folder
    start_time = time.time()
    model_name = 'midas'
    processed_image_dir = os.path.join(args.output_dir_path, model_name)
    os.makedirs(processed_image_dir, exist_ok=True)
    processed_images = set(os.listdir(processed_image_dir))  # Use set for faster lookup
    
    total_processed = 0
    total_skipped = 0
    
    for batch_data in tqdm(dataloader):
        image_names, batch_images, image_sizes = batch_data
        
        # Convert to lists if they're tensors
        if torch.is_tensor(image_names):
            image_names = image_names.tolist()
        if torch.is_tensor(image_sizes):
            image_sizes = image_sizes.tolist()
        
        # Handle different image_sizes formats
        batch_size = len(image_names)
        
        # Case 1: image_sizes is a list of tuples/lists (width, height) for each image
        if isinstance(image_sizes, list) and len(image_sizes) == batch_size:
            pass  # Already correct format
        # Case 2: image_sizes is a single tuple/list (same size for all images)
        elif isinstance(image_sizes, (tuple, list)) and len(image_sizes) == 2:
            # Check if it's [tensor([w,w,w,w]), tensor([h,h,h,h])] format
            if torch.is_tensor(image_sizes[0]) and torch.is_tensor(image_sizes[1]):
                # Convert to list of (width, height) tuples
                widths = image_sizes[0].cpu().numpy()
                heights = image_sizes[1].cpu().numpy()
                image_sizes = [(int(widths[i]), int(heights[i])) for i in range(batch_size)]
            else:
                # Regular case: replicate single size for all images
                image_sizes = [image_sizes] * batch_size
        # Case 3: image_sizes is a tensor with shape [batch_size, 2]
        elif hasattr(image_sizes, 'shape') and len(image_sizes.shape) == 2:
            image_sizes = [tuple(size) for size in image_sizes]
        # Case 4: image_sizes is something else - use default
        else:
            print(f"Warning: Unexpected image_sizes format: {type(image_sizes)}, using default (512, 512)")
            image_sizes = [(512, 512)] * batch_size
        
        # Filter out already processed images
        batch_to_process = []
        names_to_process = []
        sizes_to_process = []
        
        for i, image_name in enumerate(image_names):
            output_file_name = image_name[:-4]  # Remove file extension
            if output_file_name not in processed_images:
                batch_to_process.append(i)
                names_to_process.append(image_name)
                sizes_to_process.append(image_sizes[i])
            else:
                total_skipped += 1
        
        if not batch_to_process:
            continue  # Skip this batch if all images are already processed
        
        # Extract only the images that need processing
        batch_images_to_process = batch_images[batch_to_process]
        
        # Compute depth maps for the batch
        with torch.no_grad():
            predictions = process_batch(model, batch_images_to_process, sizes_to_process)

        # Save outputs
        if output_path is not None:
            for i, (image_name, prediction) in enumerate(zip(names_to_process, predictions)):
                output_file_name = image_name[:-4]  # Remove file extension
                filename = os.path.join(processed_image_dir, output_file_name)
                
                # Save the prediction
                # utils.write_pfm(filename + ".pfm", prediction.astype(np.float32))
                utils.write_jpeg(filename, prediction.astype(np.float32))
                
                # Add to processed set to avoid reprocessing
                processed_images.add(output_file_name)
                total_processed += 1

    print("Finished")
    print(f"Total images processed: {total_processed}")
    print(f"Total images skipped (already processed): {total_skipped}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('\033[92m' + "---- Midas Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


if __name__ == "__main__":
    args = parse_args()

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # Set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    init_distributed_mode(args)
    image_dir_path = args.image_dir_path

    # Set up output paths
    output_dir_path = args.output_dir_path
    Path(output_dir_path).mkdir(parents=True, exist_ok=True)

    batch_size_per_gpu = args.batch_size_per_gpu

    # Create dataset
    image_dataset = CustomImageDataset(image_dir_path)

    if torch.distributed.is_initialized():
        distributed_sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
    else:
        distributed_sampler = None

    image_dataloader = DataLoader(image_dataset, batch_size=batch_size_per_gpu, num_workers=4,
                                  sampler=distributed_sampler)

    # Compute depth maps
    run(output_dir_path, args.model_weights, args.model_type, dataloader=image_dataloader)