# (grand_env_4) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_1_inference/5_eva_02$
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --model_name 'eva-02-01' --local_rank 0 --batch_size_per_gpu 4
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --model_name 'eva-02-02' --local_rank 0 --batch_size_per_gpu 4
# This code allow for batch processing (all images in the datset has to have the same aspect ratio). However, testing shows that RTX3090 is compute-bounded (no differance in speed between batch 1 and 16!)
import argparse
import json
import numpy as np
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
from multiprocessing import Process, Queue
import time

eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
                                     "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
eva02_L_lvis_sys_config_path = ("projects/ViTDet/configs/eva2_mim_to_lvis/"
                                "eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py")

eva02_L_lvis_sys_o365_ckpt_path = ("eva02_L_lvis_sys_o365.pth")
eva02_L_lvis_sys_ckpt_path = ("eva02_L_lvis_sys.pth")

opts_1 = f"train.init_checkpoint={eva02_L_lvis_sys_o365_ckpt_path}"
opts_2 = f"train.init_checkpoint={eva02_L_lvis_sys_ckpt_path}"


def parse_args():
    parser = argparse.ArgumentParser(description="EVA-02 Object Detection")

    parser.add_argument("--model_name", required=True,
                        help="Either eva-02-01 or eva-02-02. Specifying eva-02-01 runs eva02_L_lvis_sys_o365_ckpt "
                             "while eva-02-02 runs eva02_L_lvis_sys")
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--opts", required=False, default="")

    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=16, type=int, 
                        help="Batch size per GPU (can be >1 since images have same aspect ratio)")
    parser.add_argument('--world_size', default=6, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
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


def post_processing(q, args, model_name, metadata):
    while True:
        data = q.get()
        if data == "STOP":
            break

        image_name, predictions = data

        bboxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores.numpy().tolist() if predictions.has("scores") else None
        labels = _create_text_labels(predictions.pred_classes.tolist() if predictions.has("pred_classes") else None,
                                     None, metadata.get("thing_classes", None))
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
        all_data[image_name][model_name] = [{k: json_serializable(v) for k, v in prediction.items()} for
                                            prediction in all_image_predictions]

        # Write all_data to a JSON file (file_wise)
        with open(f"{args.output_dir_path}/{model_name}/{image_name[:-4]}.json", 'w') as f:
            json.dump(all_data, f)


def run_inference(args, dataloader, confidence_threshold=0.05, num_post_processing_workers=8):
    q = Queue()
    workers = []

    # Start the worker processes for post-processing
    metadata = MetadataCatalog.get("lvis_v1_val")
    model_name = args.model_name
    os.makedirs(f"{args.output_dir_path}/{model_name}", exist_ok=True)
    for _ in range(num_post_processing_workers):
        p = Process(target=post_processing, args=(q, args, model_name, metadata))
        p.start()
        workers.append(p)

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
    processed_batches = 0
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Processing batches (batch_size={args.batch_size_per_gpu})"):
            # Handle batched data
            image_names, images, image_heights, image_widths = batch_data
            
            # Skip if empty batch
            if len(images) == 0:
                continue
            
            # Prepare batch inputs for the model
            batch_inputs = []
            for i in range(len(image_names)):
                batch_inputs.append({
                    'image': torch.as_tensor(images[i]).to(cfg.train.device),
                    'height': int(image_heights[i]), 
                    'width': int(image_widths[i])
                })
            
            # Run inference on the entire batch
            batch_predictions = model(batch_inputs)
            
            # Process each image's predictions in the batch
            for i, (image_name, predictions) in enumerate(zip(image_names, batch_predictions)):
                # Filter predictions by confidence
                filtered_predictions = filter_predictions_with_confidence(predictions, confidence_threshold)
                instances = filtered_predictions["instances"].to("cpu")
                
                # Send individual image results to post-processing queue
                q.put((image_name, instances))
            
            processed_batches += 1
            if processed_batches % 10 == 0:
                print(f"Processed {processed_batches} batches, {processed_batches * args.batch_size_per_gpu} images")

    # Send stop signal to workers
    for _ in range(num_post_processing_workers):
        q.put("STOP")

    # Wait for all worker processes to finish
    for p in workers:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time
    total_images = processed_batches * args.batch_size_per_gpu
    images_per_second = total_images / elapsed_time if elapsed_time > 0 else 0

    print('\033[92m' + f"---- EVA-02 Batched Inference Complete ----" + '\033[0m')
    print(f"Total time: {elapsed_time:.2f} seconds")
    print(f"Processed: {total_images} images in {processed_batches} batches")
    print(f"Throughput: {images_per_second:.2f} images/second")
    print(f"Batch size: {args.batch_size_per_gpu}")


def custom_collate_fn(batch):
    """Custom collate function to handle batched data properly"""
    image_names = [item[0] for item in batch]
    images = [item[1] for item in batch]
    heights = [item[2] for item in batch]
    widths = [item[3] for item in batch]
    
    # Since all images have the same aspect ratio, we can stack them
    # Filter out any empty entries
    valid_indices = [i for i, img in enumerate(images) if len(img) > 0]
    
    if not valid_indices:
        return [], [], [], []
    
    filtered_names = [image_names[i] for i in valid_indices]
    filtered_images = [images[i] for i in valid_indices]
    filtered_heights = [heights[i] for i in valid_indices]
    filtered_widths = [widths[i] for i in valid_indices]
    
    return filtered_names, filtered_images, filtered_heights, filtered_widths


def main():
    args = parse_args()
    init_distributed_mode(args)

    model_name = args.model_name
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    batch_size_per_gpu = args.batch_size_per_gpu

    os.makedirs(output_dir_path, exist_ok=True)

    print(f"Starting EVA-02 inference with batch size: {batch_size_per_gpu}")
    print(f"Model: {model_name}")
    print(f"Images directory: {image_dir_path}")
    print(f"Output directory: {output_dir_path}")

    # Create dataset
    transforms = T.ResizeShortestEdge([800, 800], 1333)
    image_dataset = CustomImageDataset(image_dir_path, read_image, output_dir_path,
                                       transforms=transforms, model_name=model_name)
    
    if args.local_rank > 0:
        sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
        shuffle = False
    else:
        sampler = None
        shuffle = False

    # Create dataloader with batching support
    image_dataloader = DataLoader(
        image_dataset, 
        batch_size=batch_size_per_gpu,
        num_workers=2, 
        sampler=sampler, 
        shuffle=shuffle,
        collate_fn=custom_collate_fn,  # Custom collate function for proper batching
        pin_memory=True,  # Speed up data transfer to GPU
        drop_last=False   # Don't drop the last incomplete batch
    )

    print(f"Created dataloader with {len(image_dataset)} images")
    print(f"Expected number of batches: {len(image_dataloader)}")

    run_inference(args, dataloader=image_dataloader)


if __name__ == "__main__":
    main()