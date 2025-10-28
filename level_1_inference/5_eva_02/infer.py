# (grand_env_4) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_1_inference/5_eva_02$
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --model_name 'eva-02-01' --local_rank 0
# python infer.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --model_name 'eva-02-02' --local_rank 0
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
import os

eva02_L_lvis_sys_o365_config_path = ("projects/ViTDet/configs/eva2_o365_to_lvis/"
                                     "eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py")
eva02_L_lvis_sys_config_path = ("projects/ViTDet/configs/eva2_mim_to_lvis/"
                                "eva2_lvis_cascade_mask_rcnn_vitdet_l_8attn_win32_1536_lrd0p8.py")

def parse_args():
    parser = argparse.ArgumentParser(description="EVA-02 Object Detection")
    parser.add_argument("--model_name", required=True,
                        help="Either eva-02-01 or eva-02-02. Specifying eva-02-01 runs eva02_L_lvis_sys_o365_ckpt "
                             "while eva-02-02 runs eva02_L_lvis_sys")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)
    parser.add_argument("--opts", required=False, default="")
    # DDP Related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()
    return args

# Old check points
# eva02_L_lvis_sys_o365_ckpt_path = ("eva02_L_lvis_sys_o365.pth")
# eva02_L_lvis_sys_ckpt_path = ("eva02_L_lvis_sys.pth")

# # Fine-tuned check points
# eva02_L_lvis_sys_o365_ckpt_path = ("./eva_1_FT_0/model_0029999.pth")
# eva02_L_lvis_sys_ckpt_path = ("./eva_2_FT_2/model_0099999.pth")

# opts_1 = f"train.init_checkpoint={eva02_L_lvis_sys_o365_ckpt_path}"
# opts_2 = f"train.init_checkpoint={eva02_L_lvis_sys_ckpt_path}"

def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


# def setup(config_file, opts):
#     cfg = LazyConfig.load(config_file)
#     cfg = LazyConfig.apply_overrides(cfg, [opts])
#     return cfg

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

        # all_image_predictions = compute_depth(all_image_predictions, image_name[:-4])
        all_data = {}

        all_data[image_name] = {}
        all_data[image_name][model_name] = {}
        all_data[image_name][model_name] = [{k: json_serializable(v) for k, v in prediction.items()} for
                                            prediction in all_image_predictions]

        # Write all_data to a JSON file (file_wise)
        with open(f"{args.output_dir_path}/{model_name}/{os.path.splitext(image_name)[0]}.json", 'w') as f:
            json.dump(all_data, f)


def run_inference(args, dataloader, confidence_threshold=0.05, num_post_processing_workers=8):
    q = Queue()
    workers = []

    # Start the worker processes for post-processing
    # Old LVIS names
    # metadata = MetadataCatalog.get("lvis_v1_val")

    # Option B - Create metadata directly
    metadata = MetadataCatalog.get("Agri_val")
    metadata.thing_classes = ["almond", "apple", "avocado", "banana", "broccoli", "cabbage", "capsicum", "cotton",
        "cucumber", "flower", "grape", "kiwi", "leaf", "lemon", "lettuce", "maize_tassel",
        "mango", "orange", "pineapple", "plant", "potato", "pumpkin", "rice_panicles", "rockmelon",
        "sorghum_head", "soybean_pod", "strawberry", "tomato", "weed", "wheat_head"]

    model_name = args.model_name  # FIXED: Use the actual model name from arguments
    os.makedirs(f"{args.output_dir_path}/{model_name}", exist_ok=True)
    for _ in range(num_post_processing_workers):
        p = Process(target=post_processing, args=(q, args, model_name, metadata))
        p.start()
        workers.append(p)

    opts = f"train.init_checkpoint={args.checkpoint}"
    cfg = setup(eva02_L_lvis_sys_o365_config_path if args.model_name == "eva-02-01" else eva02_L_lvis_sys_config_path, opts)

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
    with torch.no_grad():
        # for (image_name, image, image_height, image_width) in tqdm(dataloader):
        #     if len(image) == 0:
        #         continue
        #     image_name = image_name[0]

        #     inputs = [{'image': image[0], "height": int(image_height), "width": int(image_width)}]
        #     predictions = model(inputs)[0]

        #     predictions = filter_predictions_with_confidence(predictions, confidence_threshold)
        #     predictions = predictions["instances"].to("cpu")

        #     # Push the data to the queue for post-processing by the workers
        #     q.put((image_name, predictions))
        for (image_names, images, image_heights, image_widths) in tqdm(dataloader):
            for i in range(len(images)):
                if images[i] is None:  # means JSON already existed
                    continue
                inputs = [{
                    "image": images[i],
                    "height": int(image_heights[i]),
                    "width": int(image_widths[i]),
                }]
                preds = model(inputs)[0]
                preds = filter_predictions_with_confidence(preds, confidence_threshold)
                preds = preds["instances"].to("cpu")
                q.put((image_names[i], preds))




    # Send stop signal to workers
    for _ in range(num_post_processing_workers):
        q.put("STOP")

    # Wait for all worker processes to finish
    for p in workers:
        p.join()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print('\033[92m' + "---- EVA-1 Time taken: {} seconds ----".format(elapsed_time) + '\033[0m')


def detection_collate_fn(batch):
    names, images, heights, widths = zip(*batch)
    return list(names), list(images), list(heights), list(widths)




def main():
    args = parse_args()
    init_distributed_mode(args)

    model_name = args.model_name
    image_dir_path = args.image_dir_path
    output_dir_path = args.output_dir_path
    batch_size_per_gpu = args.batch_size_per_gpu

    os.makedirs(output_dir_path, exist_ok=True)

    # Create dataset
    transforms = T.ResizeShortestEdge([800, 800], 1333)
    image_dataset = CustomImageDataset(image_dir_path, read_image, output_dir_path,
                                       transforms=transforms, model_name=model_name)
    
    if args.local_rank > 0:
        sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
    else:
        sampler = None
    # num_workers = max(1, os.cpu_count() - 2)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size_per_gpu,
                              num_workers=0, sampler=sampler, shuffle=(sampler is None), collate_fn=detection_collate_fn)

    run_inference(args, dataloader=image_dataloader)


if __name__ == "__main__":
    main()