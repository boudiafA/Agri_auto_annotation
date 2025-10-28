# (grand_env_7) abood@DESKTOP-EKAJBQH:~/groundingLMM/GranD/level_2_inference/3_mdetr$
# python infer.py --image_dir_path '/home/abood/groundingLMM/GranD/images' --output_dir_path '/home/abood/groundingLMM/GranD/output' --blip2_pred_path "/home/abood/groundingLMM/GranD/output/blip2" --llava_pred_path "/home/abood/groundingLMM/GranD/output/llava" --local_rank 0
import argparse
import os.path
import numpy as np
import torch
import traceback
import logging
from torch.utils.data import DataLoader, DistributedSampler
from utils import *
from tqdm import tqdm
from ddp import *
from torch.utils.data._utils.collate import default_collate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mdetr_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.set_grad_enabled(False)


def parse_args():
    parser = argparse.ArgumentParser(description="MDETR Referring Expression")

    parser.add_argument("--image_dir_path", required=True)
    parser.add_argument("--output_dir_path", required=True)

    parser.add_argument("--blip2_pred_path", required=False, default="predictions/blip2")
    parser.add_argument("--llava_pred_path", required=False, default="predictions/llava")

    parser.add_argument("--checkpoint", required=True,
                        help="Specify the checkpoints path if you want to load from a local path.")

    # DDP related parameters
    parser.add_argument("--batch_size_per_gpu", required=False, default=1)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # Add debug level flag
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with verbose logging')

    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    return args


def json_serializable(data):
    if isinstance(data, np.float32):  # if it's a np.float32
        return round(float(data), 2)  # convert to python float
    elif isinstance(data, np.ndarray):  # if it's a np.ndarray
        return data.tolist()  # convert to python list
    else:  # for other types, let it handle normally
        return data


def run_inference(model, image, image_size, nouns, phrases, threshold=0.5):
    """
    Run inference on the image with the given phrases.
    
    Args:
        model: The MDETR model, on GPU
        image: Image tensor (currently on CPU)
        image_size: Original image size (width, height)
        nouns: List of nouns corresponding to phrases
        phrases: List of phrases to locate in the image
        threshold: Confidence threshold for detections
        
    Returns:
        tuple: (image, all_nouns, all_phrase_boxes)
    """
    num_phrases = len(phrases)
    
    # Add validation for empty phrases list
    if num_phrases == 0:
        logger.warning("No phrases found to process")
        return image, [], {}
        
    logger.debug(f"Processing {num_phrases} phrases: {phrases}")
    
    # Check if model is on CUDA
    is_model_cuda = next(model.parameters()).is_cuda
    device = next(model.parameters()).device
    logger.debug(f"Model is on CUDA: {is_model_cuda}, device: {device}")
    
    # Check input image device
    logger.debug(f"Image tensor shape: {image.shape}, original device: {image.device}")
    
    # Move image to the same device as the model
    if is_model_cuda and not image.is_cuda:
        logger.debug(f"Moving image tensor to device: {device}")
        image = image.to(device)
        logger.debug(f"Image tensor now on device: {image.device}")
    
    # Repeat the image for each phrase
    image_b = image.repeat(num_phrases, 1, 1, 1)
    logger.debug(f"Batched image tensor shape: {image_b.shape}, device: {image_b.device}")
    
    try:
        # Propagate through the model
        memory_cache = model(image_b, phrases, encode_and_save=True)
        outputs = model(image_b, phrases, encode_and_save=False, memory_cache=memory_cache)
        
        all_phrase_boxes = {}
        all_nouns = []
        for i in range(num_phrases):
            probas = 1 - outputs['pred_logits'].softmax(-1)[i, :, -1].cpu()
            keep = (probas > threshold).cpu()
            # convert boxes from [0; 1] to image scales
            bboxes_scaled = rescale_bboxes(outputs['pred_boxes'].cpu()[i, keep], image_size)
            bboxes_scaled = bboxes_scaled.cpu().numpy().tolist()
            if bboxes_scaled:
                all_phrase_boxes[phrases[i]] = bboxes_scaled[0]
                all_nouns.append(nouns[i])
                
        logger.debug(f"Found {len(all_phrase_boxes)} valid detections")
        return image, all_nouns, all_phrase_boxes
        
    except Exception as e:
        logger.error(f"Error during model inference: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def run(args, dataloader, use_ddp, model_name="mdetr-re"):
    os.makedirs(f"{args.output_dir_path}/{model_name}", exist_ok=True)

    # First, load the model without pre-trained weights
    try:
        model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB3_refcocog',
                                            pretrained=False, return_postprocessor=True)
        logger.info("Successfully loaded model architecture")
    except Exception as e:
        logger.error(f"Error loading model architecture: {str(e)}")
        return
    
    # Download weights from Zenodo if needed
    checkpoint_path = args.checkpoint
    if checkpoint_path == "None":
        # Use the RefCOCOg EfficientNet-B3 checkpoint from Zenodo
        url = "https://zenodo.org/record/4721981/files/refcocog_EB3_checkpoint.pth?download=1"
        # Define a local path to save the downloaded weights
        checkpoint_path = os.path.join(args.output_dir_path, "refcocog_EB3_checkpoint.pth")
        
        # Download if the file doesn't exist
        if not os.path.exists(checkpoint_path):
            logger.info(f"Downloading checkpoint from {url} to {checkpoint_path}")
            try:
                import urllib.request
                urllib.request.urlretrieve(url, checkpoint_path)
                logger.info("Checkpoint downloaded successfully")
            except Exception as e:
                logger.error(f"Error downloading checkpoint: {str(e)}")
                return
        else:
            logger.info(f"Using cached checkpoint at {checkpoint_path}")
    else:
        logger.info(f"Using provided checkpoint at {checkpoint_path}")
    
    # Load the weights with strict=False to ignore unexpected keys
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint["model"], strict=False)
        logger.info("Checkpoint loaded successfully")
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return

    try:
        model = model.cuda()
        logger.debug(f"Model moved to CUDA device: {next(model.parameters()).device}")
    except Exception as e:
        logger.error(f"Error moving model to CUDA: {str(e)}")
        return
        
    model.eval()
    
    if use_ddp:
        try:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
            logger.info(f"Model wrapped with DDP on device {args.local_rank}")
        except Exception as e:
            logger.error(f"Error setting up DDP: {str(e)}")
            return
    
    # Rest of the function remains the same
    all_data = {}
    for (image_name, image, image_size, captions, nouns, phrases) in tqdm(dataloader):
        image_name = image_name[0]
        image_size = image_size[0]
        captions = captions[0]
        nouns = nouns[0]
        phrases = phrases[0]
        
        # Explicitly log device information
        logger.debug(f"Processing image: {image_name}")
        logger.debug(f"Image device: {image.device}, Model device: {next(model.parameters()).device}")
        
        output_file_name = f"{args.output_dir_path}/{model_name}/{os.path.splitext(image_name)[0]}.json"
        if os.path.exists(output_file_name):
            logger.debug(f"Skipping {image_name}, output already exists")
            continue
            
        # Check if phrases list is empty
        if len(phrases) == 0:
            logger.warning(f"No phrases extracted for image: {image_name}, skipping")
            continue
            
        try:
            # Run inference with proper device management
            _, all_nouns, all_phrase_boxes = run_inference(model, image, image_size, nouns, phrases)
            
            # Process results...
        except Exception as e:
            logger.error(f"Error processing image: {image_name}. Error: {str(e)}")
            continue
            
        try:
            # Log GPU memory before inference
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.debug(f"GPU memory before inference - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
            
            _, all_nouns, all_phrase_boxes = run_inference(model, image, image_size, nouns, phrases)
            
            # Log GPU memory after inference
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                logger.debug(f"GPU memory after inference - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
                
            logger.debug(f"Inference successful, found {len(all_phrase_boxes)} boxes")
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA out of memory while processing image: {image_name}")
                logger.error(f"Image size: {image_size}, Number of phrases: {len(phrases)}")
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 3)
                    logger.error(f"GPU memory - Allocated: {mem_allocated:.2f} GB, Reserved: {mem_reserved:.2f} GB")
                # Try to free some memory
                torch.cuda.empty_cache()
            else:
                logger.error(f"RuntimeError processing image: {image_name}. Error: {str(e)}")
            # Log the full traceback for detailed debugging
            logger.error(traceback.format_exc())
            continue
        except Exception as e:
            logger.error(f"Error processing image: {image_name}. Error: {str(e)}")
            logger.error(traceback.format_exc())
            continue

        all_data[image_name] = {}
        all_data[image_name][model_name] = []
        for j, phrase in enumerate(all_phrase_boxes.keys()):
            noun = all_nouns[j]
            prediction = {
                'bbox': [round(float(b), 2) for b in all_phrase_boxes[phrase]],
                'label': noun,
                'phrase': phrase
            }
            all_data[image_name][model_name].append(prediction)

        try:
            with open(output_file_name, 'w') as f:
                json.dump(all_data, f)
            logger.debug(f"Successfully saved results for {image_name}")
        except Exception as e:
            logger.error(f"Error saving results for {image_name}: {str(e)}")
            
        all_data = {}


def custom_collate_fn(batch):
    """Custom collate function that ensures correct device handling"""
    image_names = [item[0] for item in batch]
    
    # Collect images but don't move to GPU yet (will be done in run_inference)
    images = default_collate([item[1] for item in batch])
    
    image_sizes = [item[2] for item in batch]
    captions = [item[3] for item in batch]
    nouns = [item[4] for item in batch]
    phrases = [item[5] for item in batch]

    return image_names, images, image_sizes, captions, nouns, phrases


def main():
    args = parse_args()
    init_distributed_mode(args)

    # Determine if distributed mode is active
    use_ddp = args.distributed  # This is set inside `init_distributed_mode`

    if not use_ddp:
        logger.info("Running in single GPU mode. Skipping distributed setup.")

    os.makedirs(args.output_dir_path, exist_ok=True)

    # Create dataset
    try:
        image_dataset = CustomImageDataset(
            args.image_dir_path, args.blip2_pred_path, args.llava_pred_path, transform
        )
        logger.info(f"Created dataset with {len(image_dataset)} images")
    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return

    if use_ddp:
        sampler = DistributedSampler(image_dataset, rank=args.local_rank, shuffle=False)
        logger.info(f"Using distributed sampler with rank {args.local_rank}")
    else:
        sampler = None

    try:
        image_dataloader = DataLoader(
            image_dataset,
            batch_size=int(args.batch_size_per_gpu),
            num_workers=4,
            sampler=sampler,
            shuffle=(sampler is None),
            collate_fn=custom_collate_fn
        )
        logger.info(f"Created dataloader with batch size {args.batch_size_per_gpu}")
    except Exception as e:
        logger.error(f"Error creating dataloader: {str(e)}")
        logger.error(traceback.format_exc())
        return

    run(args, image_dataloader, use_ddp)
    logger.info("Inference completed")
    
    # The close method tries to close LMDB environments that don't exist
    # Only call this if you've modified CustomImageDataset to actually use LMDB
    # image_dataset.close() 


if __name__ == "__main__":
    main()