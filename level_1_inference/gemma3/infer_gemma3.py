import os
import json
import re
import argparse
from pathlib import Path
from tqdm import tqdm
import time
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, BitsAndBytesConfig


@dataclass
class ImageTask:
    """Represents an image processing task"""
    image_path: Path
    class_name: str
    class_description: str
    image_name: str


class ImageDataset(Dataset):
    """Dataset for loading images from a folder"""
    def __init__(self, image_paths):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            return {
                'image': image,
                'path': image_path,
                'success': True
            }
        except Exception as e:
            return {
                'image': None,
                'path': image_path,
                'success': False,
                'error': str(e)
            }


def create_combined_prompt(class_name: str, class_description: str) -> str:
    """Create a single prompt for generating both tags and caption."""
    return (
        f"You are analyzing an agricultural image.\n"
        f"Image class: {class_name}\n"
        f"Class description: {class_description}\n\n"
        f"Please provide TWO outputs:\n\n"
        f"1. TAGS: Generate around 7 descriptive tags for this image.\n"
        f"   Tags should include:\n"
        f"   - Plant name (common or scientific)\n"
        f"   - Crop type\n"
        f"   - Visible elements (leaf, soil, stem, pod, flower, sky, etc.)\n"
        f"   - Growth stage if visible\n"
        f"   - Environmental conditions if visible\n"
        f"   Rules for tags:\n"
        f"   - Use lowercase single words or snake_case terms\n"
        f"   - No spaces inside tags\n"
        f"   - If something is not visible, use 'unknown'\n"
        f"   - Put tags in square brackets, separated by commas\n"
        f"   Example: [corn, leaf, soil, green, field, agriculture, mature_stage]\n\n"
        f"2. CAPTION: Write a descriptive caption of about 3-5 sentences.\n"
        f"   Include these aspects if clearly visible:\n"
        f"   - Crop name and type\n"
        f"   - Growth stage\n"
        f"   - Ground cover and plant density\n"
        f"   - Image perspective (top-down, oblique, side, macro, unknown)\n"
        f"   - Environmental conditions\n"
        f"   - Plant health\n"
        f"   Rules for caption:\n"
        f"   - Use clear, neutral language\n"
        f"   - No speculation - only describe what's visible\n"
        f"   - If something cannot be determined, use 'unknown'\n"
        f"   - Write as natural sentences\n\n"
        f"Format your response as:\n"
        f"TAGS: [tag1, tag2, tag3, ...]\n"
        f"CAPTION: Your descriptive caption here."
    )


def validate_and_extract_combined(response: str) -> tuple:
    """
    Validate and extract both tags and caption from combined response.
    Returns (is_valid, tags_list, caption_text)
    """
    try:
        response = response.strip()
        
        # Extract tags - look for TAGS: followed by brackets
        tags_match = re.search(r'TAGS?\s*:\s*\[(.*?)\]', response, re.IGNORECASE | re.DOTALL)
        if tags_match:
            tags_content = tags_match.group(1)
        else:
            # Fallback: look for any bracketed content
            tags_match = re.search(r'\[(.*?)\]', response, re.DOTALL)
            if tags_match:
                tags_content = tags_match.group(1)
            else:
                tags_content = ""
        
        # Extract caption - look for CAPTION: followed by text
        caption_match = re.search(r'CAPTION\s*:\s*(.+?)(?=\n\s*(?:TAGS|$))', response, re.IGNORECASE | re.DOTALL)
        if caption_match:
            caption_text = caption_match.group(1).strip()
        else:
            # Fallback: try to extract text after the tags
            after_tags = re.split(r'\[.*?\]', response, maxsplit=1)
            if len(after_tags) > 1:
                caption_text = after_tags[1].strip()
                # Remove "CAPTION:" prefix if present
                caption_text = re.sub(r'^CAPTION\s*:\s*', '', caption_text, flags=re.IGNORECASE)
            else:
                caption_text = ""
        
        # Process tags
        tags_raw = [tag.strip() for tag in tags_content.split(',')]
        cleaned_tags = []
        for tag in tags_raw:
            if tag:
                tag = re.sub(r'^["\']|["\']$', '', tag.strip())
                cleaned_tag = tag.lower().replace(' ', '_')
                cleaned_tag = re.sub(r'[^\w_]', '', cleaned_tag)
                
                if cleaned_tag and len(cleaned_tag) > 0:
                    cleaned_tags.append(cleaned_tag)
        
        # Remove duplicate tags while preserving order
        seen = set()
        unique_tags = []
        for tag in cleaned_tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)
        
        # Process caption
        caption = caption_text.strip()
        
        # Remove quotes if wrapped
        if (caption.startswith('"') and caption.endswith('"')) or \
           (caption.startswith("'") and caption.endswith("'")):
            caption = caption[1:-1].strip()
        
        # Validate tags (3-12 tags)
        tags_valid = 3 <= len(unique_tags) <= 12
        
        # Validate caption (reasonable length and sentence count)
        caption_valid = False
        if len(caption) >= 20 and len(caption) <= 800:
            sentences = re.split(r'[.!?]+', caption)
            sentences = [s.strip() for s in sentences if s.strip()]
            # Accept 2-8 sentences, or if length is reasonable (50-400 chars)
            if 2 <= len(sentences) <= 8 or (50 <= len(caption) <= 400):
                caption_valid = True
        
        # Both must be valid
        is_valid = tags_valid and caption_valid
        
        return is_valid, unique_tags if tags_valid else [], caption if caption_valid else ""
        
    except Exception:
        return False, [], ""


def check_structured_output_exists(output_dir: Path, class_name: str, image_name: str) -> bool:
    """
    Check if structured outputs already exist for this image.
    Returns True if ALL formats exist, False otherwise.
    """
    base_name = os.path.splitext(image_name)[0]
    class_path = output_dir
    
    required_files = [
        class_path / "blip2" / f"{base_name}.json",
        class_path / "llava" / f"{base_name}.json",
        class_path / "ram" / f"{base_name}.json",
        class_path / "tag2text" / f"{base_name}.json",
        class_path / "landmark" / f"{base_name}.json"  # Added landmark
    ]
    
    return all(f.exists() for f in required_files)


def scan_class_folder(args_tuple):
    """
    Scan a single class folder for missing files.
    This function is used for parallel scanning.
    Returns: (class_name, list_of_missing_image_tasks)
    """
    class_folder, class_name, class_description, output_dir = args_tuple
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    all_image_files = [
        f for f in class_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    # Check which images need processing
    missing_tasks = []
    for img_file in all_image_files:
        if not check_structured_output_exists(output_dir, class_name, img_file.name):
            task = ImageTask(
                image_path=img_file,
                class_name=class_name,
                class_description=class_description,
                image_name=img_file.name
            )
            missing_tasks.append(task)
    
    return class_name, len(all_image_files), missing_tasks


def get_model_response(image_path, prompt_text, model, processor, args):
    """Get response from Gemma 3 model for a given prompt."""
    
    # Prepare messages in Gemma 3 format
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt_text}
            ]
        }
    ]
    
    # Preprocess inputs
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    with torch.inference_mode():
        generation = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False
        )
    
    # Decode the result
    input_len = inputs["input_ids"].shape[-1]
    response = processor.decode(generation[0][input_len:], skip_special_tokens=True)
    
    return response.strip()


def save_structured_outputs(output_dir: Path, class_name: str, image_name: str, tags: list, caption: str):
    """Save annotations in structured format (blip2, llava, ram, tag2text, landmark)"""
    base_name = os.path.splitext(image_name)[0]
    class_path = output_dir / class_name
    
    # Create subfolders
    for subfolder in ["blip2", "llava", "ram", "tag2text", "landmark"]:
        (class_path / subfolder).mkdir(parents=True, exist_ok=True)
    
    # BLIP2 (empty)
    blip2_data = {image_name: {"blip2": ""}}
    with open(class_path / "blip2" / f"{base_name}.json", 'w', encoding='utf-8') as f:
        json.dump(blip2_data, f, indent=2, ensure_ascii=False)
    
    # LLaVA (caption)
    llava_data = {image_name: {"llava": caption}}
    with open(class_path / "llava" / f"{base_name}.json", 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)
    
    # RAM (tags with scores)
    ram_data = {image_name: {"ram": {"tags": tags, "scores": [1.0] * len(tags)}}}
    with open(class_path / "ram" / f"{base_name}.json", 'w', encoding='utf-8') as f:
        json.dump(ram_data, f, indent=2, ensure_ascii=False)
    
    # Tag2Text (empty)
    tag2text_data = {image_name: {"tag2text": {"tags": [], "scores": []}}}
    with open(class_path / "tag2text" / f"{base_name}.json", 'w', encoding='utf-8') as f:
        json.dump(tag2text_data, f, indent=2, ensure_ascii=False)
    
    # Landmark (category information)
    landmark_data = {
        image_name: {
            "landmark": {
                "category": "Outdoor scene",
                "fine_category": "Natural landscape"
            }
        }
    }
    with open(class_path / "landmark" / f"{base_name}.json", 'w', encoding='utf-8') as f:
        json.dump(landmark_data, f, indent=2, ensure_ascii=False)


def format_time(seconds):
    """Format seconds into human-readable time string"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        mins = seconds / 60
        return f"{mins:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def scan_all_classes(dataset_root: Path, class_descriptions: Dict, output_dir: Path, num_workers: int = None):
    """
    Scan all class folders in parallel to identify missing files.
    Can handle both:
    - Parent directory containing multiple class folders
    - Single class folder containing images directly
    Returns: List of ImageTask objects for missing files
    """
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)  # Leave one core free
    
    print(f"Scanning dataset with {num_workers} workers...")
    
    # Check if this is a single class folder (contains image files) or parent directory
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    has_images = any(f.is_file() and f.suffix.lower() in image_extensions 
                     for f in dataset_root.iterdir() if not f.name.startswith('.'))
    
    if has_images:
        # This is a single class folder
        print("Detected single class folder (contains images directly)")
        class_name = dataset_root.name
        
        if class_name not in class_descriptions:
            print(f"⚠️  Warning: Class '{class_name}' not found in descriptions JSON")
            print(f"Available classes: {len(class_descriptions)}")
            return [], {}
        
        scan_args = [(
            dataset_root,
            class_name,
            class_descriptions[class_name],
            output_dir
        )]
    else:
        # This is a parent directory with class folders
        print("Detected parent directory (contains class folders)")
        class_folders = sorted([f for f in dataset_root.iterdir() if f.is_dir()])
        
        # Prepare arguments for parallel scanning
        scan_args = []
        for class_folder in class_folders:
            class_name = class_folder.name
            if class_name in class_descriptions:
                scan_args.append((
                    class_folder,
                    class_name,
                    class_descriptions[class_name],
                    output_dir
                ))
    
    if not scan_args:
        print("⚠️  No valid class folders found!")
        return [], {}
    
    # Scan in parallel
    all_missing_tasks = []
    scan_summary = {}
    
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(scan_class_folder, scan_args),
            total=len(scan_args),
            desc="Scanning classes"
        ))
    
    # Collect results
    for class_name, total_images, missing_tasks in results:
        scan_summary[class_name] = {
            'total': total_images,
            'missing': len(missing_tasks),
            'complete': total_images - len(missing_tasks)
        }
        all_missing_tasks.extend(missing_tasks)
    
    return all_missing_tasks, scan_summary


def process_all_tasks(tasks: List[ImageTask], model, processor, args, output_dir):
    """Process all missing image tasks using combined prompt"""
    
    if not tasks:
        print("\n✓ All images already processed!")
        return
    
    print(f"\nProcessing {len(tasks)} missing images with combined prompt...")
    
    # Track statistics
    stats = {
        'successful': 0,
        'failed': 0
    }
    
    start_time = time.time()
    
    # Process all tasks with progress bar
    with tqdm(total=len(tasks), desc="Processing images") as pbar:
        for task in tasks:
            try:
                # Create combined prompt for this specific class
                combined_prompt = create_combined_prompt(task.class_name, task.class_description)
                
                tags = None
                caption = None
                
                # Generate combined response (with retry)
                for attempt in range(2):
                    try:
                        response = get_model_response(task.image_path, combined_prompt, model, processor, args)
                        valid, tags, caption = validate_and_extract_combined(response)
                        
                        if valid:
                            stats['successful'] += 1
                            break
                        
                        if attempt == 1:  # Last attempt
                            tags = []
                            caption = ""
                            stats['failed'] += 1
                            tqdm.write(f"Failed to generate valid output for {task.image_name}")
                    except Exception as e:
                        if attempt == 1:
                            tags = []
                            caption = ""
                            stats['failed'] += 1
                            tqdm.write(f"Error processing {task.image_name}: {e}")
                        time.sleep(0.1)
                
                # Save to structured output if we have both
                if tags and caption:
                    save_structured_outputs(output_dir, task.class_name, task.image_name, tags, caption)
                
                # Update progress bar with stats
                pbar.set_postfix({
                    'success': stats['successful'],
                    'failed': stats['failed']
                })
                pbar.update(1)
                
            except Exception as e:
                tqdm.write(f"Error processing {task.image_name}: {e}")
                stats['failed'] += 1
                pbar.update(1)
    
    elapsed = time.time() - start_time
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"Processing complete in {format_time(elapsed)}")
    print(f"  Successful: {stats['successful']}/{len(tasks)} ({100*stats['successful']/len(tasks):.1f}%)")
    print(f"  Failed: {stats['failed']}/{len(tasks)} ({100*stats['failed']/len(tasks):.1f}%)")
    if stats['successful'] > 0:
        print(f"  Average time per image: {elapsed/stats['successful']:.2f}s")
    print(f"{'='*60}")


def main(args):
    start_overall = time.time()
    
    # Load class descriptions
    print(f"Loading class descriptions from {args.descriptions_json}")
    with open(args.descriptions_json, 'r', encoding='utf-8') as f:
        class_descriptions = json.load(f)
    print(f"Loaded {len(class_descriptions)} class descriptions\n")
    
    # Create output directory
    output_dir = Path(args.output_dir_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Scan all classes in parallel
    print("="*60)
    print("PHASE 1: SCANNING DATASET")
    print("="*60)
    
    dataset_root = Path(args.image_dir_path)
    missing_tasks, scan_summary = scan_all_classes(
        dataset_root, 
        class_descriptions, 
        output_dir,
        num_workers=args.scan_workers
    )
    
    # Print scan summary
    print(f"\n{'='*60}")
    print("SCAN SUMMARY")
    print(f"{'='*60}")
    
    total_images = sum(s['total'] for s in scan_summary.values())
    total_complete = sum(s['complete'] for s in scan_summary.values())
    total_missing = sum(s['missing'] for s in scan_summary.values())
    
    print(f"Total classes: {len(scan_summary)}")
    print(f"Total images: {total_images}")
    
    if total_images == 0:
        print("\n⚠️  ERROR: No images found!")
        print("Please check:")
        print(f"  1. Path exists: {dataset_root}")
        print(f"  2. Path contains images or class folders with images")
        print(f"  3. Class name '{dataset_root.name}' exists in descriptions JSON")
        return
    
    print(f"Already complete: {total_complete} ({100*total_complete/total_images:.1f}%)")
    print(f"Missing: {total_missing} ({100*total_missing/total_images:.1f}%)")
    
    # Show classes with missing files
    classes_with_missing = {k: v for k, v in scan_summary.items() if v['missing'] > 0}
    if classes_with_missing:
        print(f"\nClasses requiring processing: {len(classes_with_missing)}")
        for class_name, info in sorted(classes_with_missing.items(), key=lambda x: x[1]['missing'], reverse=True)[:10]:
            print(f"  {class_name}: {info['missing']}/{info['total']} missing")
        if len(classes_with_missing) > 10:
            print(f"  ... and {len(classes_with_missing) - 10} more classes")
    
    print(f"{'='*60}\n")
    
    if not missing_tasks:
        print("✓ All images already processed! Nothing to do.")
        return
    
    # Phase 2: Load model and process missing images
    print("="*60)
    print("PHASE 2: LOADING MODEL")
    print("="*60)
    
    print("Loading Gemma 3 model...")
    
    # Configure 4-bit quantization for better efficiency
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Load model with quantization
    model = Gemma3ForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()
    
    # Load processor with fast tokenizer
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        use_fast=True
    )
    
    print("✓ Model and processor loaded successfully!\n")
    
    # Phase 3: Process all missing tasks
    print("="*60)
    print("PHASE 3: PROCESSING IMAGES")
    print("="*60)
    
    process_all_tasks(missing_tasks, model, processor, args, output_dir)
    
    total_time = time.time() - start_overall
    print(f"\n✓ All processing complete in {format_time(total_time)}!")
    print(f"Structured outputs saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized Gemma 3 annotation script with combined prompt and landmark output"
    )
    parser.add_argument("--image_dir_path", type=str, required=True,
                       help="Path to images organized by class folders")
    parser.add_argument("--descriptions_json", type=str, default="./species_discription_cleaned.json",
                       help="Path to JSON file with class descriptions")
    parser.add_argument("--output_dir_path", type=str, required=True,
                       help="Output directory for structured annotations (blip2/llava/ram/tag2text/landmark)")
    
    # Model arguments
    parser.add_argument("--model-path", type=str, default="google/gemma-3-12b-it",
                       help="Path or name of Gemma 3 model")
    
    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=300,
                       help="Maximum number of new tokens to generate (increased for combined output)")
    
    # Scanning parameters
    parser.add_argument("--scan-workers", type=int, default=None,
                       help="Number of workers for parallel scanning (default: CPU count - 1)")
    
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()