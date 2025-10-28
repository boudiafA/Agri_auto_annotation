import argparse
import os
import json
from tqdm import tqdm
from prompt_template import get_simple_scene_graph
from in_context_examples import get_level_3_chat_prompt
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc

def parse_args():
    parser = argparse.ArgumentParser(description="Level-3 Dense Captioning with Llama 3.1")

    parser.add_argument("--image_names_txt_path", required=True,
                        help="Path to the text file listing image names to process.")
    parser.add_argument("--model_path", required=False,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Path to Llama model or HF model ID")
    parser.add_argument("--level_2_dir_path", required=False,
                        default="predictions/level-2-processed_labelled",
                        help="Directory containing Level-2 processed JSON files.")
    parser.add_argument("--output_directory_path", required=False,
                        default="predictions/level-3-llama3-8B",
                        help="Directory to write Level-3 captions.")
    parser.add_argument("--batch_size", required=False, type=int,
                        default=4, help="Batch size for inference.")
    parser.add_argument("--quantization", type=str, default="8bit", 
                        choices=["none", "8bit", "4bit"], 
                        help="Quantization method for memory efficiency.")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Maximum new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for nucleus sampling.")
    parser.add_argument("--max_length", type=int, default=4096 ,
                        help="Maximum context length for tokenization.")

    return parser.parse_args()


def setup_model_and_tokenizer(model_path, quantization="8bit"):
    """Initialize the Llama 3.1 model and tokenizer with GPU compatibility"""
    print(f"Loading Llama 3.1 model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # --- REVISION: Critical settings for batch processing ---
    # 1. Set pad_token if not set. Many instruction-tuned models don't have one.
    #    Using the EOS token for padding is a common and effective practice.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 2. Set padding side to left for batch generation.
    #    This ensures that all prompts are aligned to the right, and generation can
    #    start consistently after the prompt for every item in the batch.
    #    Example: [PAD][PAD][prompt_A], [prompt_B_longer]
    tokenizer.padding_side = "left"
    # --------------------------------------------------------
    
    # Check bfloat16 support for better performance on modern GPUs
    compute_dtype = torch.bfloat16
    if not torch.cuda.is_bf16_supported():
        print("WARNING: BFloat16 not supported, falling back to Float16")
        compute_dtype = torch.float16
    
    # Configure model loading arguments
    model_kwargs = {"device_map": "auto"}
    
    if quantization == "8bit":
        model_kwargs["load_in_8bit"] = True
    elif quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
    else: # "none"
        model_kwargs["torch_dtype"] = compute_dtype
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()  # Set to evaluation mode
    
    print("Model loaded successfully!")
    print(f"Device Map: {model.hf_device_map}")
    print(f"Model Dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer


def find_level2_json(image_name, level2_dir):
    """Find the corresponding level-2 JSON file for an image, trying common patterns."""
    base = os.path.splitext(image_name)[0]
    
    # Try common suffixes first for speed
    for suffix in ["_level_2_processed.json", ".json"]:
        candidate_path = os.path.join(level2_dir, f"{base}{suffix}")
        if os.path.exists(candidate_path):
            return candidate_path
    
    # Fallback: search directory for any matching file prefix
    for fname in os.listdir(level2_dir):
        if fname.startswith(base) and fname.endswith(".json"):
            return os.path.join(level2_dir, fname)
    
    return None


def create_chat_prompt(scene_graph, tokenizer):
    """Create a properly formatted chat prompt for Llama 3.1 Instruct."""
    prompt_text = get_level_3_chat_prompt(scene_graph)
    
    # Wrap the prompt in a proper message structure
    messages = [
        {"role": "user", "content": prompt_text} 
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return formatted_prompt


def run_llama_inference(model, tokenizer, scene_graphs, max_new_tokens=150, 
                       temperature=0.7, top_p=0.9, max_length=3072):
    """
    Run batch inference with proper handling of variable-length inputs and outputs.
    
    --- REVISION: This function is completely overhauled for correct batch processing ---
    
    Key Design:
    1. VARIABLE INPUT LENGTHS: Solved by tokenizing with `padding=True` and `padding_side='left'`.
       The tokenizer pads all sequences to the length of the longest in the batch, and the
       attention mask tells the model which tokens are padding to be ignored.
    
    2. VARIABLE OUTPUT LENGTHS: Solved by slicing the generated output tensor.
       Since all inputs were padded to a uniform length `L`, the new tokens for *every*
       sequence start at index `L`. We can slice `output[i, L:]` to get just the new text.
    """
    
    if not scene_graphs:
        return [], []
    
    # Step 1: Create all prompts for the batch
    prompts = [create_chat_prompt(sg, tokenizer) for sg in scene_graphs]
    
    # Step 2: Tokenize the batch with LEFT padding
    # - padding=True: Pads all sequences to the length of the longest in the batch.
    # - padding_side="left": Essential for correct generation alignment.
    # - return_attention_mask=True: Tells the model which tokens to ignore.
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True,              
        truncation=True,           
        max_length=max_length,
        return_attention_mask=True 
    ).to(model.device)
    
    # The length of the padded input is the same for all items in the batch.
    batch_size = inputs.input_ids.shape[0]
    uniform_input_length = inputs.input_ids.shape[1]
    
    try:
        # Step 3: Generate text. The attention_mask ensures padding is ignored.
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0, # do_sample must be True for temperature/top_p to have an effect
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Step 4: Extract and decode only the newly generated text for each sequence
        responses = []
        reasons = []
        
        for i in range(batch_size):
            # The output tensor contains the input prompt + the new tokens.
            # We slice it to get ONLY the new tokens.
            generated_ids = outputs[i, uniform_input_length:]
            
            generated_text = tokenizer.decode(
                generated_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            ).strip()
            
            # Determine if generation stopped naturally or was cut off
            if tokenizer.eos_token_id in generated_ids:
                reason = "stop"  # Natural completion
            elif len(generated_ids) >= max_new_tokens:
                reason = "length"  # Hit token limit
            else:
                reason = "stop" # Default to stop if no other condition met
            
            responses.append(generated_text)
            reasons.append(reason)
        
    except RuntimeError as e:
        # Gracefully handle GPU Out-of-Memory errors
        if "out of memory" in str(e).lower():
            print(f"\n[ERROR] GPU OOM with batch_size={len(scene_graphs)}.")
            print(f"  -> Suggestion: Try running with a smaller batch size, e.g., --batch_size {max(1, len(scene_graphs)//2)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        else:
            print(f"\n[ERROR] Runtime error during inference: {str(e)}")
        # Return empty/error results for the entire batch on failure
        responses = [""] * len(scene_graphs)
        reasons = ["error"] * len(scene_graphs)
        
    except Exception as e:
        print(f"\n[ERROR] Unexpected error during batch inference: {str(e)}")
        responses = [""] * len(scene_graphs)
        reasons = ["error"] * len(scene_graphs)
    
    return responses, reasons


def validate_batch_outputs(captions, valid_names, reasons):
    """Validate and categorize batch outputs into successful and failed items."""
    successful = []
    failed_items = []
    
    for name, caption, reason in zip(valid_names, captions, reasons):
        if reason == "error":
            failed_items.append((name, "generation_error"))
        elif not caption or not caption.strip():
            failed_items.append((name, "empty_caption"))
        elif len(caption.strip()) < 10:
            failed_items.append((name, "caption_too_short"))
        else:
            successful.append((name, caption))
            if reason == "length":
                print(f"[WARNING] Caption for {name} may be truncated (hit max_new_tokens).")
    
    return successful, failed_items


def prepare_batch(image_batch, level2_dir, processed_files):
    """Load and validate scene graphs for a batch, skipping already processed files."""
    batch_data = []
    skipped_count = 0
    load_error_count = 0
    
    for img in image_batch:
        # Check if the output file already exists
        out_txt = f"{os.path.splitext(img)[0]}.txt"
        if out_txt in processed_files:
            skipped_count += 1
            continue
        
        json_path = find_level2_json(img, level2_dir)
        if not json_path:
            load_error_count += 1
            print(f"[WARNING] No Level-2 JSON found for: {img}")
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            scene_graph = get_simple_scene_graph(img, data)
            if not scene_graph or not isinstance(scene_graph, str) or not scene_graph.strip():
                load_error_count += 1
                print(f"[WARNING] Empty or invalid scene graph for: {img}")
                continue
            
            batch_data.append((img, scene_graph))
            
        except (json.JSONDecodeError, TypeError, Exception) as e:
            print(f"[ERROR] Failed to load or process JSON for {img}: {str(e)}")
            load_error_count += 1
    
    return batch_data, skipped_count, load_error_count


def process_batch_with_validation(model, tokenizer, batch_data, args, output_dir):
    """Process a batch, run inference, validate, and save results."""
    if not batch_data:
        return 0, 0
    
    valid_names = [name for name, _ in batch_data]
    scene_graphs = [sg for _, sg in batch_data]
    
    captions, reasons = run_llama_inference(
        model, tokenizer, scene_graphs,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_length=args.max_length
    )
    
    successful, failed_items = validate_batch_outputs(captions, valid_names, reasons)
    
    for name, caption in successful:
        out_path = os.path.join(output_dir, f"{os.path.splitext(name)[0]}.txt")
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(caption)
        except Exception as e:
            print(f"[ERROR] Failed to write file for {name}: {str(e)}")
            # Add to failed items if write fails
            failed_items.append((name, "write_error"))
    
    if failed_items:
        print(f"  -> Failures in batch: {len(failed_items)}")
        # Print a few examples of failures for diagnosis
        for name, reason in failed_items[:3]: 
            print(f"     - {name}: {reason}")
    
    # Return count of successful writes vs. total items attempted in the batch
    return len(successful), len(failed_items)


def main():
    args = parse_args()
    os.makedirs(args.output_directory_path, exist_ok=True)

    if torch.cuda.is_available():
        print(f"GPU detected: {torch.cuda.get_device_name()}")
        print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: No GPU available, using CPU. This will be extremely slow.")

    model, tokenizer = setup_model_and_tokenizer(args.model_path, args.quantization)

    with open(args.image_names_txt_path, 'r') as f:
        image_names = [ln.strip() for ln in f if ln.strip()]
    
    print(f"\n{'='*60}\nProcessing Configuration:\n{'='*60}")
    print(f"Total images to check: {len(image_names)}")
    print(f"Output directory: {args.output_directory_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Quantization: {args.quantization}\n")

    # Pre-load the set of already processed files for faster checking
    processed_files = set(os.listdir(args.output_directory_path))
    print(f"Found {len(processed_files)} already processed files. They will be skipped.\n")
    
    start_time = time.time()
    total_successful = 0
    total_failed = 0
    total_skipped = 0

    # Main processing loop
    for i in tqdm(range(0, len(image_names), args.batch_size), desc="Processing Batches"):
        batch_image_names = image_names[i:i + args.batch_size]
        
        # Prepare data: load JSONs, build scene graphs, skip processed files
        batch_data, skipped_in_batch, load_errors_in_batch = prepare_batch(
            batch_image_names, args.level_2_dir_path, processed_files
        )
        
        total_skipped += skipped_in_batch
        total_failed += load_errors_in_batch
        
        # Run inference only if there's valid data in the batch
        if batch_data:
            success_count, fail_count = process_batch_with_validation(
                model, tokenizer, batch_data, args, args.output_directory_path
            )
            total_successful += success_count
            total_failed += fail_count
        
        # Periodic memory cleanup can help prevent fragmentation on long runs
        if (i // args.batch_size) % 10 == 9: # Every 10 batches
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Final cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final statistics
    elapsed_time = time.time() - start_time
    print(f"\n{'='*60}\nPROCESSING COMPLETE\n{'='*60}")
    print(f"Total Time: {elapsed_time:.2f} seconds")
    print(f"  - Successfully processed: {total_successful}")
    print(f"  - Failed: {total_failed}")
    print(f"  - Skipped (already existed): {total_skipped}")
    
    if total_successful > 0:
        print(f"Average time per successful image: {elapsed_time / total_successful:.2f}s")
        print(f"Throughput: {total_successful / elapsed_time:.2f} images/sec")


if __name__ == "__main__":
    main()