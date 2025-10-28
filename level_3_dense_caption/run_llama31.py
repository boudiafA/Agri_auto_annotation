# python run_llama3_multi_gpu.py --image_dir_path "/mnt/e/Desktop/AgML/datasets_sorted/iNatAg_subset/abelmoschus_esculentus" --level_2_dir_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/iNatAg_subset/abelmoschus_esculentus/level-2-processed_labelled" --output_dir_path "/mnt/e/Desktop/AgML//AgriDataset_GranD_annotation/iNatAg_subset/abelmoschus_esculentus/level-3-llama3-8B" --gpu_ids "0,1,2,3" --model_path "meta-llama/Meta-Llama-3.1-8B-Instruct" --quantization "8bit" --batch_size 4

import os
import argparse
import subprocess
import time
import multiprocessing as mp
from multiprocessing import Process
from tqdm import tqdm


def launch_llama3(image_names_txt_path, level_2_dir_path, gpu_id, output_dir_path, 
                  model_path, quantization, batch_size, max_new_tokens, temperature, top_p
):
    """Launch Llama3 processing on a specific GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Verify paths exist
    if not os.path.exists(level_2_dir_path):
        print(f"[GPU {gpu_id}] ERROR: Level-2 directory does not exist: {level_2_dir_path}")
        return
    
    if not os.path.exists(image_names_txt_path):
        print(f"[GPU {gpu_id}] ERROR: Image names file does not exist: {image_names_txt_path}")
        return

    # Build command list
    cmd = [
        "python", "query_llama3_level_3.py",
        "--image_names_txt_path", image_names_txt_path,
        "--level_2_dir_path", level_2_dir_path,
        "--output_directory_path", output_dir_path,
        "--model_path", model_path,
        "--quantization", quantization,
        "--batch_size", str(batch_size),
        "--max_new_tokens", str(max_new_tokens),
        "--temperature", str(temperature),
        "--top_p", str(top_p)
    ]
    
    # Launch the external script
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[GPU {gpu_id}] ERROR: Processing failed with return code {e.returncode}")
    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {str(e)}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images with Llama 3.1 across multiple GPUs.")

    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images")
    parser.add_argument("--level_2_dir_path", type=str, required=True,
                        help="Path to the processed level-2 directory.")
    parser.add_argument("--output_dir_path", required=True, 
                        help="Path to the output directory")
    parser.add_argument("--gpu_ids", type=str, required=True, 
                        help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", 
                        help="Path to Llama model or HuggingFace model ID")
    parser.add_argument("--quantization", type=str, default="8bit", 
                        choices=["none", "8bit", "4bit"],
                        help="Quantization method for memory efficiency")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for processing")
    parser.add_argument("--max_new_tokens", type=int, default=150,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p for nucleus sampling")
    parser.add_argument("--job_id", type=str, default="llama3_job",
                        help="Job identifier for tracking")

    return parser.parse_args()


def main():
    args = parse_arguments()
    
    print("="*60)
    print("LLAMA 3.1 MULTI-GPU PROCESSING")
    print("="*60)
    print(f"Model: {args.model_path} | Quantization: {args.quantization} | Batch size: {args.batch_size}")
    print(f"GPUs: {args.gpu_ids}")
    
    # Check if directories exist
    if not os.path.exists(args.image_dir_path):
        print(f"ERROR: Image directory does not exist: {args.image_dir_path}")
        return
    
    if not os.path.exists(args.level_2_dir_path):
        print(f"ERROR: Level-2 directory does not exist: {args.level_2_dir_path}")
        return
        
    if not os.path.exists(args.output_dir_path):
        os.makedirs(args.output_dir_path, exist_ok=True)

    # Convert GPU IDs to list
    gpu_ids = list(map(int, args.gpu_ids.split(',')))

    # Get image files
    try:
        image_names = [f for f in os.listdir(args.image_dir_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_names)} images")
    except Exception as e:
        print(f"ERROR: Failed to list images: {str(e)}")
        return
        
    if not image_names:
        print(f"ERROR: No images found in {args.image_dir_path}")
        return

    # Split images across GPUs
    num_images_per_gpu = len(image_names) // len(gpu_ids)
    print(f"Distributing ~{num_images_per_gpu} images per GPU")

    processes = []
    temp_files = []
    start_time = time.time()

    # Create image list files and launch processes
    for i, gpu_id in enumerate(tqdm(gpu_ids, desc="Launching GPUs", unit="GPU")):
        start_idx = i * num_images_per_gpu
        end_idx = start_idx + num_images_per_gpu if i != len(gpu_ids) - 1 else None
        batch_images = image_names[start_idx:end_idx]

        # Write temporary file
        timestamp = time.strftime("%Y%m%d%H%M%S")
        txt_path = f"{args.job_id}_{timestamp}_gpu_{gpu_id}.txt"
        temp_files.append(txt_path)
        
        try:
            with open(txt_path, "w") as f:
                for name in batch_images:
                    f.write(name + "\n")
        except Exception as e:
            print(f"ERROR: Failed to create file for GPU {gpu_id}: {str(e)}")
            continue

        # Launch process
        p = Process(target=launch_llama3, args=(
            txt_path, 
            args.level_2_dir_path, 
            gpu_id, 
            args.output_dir_path, 
            args.model_path,
            args.quantization,
            args.batch_size,
            args.max_new_tokens,
            args.temperature,
            args.top_p
        ))
        processes.append(p)
        p.start()

    # Wait for completion with progress bar
    print("\nProcessing images...")
    for i, p in enumerate(tqdm(processes, desc="GPU Progress", unit="GPU")):
        p.join()

    elapsed_time = time.time() - start_time

    # Cleanup temporary files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            print(f"WARNING: Failed to remove {temp_file}: {str(e)}")

    # Final statistics
    print("\n" + "="*60)
    print("PROCESSING COMPLETED")
    print("="*60)
    print(f"Total time: {elapsed_time:.2f}s | Avg per image: {elapsed_time/len(image_names):.2f}s")
    
    try:
        output_files = [f for f in os.listdir(args.output_dir_path) 
                       if f.endswith(('.txt', '.json'))]
        print(f"Output files: {len(output_files)}/{len(image_names)}")
        
        if len(output_files) < len(image_names):
            print(f"WARNING: {len(image_names) - len(output_files)} images not processed")
            
    except Exception as e:
        print(f"Could not count output files: {e}")

    print(f"Job ID: {args.job_id} - Complete")


if __name__ == "__main__":
    main()