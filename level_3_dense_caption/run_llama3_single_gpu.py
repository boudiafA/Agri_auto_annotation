# python run_llama3_single_gpu.py --image_dir_path "/mnt/e/Desktop/AgML/datasets_sorted/iNatAg_subset/abelmoschus_esculentus" --level_2_dir_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/iNatAg_subset/abelmoschus_esculentus/level-2-processed_labelled" --output_dir_path "/mnt/e/Desktop/AgML//AgriDataset_GranD_annotation/iNatAg_subset/abelmoschus_esculentus/level-3-llama3-8B" --model_path "meta-llama/Meta-Llama-3.1-8B-Instruct" --quantization "8bit" --batch_size 4

import os
import argparse
import subprocess
import time
import multiprocessing as mp

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images with Llama 3.1 on single GPU.")

    parser.add_argument("--image_dir_path", required=True,
                        help="Path to the directory containing images")
    parser.add_argument("--level_2_dir_path", type=str, required=True,
                        help="Path to the processed level-2 directory.")
    parser.add_argument("--output_dir_path", required=True, 
                        help="Path to the output directory")
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
    parser.add_argument("--num_cpu_workers", type=int, 
                        default=min(8, mp.cpu_count()),
                        help="Number of CPU workers for data processing")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID to use (default: 0)")

    return parser.parse_args()


def create_image_list_file(image_dir_path, output_path="image_list.txt"):
    """Create a text file with all image names"""
    try:
        image_names = [f for f in os.listdir(image_dir_path) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        with open(output_path, "w") as f:
            for name in image_names:
                f.write(name + "\n")
        
        print(f"Created image list file: {output_path}")
        print(f"Total images: {len(image_names)}")
        return output_path, len(image_names)
        
    except Exception as e:
        print(f"ERROR: Failed to create image list: {str(e)}")
        return None, 0


def main():
    args = parse_arguments()
    
    print("="*60)
    print("LLAMA 3.1 SINGLE GPU PROCESSING")
    print("="*60)
    print(f"Image directory: {args.image_dir_path}")
    print(f"Level-2 directory: {args.level_2_dir_path}")
    print(f"Output directory: {args.output_dir_path}")
    print(f"Model: {args.model_path}")
    print(f"Quantization: {args.quantization}")
    print(f"Batch size: {args.batch_size}")
    print(f"CPU workers: {args.num_cpu_workers}")
    print(f"GPU ID: {args.gpu_id}")
    
    # Check if directories exist
    if not os.path.exists(args.image_dir_path):
        print(f"ERROR: Image directory does not exist: {args.image_dir_path}")
        return
    
    if not os.path.exists(args.level_2_dir_path):
        print(f"ERROR: Level-2 directory does not exist: {args.level_2_dir_path}")
        return
        
    if not os.path.exists(args.output_dir_path):
        print(f"Creating output directory: {args.output_dir_path}")
        os.makedirs(args.output_dir_path, exist_ok=True)

    # Create image list file
    image_list_path, total_images = create_image_list_file(
        args.image_dir_path, "single_gpu_image_list.txt"
    )
    
    if not image_list_path or total_images == 0:
        print("No images found to process")
        return

    # Set GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    
    # Build command for the main processing script
    cmd = [
        "python", "query_llama3_level_3.py",
        "--image_names_txt_path", image_list_path,
        "--level_2_dir_path", args.level_2_dir_path,
        "--output_directory_path", args.output_dir_path,
        "--model_path", args.model_path,
        "--quantization", args.quantization,
        "--batch_size", str(args.batch_size),
        "--max_new_tokens", str(args.max_new_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p)
    ]
    
    print("\n" + "="*60)
    print("STARTING PROCESSING")
    print("="*60)
    print(f"Command: {' '.join(cmd)}")
    print(f"Processing {total_images} images...")
    
    start_time = time.time()
    
    try:
        # Run the processing script
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Average time per image: {elapsed_time/total_images:.2f} seconds")
        
        # Show some output if available
        if result.stdout:
            print("\nLast few lines of output:")
            print(result.stdout.split('\n')[-10:])
            
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\nERROR: Processing failed after {elapsed_time:.2f} seconds")
        print(f"Return code: {e.returncode}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        if e.stdout:
            print("Standard output:")
            print(e.stdout)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")
    
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(image_list_path):
                os.remove(image_list_path)
                print(f"Cleaned up temporary file: {image_list_path}")
        except:
            pass

    # Show final statistics
    try:
        output_files = [f for f in os.listdir(args.output_dir_path) if f.endswith('.txt')]
        print(f"\nFinal result: {len(output_files)} caption files generated")
        
        if len(output_files) < total_images:
            print(f"Warning: {total_images - len(output_files)} images were not processed")
            
    except Exception as e:
        print(f"Could not count output files: {e}")

if __name__ == "__main__":
    main()