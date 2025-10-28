# python launch_ov_sam_multi_gpu_inference.py --image_dir_path "/home/abood/groundingLMM/GranD/images" --output_dir_path "/home/abood/groundingLMM/GranD/output" --sam_annotations_dir "/home/abood/groundingLMM/GranD/checkpoints" --gpu_ids "0" --batch_size 4 --bbox_batch_size 32
import os
import argparse
import subprocess
import time
from multiprocessing import Process


def launch_ovsam(image_names_txt_path, gpu_id, image_dir_path, sam_annotations_dir, output_dir_path, batch_size, bbox_batch_size):
    """Launch the batch inference script."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    subprocess.run([
        "python",
        "infer.py",
        "--image_names_txt_path", image_names_txt_path,
        "--image_dir_path", image_dir_path,
        "--sam_annotations_dir", sam_annotations_dir,
        "--output_dir_path", output_dir_path,
        "--batch_size", str(batch_size),
        "--bbox_batch_size", str(bbox_batch_size),
    ])


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run batch OV-SAM prediction across multiple GPUs.")

    parser.add_argument("--image_dir_path", type=str, required=True, 
                        help="Path to the image directory.")
    parser.add_argument("--output_dir_path", type=str, required=True, 
                        help="Path for saving the outputs.")
    parser.add_argument("--sam_annotations_dir", required=True,
                        help="Path to the directory containing all sam annotations.")
    parser.add_argument("--gpu_ids", type=str, required=True, 
                        help="Comma-separated GPU IDs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Number of images to process in each batch per GPU.")
    parser.add_argument("--bbox_batch_size", type=int, default=8,
                        help="Number of bounding boxes to process in each inference batch.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    # Convert the comma-separated GPU IDs to a list of integers
    gpu_ids = list(map(int, args.gpu_ids.split(',')))

    # Get the image names from the directory
    image_names = os.listdir(args.image_dir_path)
    
    # Filter for image files only
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_names = [name for name in image_names 
                   if any(name.lower().endswith(ext) for ext in image_extensions)]

    print(f"Found {len(image_names)} images to process")
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Bbox batch size: {args.bbox_batch_size}")

    # Split the image names across the GPUs
    num_images_per_gpu = len(image_names) // len(gpu_ids)

    processes = []
    temp_files = []  # List to track temporary files created

    for i, gpu_id in enumerate(gpu_ids):
        time.sleep(1)
        start_idx = i * num_images_per_gpu
        end_idx = start_idx + num_images_per_gpu if i != len(gpu_ids) - 1 else None  # Take the rest for the last GPU

        # Write to a temporary file
        timestamp = time.strftime("%Y%m%d%H%M%S")
        txt_path = f"temp_gpu_{gpu_id}_{timestamp}.txt"
        temp_files.append(txt_path)
        
        gpu_images = image_names[start_idx:end_idx]
        print(f"GPU {gpu_id} will process {len(gpu_images)} images")
        
        with open(txt_path, "w") as f:
            for name in gpu_images:
                f.write(name + "\n")

        # Launch a new process for this GPU
        p = Process(target=launch_ovsam,
                    args=(txt_path, gpu_id, args.image_dir_path, args.sam_annotations_dir, 
                          args.output_dir_path, args.batch_size, args.bbox_batch_size))
        processes.append(p)
        p.start()

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Remove all temporary files created
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print("All processes completed!")


if __name__ == "__main__":
    main()