#!/usr/bin/env python3
"""
SAM Mask Visualization Tool
Visualizes random images with their corresponding SAM mask annotations from JSON files
"""

# python visualize_random_sam.py --images_path "/mnt/e/Desktop/AgML/datasets_sorted/classification/arabica_coffee_leaf_disease_classification/Cerscospora" --json_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/classification/arabica_coffee_leaf_disease_classification/sam/Cerscospora" --n_samples 1
# python visualize_random_sam.py --images_path "/mnt/e/Desktop/AgML/datasets_sorted/detection/apple_detection_usa/images" --json_path "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection/apple_detection_usa/sam" --n_samples 1

import os
import json
import random
import argparse
from pathlib import Path
import glob
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns

try:
    from pycocotools import mask as mask_util
    HAS_PYCOCOTOOLS = True
    print("✅ Using pycocotools for RLE decoding")
except ImportError:
    HAS_PYCOCOTOOLS = False
    print("⚠️  pycocotools not found. Using numpy RLE decoding")


class SAMVisualizerError(Exception):
    """Custom exception for SAM visualizer errors"""
    pass


class SAMVisualizationTool:
    """Tool for visualizing SAM mask annotations on images"""
    
    def __init__(self, images_path: str, json_path: str):
        """
        Initialize the visualization tool
        
        Args:
            images_path: Path to folder containing images
            json_path: Path to folder containing JSON annotation files
        """
        self.images_path = Path(images_path)
        self.json_path = Path(json_path)
        
        if not self.images_path.exists():
            raise SAMVisualizerError(f"Images path does not exist: {images_path}")
        if not self.json_path.exists():
            raise SAMVisualizerError(f"JSON path does not exist: {json_path}")
        
        # Supported image extensions
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
        
        # Find matching image-json pairs
        self.image_json_pairs = self._find_matching_pairs()
        
        if not self.image_json_pairs:
            raise SAMVisualizerError("No matching image-JSON pairs found!")
        
        print(f"Found {len(self.image_json_pairs)} matching image-JSON pairs")
    
    def _find_matching_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image and JSON file pairs"""
        pairs = []
        
        # Check if we have a classification structure (class folders)
        image_files = []
        json_files = []
        
        # Collect all image files (handling both flat and classification structure)
        if any(p.is_dir() for p in self.images_path.iterdir()):
            # Classification structure: images_path/class_name/image_files
            for class_dir in self.images_path.iterdir():
                if class_dir.is_dir():
                    for img_file in class_dir.iterdir():
                        if img_file.suffix.lower() in self.image_extensions:
                            image_files.append(img_file)
        else:
            # Flat structure: all images in images_path
            for img_file in self.images_path.iterdir():
                if img_file.suffix.lower() in self.image_extensions:
                    image_files.append(img_file)
        
        # Collect all JSON files (handling both flat and classification structure)
        if any(p.is_dir() for p in self.json_path.iterdir()):
            # Classification structure: json_path/class_name/json_files
            for class_dir in self.json_path.iterdir():
                if class_dir.is_dir():
                    for json_file in class_dir.glob("*.json"):
                        json_files.append(json_file)
        else:
            # Flat structure: all JSONs in json_path
            for json_file in self.json_path.glob("*.json"):
                json_files.append(json_file)
        
        # Create mappings for faster lookup
        image_map = {}
        for img_file in image_files:
            # Key: (class_name, stem) or just stem for flat structure
            if img_file.parent.name != self.images_path.name:
                key = (img_file.parent.name, img_file.stem)
            else:
                key = img_file.stem
            image_map[key] = img_file
        
        json_map = {}
        for json_file in json_files:
            # Key: (class_name, stem) or just stem for flat structure
            if json_file.parent.name != self.json_path.name:
                key = (json_file.parent.name, json_file.stem)
            else:
                key = json_file.stem
            json_map[key] = json_file
        
        # Find matching pairs
        for key in image_map:
            if key in json_map:
                pairs.append((image_map[key], json_map[key]))
        
        return pairs
    
    def _decode_rle_mask(self, rle_data: Dict, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Decode RLE mask data to binary mask
        
        Args:
            rle_data: RLE data from JSON (with 'size' and 'counts')
            image_shape: Target image shape (height, width)
            
        Returns:
            Binary mask as numpy array
        """
        try:
            if HAS_PYCOCOTOOLS:
                # Use pycocotools for decoding
                mask = mask_util.decode(rle_data)
            else:
                # Use custom numpy RLE decoder
                mask = self._numpy_rle_decode(rle_data['counts'], rle_data['size'])
            
            # Resize mask if necessary
            if mask.shape != image_shape:
                mask = cv2.resize(mask.astype(np.uint8), 
                                (image_shape[1], image_shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            return mask.astype(bool)
            
        except Exception as e:
            print(f"Warning: Failed to decode RLE mask: {e}")
            return np.zeros(image_shape, dtype=bool)
    
    def _numpy_rle_decode(self, counts: List[int], size: List[int]) -> np.ndarray:
        """
        Decode RLE using numpy (fallback when pycocotools not available)
        
        Args:
            counts: RLE counts list
            size: [height, width] of the mask
            
        Returns:
            Binary mask as numpy array
        """
        h, w = size
        mask = np.zeros(h * w, dtype=np.uint8)
        
        if not counts:
            return mask.reshape(h, w)
        
        # RLE decoding
        pos = 0
        for i, count in enumerate(counts):
            if i % 2 == 1:  # Odd indices represent mask pixels
                mask[pos:pos + count] = 1
            pos += count
            if pos >= len(mask):
                break
        
        return mask.reshape(h, w)
    
    def load_image_and_annotations(self, image_path: Path, json_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load image and its corresponding annotations
        
        Args:
            image_path: Path to image file
            json_path: Path to JSON annotation file
            
        Returns:
            Tuple of (image_array, annotation_data)
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise SAMVisualizerError(f"Failed to load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load annotations
        try:
            with open(json_path, 'r') as f:
                annotation_data = json.load(f)
        except Exception as e:
            raise SAMVisualizerError(f"Failed to load JSON: {json_path}, Error: {e}")
        
        return image, annotation_data
    
    def create_mask_overlay(self, image: np.ndarray, annotations: List[Dict], 
                           show_boxes: bool = True, show_masks: bool = True,
                           alpha: float = 0.4) -> np.ndarray:
        """
        Create image with mask overlays
        
        Args:
            image: Original image array
            annotations: List of annotation dictionaries
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show mask overlays
            alpha: Transparency for mask overlay
            
        Returns:
            Image with overlays
        """
        overlay_image = image.copy()
        h, w = image.shape[:2]
        
        # Generate colors for masks
        colors = plt.cm.Set3(np.linspace(0, 1, len(annotations)))
        
        # Create combined mask overlay
        if show_masks and annotations:
            combined_mask = np.zeros((h, w, 3), dtype=np.float32)
            
            for i, ann in enumerate(annotations):
                try:
                    # Decode mask
                    segmentation = ann.get('segmentation', {})
                    if not segmentation:
                        continue
                        
                    mask = self._decode_rle_mask(segmentation, (h, w))
                    
                    # Apply color to mask
                    color = colors[i][:3]  # RGB only
                    for c in range(3):
                        combined_mask[:, :, c] += mask * color[c]
                
                except Exception as e:
                    print(f"Warning: Failed to process mask {i}: {e}")
                    continue
            
            # Blend with original image
            mask_valid = np.any(combined_mask > 0, axis=2)
            overlay_image = overlay_image.astype(np.float32)
            
            for c in range(3):
                overlay_image[:, :, c] = np.where(
                    mask_valid,
                    (1 - alpha) * overlay_image[:, :, c] + alpha * combined_mask[:, :, c] * 255,
                    overlay_image[:, :, c]
                )
            
            overlay_image = np.clip(overlay_image, 0, 255).astype(np.uint8)
        
        return overlay_image
    
    def visualize_sample(self, n_samples: int = 5, figsize: Tuple[int, int] = (20, 4),
                        show_boxes: bool = True, show_masks: bool = True,
                        alpha: float = 0.4, save_path: Optional[str] = None):
        """
        Visualize N random samples with their masks
        
        Args:
            n_samples: Number of random samples to visualize
            figsize: Figure size for the plot
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show mask overlays
            alpha: Transparency for mask overlays
            save_path: Optional path to save the visualization
        """
        if n_samples > len(self.image_json_pairs):
            print(f"Warning: Requested {n_samples} samples but only {len(self.image_json_pairs)} available")
            n_samples = len(self.image_json_pairs)
        
        # Randomly sample pairs
        sampled_pairs = random.sample(self.image_json_pairs, n_samples)
        
        # Create subplot
        fig, axes = plt.subplots(2, n_samples, figsize=figsize)
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i, (image_path, json_path) in enumerate(sampled_pairs):
            try:
                # Load image and annotations
                image, annotation_data = self.load_image_and_annotations(image_path, json_path)
                annotations = annotation_data.get('annotations', [])
                
                # Get image info
                image_info = annotation_data.get('image_info', {})
                class_name = annotation_data.get('class_name', 'Unknown')
                num_masks = len(annotations)
                
                # Original image
                axes[0, i].imshow(image)
                axes[0, i].set_title(f'Original\n{class_name}\n{image_path.name}', fontsize=10)
                axes[0, i].axis('off')
                
                # Add bounding boxes to original if requested
                if show_boxes and annotations:
                    for ann in annotations:
                        bbox = ann.get('bbox', [])
                        if len(bbox) >= 4:
                            x, y, w, h = bbox
                            rect = patches.Rectangle((x, y), w, h, linewidth=1, 
                                                   edgecolor='red', facecolor='none')
                            axes[0, i].add_patch(rect)
                
                # Image with mask overlay
                overlay_image = self.create_mask_overlay(image, annotations, 
                                                       show_boxes, show_masks, alpha)
                axes[1, i].imshow(overlay_image)
                axes[1, i].set_title(f'With Masks\n{num_masks} masks', fontsize=10)
                axes[1, i].axis('off')
                
                # Print info
                print(f"Sample {i+1}: {image_path.name}")
                print(f"  Class: {class_name}")
                print(f"  Masks: {num_masks}")
                if image_info:
                    orig_size = image_info.get('original_size', {})
                    if orig_size:
                        print(f"  Size: {orig_size.get('width', '?')}x{orig_size.get('height', '?')}")
                print()
                
            except Exception as e:
                print(f"Error processing sample {i+1}: {e}")
                # Show error placeholder
                for row in range(2):
                    axes[row, i].text(0.5, 0.5, f'Error\n{e}', ha='center', va='center',
                                    transform=axes[row, i].transAxes, fontsize=8)
                    axes[row, i].set_title(f'Error: {image_path.name}', fontsize=10)
                    axes[row, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about the dataset"""
        stats = {
            'total_pairs': len(self.image_json_pairs),
            'total_masks': 0,
            'classes': set(),
            'avg_masks_per_image': 0,
            'mask_areas': [],
            'image_sizes': []
        }
        
        valid_pairs = 0
        
        for image_path, json_path in self.image_json_pairs:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                annotations = data.get('annotations', [])
                class_name = data.get('class_name', 'Unknown')
                image_info = data.get('image_info', {})
                
                stats['classes'].add(class_name)
                stats['total_masks'] += len(annotations)
                
                # Collect mask areas
                for ann in annotations:
                    area = ann.get('area', 0)
                    if area > 0:
                        stats['mask_areas'].append(area)
                
                # Collect image size
                orig_size = image_info.get('original_size', {})
                if orig_size and 'width' in orig_size and 'height' in orig_size:
                    w, h = orig_size['width'], orig_size['height']
                    stats['image_sizes'].append((w, h))
                
                valid_pairs += 1
                
            except Exception as e:
                print(f"Warning: Error reading {json_path}: {e}")
        
        if valid_pairs > 0:
            stats['avg_masks_per_image'] = stats['total_masks'] / valid_pairs
        
        stats['classes'] = sorted(list(stats['classes']))
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_dataset_statistics()
        
        print("="*60)
        print("DATASET STATISTICS")
        print("="*60)
        print(f"Total image-JSON pairs: {stats['total_pairs']}")
        print(f"Total masks: {stats['total_masks']:,}")
        print(f"Average masks per image: {stats['avg_masks_per_image']:.2f}")
        print(f"Number of classes: {len(stats['classes'])}")
        print(f"Classes: {', '.join(stats['classes'])}")
        
        if stats['mask_areas']:
            areas = np.array(stats['mask_areas'])
            print(f"\nMask area statistics:")
            print(f"  Min area: {areas.min():,} pixels")
            print(f"  Max area: {areas.max():,} pixels")
            print(f"  Mean area: {areas.mean():.0f} pixels")
            print(f"  Median area: {np.median(areas):.0f} pixels")
        
        if stats['image_sizes']:
            sizes = np.array(stats['image_sizes'])
            print(f"\nImage size statistics:")
            print(f"  Min dimensions: {sizes.min(axis=0)}")
            print(f"  Max dimensions: {sizes.max(axis=0)}")
            print(f"  Mean dimensions: {sizes.mean(axis=0).astype(int)}")


def main():
    parser = argparse.ArgumentParser(description="Visualize SAM mask annotations on images")
    parser.add_argument("--images_path", required=True, help="Path to images folder")
    parser.add_argument("--json_path", required=True, help="Path to JSON annotations folder")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of random samples to visualize")
    parser.add_argument("--figsize", nargs=2, type=int, default=[20, 4], help="Figure size (width, height)")
    parser.add_argument("--no_boxes", action='store_true', help="Don't show bounding boxes")
    parser.add_argument("--no_masks", action='store_true', help="Don't show mask overlays")
    parser.add_argument("--alpha", type=float, default=0.4, help="Mask overlay transparency")
    parser.add_argument("--save_path", help="Path to save the visualization image")
    parser.add_argument("--stats_only", action='store_true', help="Only show statistics, don't visualize")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible sampling")
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    try:
        # Initialize visualizer
        print("🚀 Initializing SAM Visualization Tool...")
        visualizer = SAMVisualizationTool(args.images_path, args.json_path)
        
        # Show statistics
        visualizer.print_statistics()
        
        if not args.stats_only:
            print(f"\n📊 Visualizing {args.n_samples} random samples...")
            
            # Visualize samples
            visualizer.visualize_sample(
                n_samples=args.n_samples,
                figsize=tuple(args.figsize),
                show_boxes=not args.no_boxes,
                show_masks=not args.no_masks,
                alpha=args.alpha,
                save_path=args.save_path
            )
        
        print("✅ Visualization complete!")
        
    except SAMVisualizerError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()