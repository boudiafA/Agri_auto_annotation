#!/usr/bin/env python3
"""
Cleanup script to remove duplicate depth map outputs.
Keeps only the most recent file for each original image.

Usage:
python cleanup_duplicates.py --output_dir "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/detection" --dry_run
python cleanup_duplicates.py --output_dir "/mnt/e/Desktop/AgML/AgriDataset_GranD_annotation/segm --execute --keep largest
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict
import re

def extract_original_stem(filename):
    """
    Extract the original image stem from depth map filenames.
    
    Examples:
    - "0001.jpg_min_-156.1252_max_6535.0688.jpg" -> "0001"
    - "photo.png_min_100.5_max_500.2.jpg" -> "photo"
    - "image.jpg" -> "image" (fallback for normal files)
    """
    if '_min_' in filename and '_max_' in filename:
        # Split at first occurrence of '_min_' to get the original name
        original_name = filename.split('_min_')[0]
        # Remove the extension if present to get just the stem
        if '.' in original_name:
            stem = original_name.rsplit('.', 1)[0]  # Remove last extension
        else:
            stem = original_name
        return stem
    else:
        # Fallback for normal naming
        return Path(filename).stem

def get_file_info(filepath):
    """Get file modification time and size for comparison."""
    stat = os.stat(filepath)
    return {
        'mtime': stat.st_mtime,
        'size': stat.st_size,
        'path': filepath
    }

def find_duplicates_in_directory(directory):
    """
    Find duplicate depth map files in a directory.
    
    Returns:
        dict: {original_stem: [list of file info dicts]}
    """
    if not os.path.exists(directory):
        return {}
    
    # Group files by their original stem
    stem_to_files = defaultdict(list)
    
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        
        # Only process image files
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        if os.path.isfile(filepath):
            original_stem = extract_original_stem(filename)
            file_info = get_file_info(filepath)
            file_info['filename'] = filename
            stem_to_files[original_stem].append(file_info)
    
    # Only return stems that have duplicates
    duplicates = {stem: files for stem, files in stem_to_files.items() if len(files) > 1}
    return duplicates

def cleanup_directory(directory, dry_run=True, keep_strategy='newest'):
    """
    Clean up duplicate files in a directory.
    
    Args:
        directory: Path to directory to clean
        dry_run: If True, only show what would be deleted
        keep_strategy: 'newest', 'oldest', or 'largest'
    
    Returns:
        dict: Statistics about the cleanup
    """
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Cleaning directory: {directory}")
    
    duplicates = find_duplicates_in_directory(directory)
    
    if not duplicates:
        print("  No duplicates found.")
        return {'duplicates_found': 0, 'files_removed': 0, 'space_freed': 0}
    
    total_files_removed = 0
    total_space_freed = 0
    
    print(f"  Found {len(duplicates)} images with duplicate outputs:")
    
    for original_stem, files in duplicates.items():
        print(f"\n    Image '{original_stem}' has {len(files)} outputs:")
        
        # Sort files based on keep strategy
        if keep_strategy == 'newest':
            files.sort(key=lambda x: x['mtime'], reverse=True)
        elif keep_strategy == 'oldest':
            files.sort(key=lambda x: x['mtime'])
        elif keep_strategy == 'largest':
            files.sort(key=lambda x: x['size'], reverse=True)
        
        # Keep the first file, remove the rest
        file_to_keep = files[0]
        files_to_remove = files[1:]
        
        print(f"      KEEPING: {file_to_keep['filename']} (size: {file_to_keep['size']} bytes)")
        
        for file_info in files_to_remove:
            print(f"      {'WOULD REMOVE' if dry_run else 'REMOVING'}: {file_info['filename']} (size: {file_info['size']} bytes)")
            
            if not dry_run:
                try:
                    os.remove(file_info['path'])
                    total_files_removed += 1
                    total_space_freed += file_info['size']
                except Exception as e:
                    print(f"        ERROR: Could not remove {file_info['filename']}: {e}")
            else:
                total_files_removed += 1
                total_space_freed += file_info['size']
    
    return {
        'duplicates_found': len(duplicates),
        'files_removed': total_files_removed,
        'space_freed': total_space_freed
    }

def cleanup_recursive(root_dir, dry_run=True, keep_strategy='newest'):
    """
    Recursively clean up duplicate files in all subdirectories.
    """
    print(f"\n{'='*60}")
    print(f"{'DRY RUN - ' if dry_run else ''}DUPLICATE DEPTH MAP CLEANUP")
    print(f"Root directory: {root_dir}")
    print(f"Keep strategy: {keep_strategy}")
    print(f"{'='*60}")
    
    total_stats = {'duplicates_found': 0, 'files_removed': 0, 'space_freed': 0}
    
    # Find all directories that might contain depth maps
    depth_dirs = []
    
    for root, dirs, files in os.walk(root_dir):
        # Look for 'midas' directories or directories with depth map files
        if os.path.basename(root) == 'midas' or any(f.endswith('.jpg') and '_min_' in f for f in files):
            depth_dirs.append(root)
    
    if not depth_dirs:
        print("No depth map directories found!")
        return total_stats
    
    print(f"\nFound {len(depth_dirs)} directories to check:")
    for d in depth_dirs:
        print(f"  {d}")
    
    # Process each directory
    for directory in depth_dirs:
        stats = cleanup_directory(directory, dry_run, keep_strategy)
        total_stats['duplicates_found'] += stats['duplicates_found']
        total_stats['files_removed'] += stats['files_removed']
        total_stats['space_freed'] += stats['space_freed']
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  Images with duplicates: {total_stats['duplicates_found']}")
    print(f"  Files {'that would be' if dry_run else ''} removed: {total_stats['files_removed']}")
    print(f"  Space {'that would be' if dry_run else ''} freed: {total_stats['space_freed'] / (1024*1024):.1f} MB")
    
    if dry_run:
        print(f"\nThis was a DRY RUN. Use --execute to actually remove files.")
    else:
        print(f"\nCleanup completed!")
    
    return total_stats

def parse_args():
    parser = argparse.ArgumentParser(description="Clean up duplicate depth map outputs")
    
    parser.add_argument("--output_dir", required=True,
                        help="Path to the root output directory")
    
    parser.add_argument("--dry_run", action="store_true", default=False,
                        help="Show what would be deleted without actually deleting")
    
    parser.add_argument("--execute", action="store_true", default=False,
                        help="Actually delete duplicate files")
    
    parser.add_argument("--keep", choices=['newest', 'oldest', 'largest'], default='newest',
                        help="Which file to keep when duplicates are found (default: newest)")
    
    parser.add_argument("--specific_dir", 
                        help="Clean only a specific subdirectory instead of recursive search")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.dry_run and not args.execute:
        parser.error("Must specify either --dry_run or --execute")
    
    if args.dry_run and args.execute:
        parser.error("Cannot specify both --dry_run and --execute")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist!")
        exit(1)
    
    # Run cleanup
    if args.specific_dir:
        target_dir = os.path.join(args.output_dir, args.specific_dir)
        if not os.path.exists(target_dir):
            print(f"Error: Specific directory '{target_dir}' does not exist!")
            exit(1)
        cleanup_directory(target_dir, dry_run=args.dry_run, keep_strategy=args.keep)
    else:
        cleanup_recursive(args.output_dir, dry_run=args.dry_run, keep_strategy=args.keep)
