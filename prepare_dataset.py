"""
Dataset Preparation and Alignment Script

This script identifies and fixes alignment issues in the HDR dataset:
1. Detects rotation/alignment differences between raw and edited images
2. Creates an aligned dataset where raw images match edited orientation
3. Preserves original dataset - creates a new aligned copy

Usage:
    # Check dataset for alignment issues
    python prepare_dataset.py --check --data "C:/Users/Administrator/Documents/HDR dataset"
    
    # Create aligned dataset
    python prepare_dataset.py --align --data "C:/Users/Administrator/Documents/HDR dataset" --output "C:/Users/Administrator/Documents/HDR dataset aligned"
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not installed. Install with: pip install opencv-python")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare and align HDR dataset")
    
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Check dataset for alignment issues"
    )
    parser.add_argument(
        "--align", action="store_true",
        help="Create aligned dataset"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for aligned dataset"
    )
    parser.add_argument(
        "--report", type=str, default="dataset_report.json",
        help="Output file for dataset report"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality for saved images (default: 95)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    
    return parser.parse_args()


def find_edited_folder(folder_path: Path) -> Optional[Path]:
    """Find the edited folder (handles case variations)"""
    for name in ['edited', 'Edited', 'EDITED']:
        edited_path = folder_path / name
        if edited_path.exists():
            return edited_path
    return None


def get_image_sets(dataset_path: Path, num_brackets: int = 5) -> List[Dict]:
    """Get all image sets from the dataset"""
    sets = []
    
    for folder in sorted(dataset_path.iterdir()):
        if not folder.is_dir():
            continue
        
        raw_folder = folder / 'raw'
        edited_folder = find_edited_folder(folder)
        
        if not raw_folder.exists() or edited_folder is None:
            continue
        
        # Get edited images
        edited_images = {}
        for img_path in edited_folder.iterdir():
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                edited_images[img_path.stem.upper()] = img_path
        
        # Get raw images (use set to avoid duplicates on case-insensitive Windows)
        raw_images_set = set()
        for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
            for img in raw_folder.glob(ext):
                raw_images_set.add(img)
        raw_images = list(raw_images_set)
        
        # Sort by numeric part of filename
        def extract_number(filename: str) -> int:
            match = re.search(r'(\d+)', filename)
            return int(match.group(1)) if match else 0
        
        raw_images.sort(key=lambda x: extract_number(x.stem))
        
        # Match sets
        i = 0
        while i < len(raw_images):
            first_img = raw_images[i]
            first_stem = first_img.stem.upper()
            
            if first_stem in edited_images:
                if i + num_brackets <= len(raw_images):
                    bracket_set = raw_images[i:i + num_brackets]
                    edited_path = edited_images[first_stem]
                    sets.append({
                        'folder': folder.name,
                        'folder_path': folder,
                        'raw_folder': raw_folder,
                        'edited_folder': edited_folder,
                        'raw_images': bracket_set,
                        'edited_image': edited_path,
                    })
                    i += num_brackets
                else:
                    i += 1
            else:
                i += 1
    
    return sets


def detect_transform(raw_path: Path, edited_path: Path, max_dim: int = 1200) -> Optional[np.ndarray]:
    """
    Detect the affine transformation from raw to edited image.
    
    Returns:
        2x3 affine transformation matrix, or None if detection failed
    """
    if not HAS_CV2:
        return None
    
    # Load images
    raw_img = cv2.imread(str(raw_path))
    edited_img = cv2.imread(str(edited_path))
    
    if raw_img is None or edited_img is None:
        return None
    
    # Convert to grayscale
    raw_gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited_img, cv2.COLOR_BGR2GRAY)
    
    # Get original dimensions
    raw_h, raw_w = raw_gray.shape
    edited_h, edited_w = edited_gray.shape
    
    # Resize for faster feature detection (keep aspect ratio)
    raw_scale = min(max_dim / max(raw_h, raw_w), 1.0)
    edited_scale = min(max_dim / max(edited_h, edited_w), 1.0)
    
    if raw_scale < 1.0:
        raw_gray = cv2.resize(raw_gray, None, fx=raw_scale, fy=raw_scale)
    if edited_scale < 1.0:
        edited_gray = cv2.resize(edited_gray, None, fx=edited_scale, fy=edited_scale)
    
    try:
        # Use ORB for feature detection (fast and robust)
        orb = cv2.ORB_create(nfeatures=2000)
        kp1, des1 = orb.detectAndCompute(raw_gray, None)
        kp2, des2 = orb.detectAndCompute(edited_gray, None)
        
        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            # Fall back to SIFT if ORB doesn't find enough features
            sift = cv2.SIFT_create(nfeatures=2000)
            kp1, des1 = sift.detectAndCompute(raw_gray, None)
            kp2, des2 = sift.detectAndCompute(edited_gray, None)
            
            if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
                return None
            
            # Use FLANN for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            matches = good_matches
        else:
            # Use BFMatcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            matches = good_matches
        
        if len(matches) < 10:
            return None
        
        # Get matched points (scale back to original coordinates)
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Scale points back to original image coordinates
        src_pts = src_pts / raw_scale
        dst_pts = dst_pts / edited_scale
        
        # Estimate affine transformation (allows rotation, scale, translation)
        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts, 
            method=cv2.RANSAC, 
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if M is None:
            return None
        
        # Check if transformation is reasonable
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        
        # Reject if rotation > 45 degrees or scale is way off
        if abs(angle) > 45 or scale < 0.5 or scale > 2.0:
            return None
        
        return M
        
    except Exception as e:
        print(f"Error detecting transform: {e}")
        return None


def apply_transform(image_path: Path, transform: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """
    Apply affine transformation to an image.
    
    Args:
        image_path: Path to input image
        transform: 2x3 affine transformation matrix
        output_size: (width, height) of output image
    
    Returns:
        Transformed image as numpy array
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Apply transformation
    transformed = cv2.warpAffine(
        img, 
        transform, 
        output_size,
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REFLECT_101
    )
    
    return transformed


def process_image_set(
    img_set: Dict, 
    output_base: Path, 
    quality: int = 95
) -> Dict:
    """
    Process a single image set: detect transform and align raw images.
    
    Returns:
        Result dictionary with status and details
    """
    result = {
        'folder': img_set['folder'],
        'status': 'unknown',
        'transform': None,
        'error': None
    }
    
    try:
        # Get paths
        edited_path = img_set['edited_image']
        raw_paths = img_set['raw_images']
        first_raw = raw_paths[0]
        
        # Create output directories
        output_folder = output_base / img_set['folder']
        output_raw = output_folder / 'raw'
        output_edited = output_folder / 'edited'
        output_raw.mkdir(parents=True, exist_ok=True)
        output_edited.mkdir(parents=True, exist_ok=True)
        
        # Copy edited image as-is
        shutil.copy2(edited_path, output_edited / edited_path.name)
        
        # Detect transformation
        transform = detect_transform(first_raw, edited_path)
        
        if transform is None:
            # No transform detected - just copy raw images
            result['status'] = 'copied'
            for raw_path in raw_paths:
                shutil.copy2(raw_path, output_raw / raw_path.name)
        else:
            # Get output size from edited image
            edited_img = cv2.imread(str(edited_path))
            output_size = (edited_img.shape[1], edited_img.shape[0])  # (width, height)
            
            # Extract transform info for logging
            angle = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
            result['transform'] = {
                'rotation_degrees': float(angle),
                'scale': float(np.sqrt(transform[0, 0]**2 + transform[1, 0]**2)),
                'tx': float(transform[0, 2]),
                'ty': float(transform[1, 2])
            }
            
            # Apply transformation to all raw images
            for raw_path in raw_paths:
                aligned = apply_transform(raw_path, transform, output_size)
                
                # Save with high quality
                output_path = output_raw / raw_path.name
                cv2.imwrite(
                    str(output_path), 
                    aligned, 
                    [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
            
            result['status'] = 'aligned'
        
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        
        # On error, try to copy files as-is
        try:
            output_folder = output_base / img_set['folder']
            output_raw = output_folder / 'raw'
            output_edited = output_folder / 'edited'
            output_raw.mkdir(parents=True, exist_ok=True)
            output_edited.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(img_set['edited_image'], output_edited / img_set['edited_image'].name)
            for raw_path in img_set['raw_images']:
                shutil.copy2(raw_path, output_raw / raw_path.name)
        except:
            pass
    
    return result


def align_dataset(dataset_path: Path, output_path: Path, quality: int = 95, workers: int = 4):
    """
    Create aligned dataset.
    
    Args:
        dataset_path: Path to original dataset
        output_path: Path for aligned dataset output
        quality: JPEG quality for saved images
        workers: Number of parallel workers
    """
    print(f"Creating aligned dataset...")
    print(f"  Source: {dataset_path}")
    print(f"  Output: {output_path}")
    print(f"  Quality: {quality}")
    print(f"  Workers: {workers}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all image sets
    image_sets = get_image_sets(dataset_path)
    print(f"\nFound {len(image_sets)} image sets to process")
    
    if len(image_sets) == 0:
        print("No image sets found!")
        return
    
    # Process sets
    results = {
        'aligned': 0,
        'copied': 0,
        'error': 0,
        'details': []
    }
    
    # Use thread pool for parallel processing
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_image_set, img_set, output_path, quality): img_set 
            for img_set in image_sets
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            result = future.result()
            results[result['status']] = results.get(result['status'], 0) + 1
            results['details'].append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT COMPLETE")
    print("="*60)
    print(f"Total sets processed: {len(image_sets)}")
    print(f"  Aligned (transform applied): {results['aligned']}")
    print(f"  Copied (no transform needed): {results['copied']}")
    print(f"  Errors: {results['error']}")
    print(f"\nAligned dataset saved to: {output_path}")
    
    # Save detailed report
    report_path = output_path / "alignment_report.json"
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed report: {report_path}")
    
    # Show some alignment stats
    aligned_details = [r for r in results['details'] if r['status'] == 'aligned' and r['transform']]
    if aligned_details:
        rotations = [abs(r['transform']['rotation_degrees']) for r in aligned_details]
        print(f"\nAlignment statistics:")
        print(f"  Min rotation: {min(rotations):.2f}°")
        print(f"  Max rotation: {max(rotations):.2f}°")
        print(f"  Avg rotation: {sum(rotations)/len(rotations):.2f}°")


def check_dataset(dataset_path: Path, report_path: str):
    """Check dataset for issues and generate report"""
    print(f"Checking dataset: {dataset_path}")
    
    image_sets = get_image_sets(dataset_path)
    print(f"Found {len(image_sets)} image sets")
    
    report = {
        'total_sets': len(image_sets),
        'issues': [],
        'sets_with_rotation': 0,
        'sets_with_size_mismatch': 0,
        'sets_ok': 0
    }
    
    for img_set in tqdm(image_sets, desc="Checking alignment"):
        first_raw = img_set['raw_images'][0]
        edited = img_set['edited_image']
        
        # Check dimensions
        raw_img = Image.open(first_raw)
        edited_img = Image.open(edited)
        raw_size = raw_img.size
        edited_size = edited_img.size
        
        size_match = raw_size == edited_size
        
        # Check rotation
        transform = detect_transform(first_raw, edited)
        
        has_issue = False
        issue_details = {
            'folder': img_set['folder'],
            'first_raw': str(first_raw),
            'edited': str(edited)
        }
        
        if not size_match:
            has_issue = True
            report['sets_with_size_mismatch'] += 1
            issue_details['size_issue'] = {
                'raw_size': raw_size,
                'edited_size': edited_size
            }
        
        if transform is not None:
            angle = np.arctan2(transform[1, 0], transform[0, 0]) * 180 / np.pi
            if abs(angle) > 0.5:
                has_issue = True
                report['sets_with_rotation'] += 1
                issue_details['rotation'] = {
                    'rotation_degrees': float(angle),
                    'scale': float(np.sqrt(transform[0, 0]**2 + transform[1, 0]**2))
                }
        
        if has_issue:
            report['issues'].append(issue_details)
        else:
            report['sets_ok'] += 1
    
    # Save report
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("DATASET REPORT")
    print("="*60)
    print(f"Total image sets: {report['total_sets']}")
    print(f"Sets OK: {report['sets_ok']}")
    print(f"Sets with size mismatch: {report['sets_with_size_mismatch']}")
    print(f"Sets with rotation: {report['sets_with_rotation']}")
    print(f"\nFull report saved to: {report_path}")
    
    if report['sets_with_rotation'] > 0:
        print("\n" + "!"*60)
        print("WARNING: Rotated images detected!")
        print("Run with --align to create an aligned dataset:")
        print(f'  python prepare_dataset.py --align --data "{dataset_path}" --output "path/to/aligned"')
        print("!"*60)
    
    return report


def main():
    args = parse_args()
    dataset_path = Path(args.data)
    
    if not dataset_path.exists():
        print(f"ERROR: Dataset path does not exist: {dataset_path}")
        return
    
    if args.check:
        if not HAS_CV2:
            print("WARNING: OpenCV not available. Install for rotation detection.")
        check_dataset(dataset_path, args.report)
    
    if args.align:
        if not HAS_CV2:
            print("ERROR: OpenCV is required for alignment.")
            print("Install with: pip install opencv-python")
            return
        
        if args.output is None:
            print("ERROR: --output is required for alignment mode")
            return
        
        output_path = Path(args.output)
        align_dataset(dataset_path, output_path, args.quality, args.workers)


if __name__ == "__main__":
    main()
