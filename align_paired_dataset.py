"""
Paired Dataset Alignment Script

Aligns edited images (trainB) to match source images (trainA).
Detects and fixes crop/rotation differences.

Usage:
    python align_paired_dataset.py --source "X:/trainA" --edited "X:/trainB" --output "X:/trainB_Aligned"
"""

import argparse
import os
import re
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
    print("ERROR: OpenCV is required. Install with: pip install opencv-python")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Align paired dataset images")
    
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to source images directory (trainA)"
    )
    parser.add_argument(
        "--edited", type=str, required=True,
        help="Path to edited images directory (trainB)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for aligned edited images"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="JPEG quality for saved images (default: 95)"
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel workers (default: 4)"
    )
    parser.add_argument(
        "--max-dim", type=int, default=1200,
        help="Max dimension for feature detection (default: 1200)"
    )
    parser.add_argument(
        "--report", type=str, default="alignment_report.json",
        help="Output file for alignment report"
    )
    
    return parser.parse_args()


def get_image_pairs(source_path: Path, edited_path: Path) -> List[Dict]:
    """
    Find matching image pairs between source and edited folders.
    Matches by filename (case-insensitive).
    """
    pairs = []
    
    # Get all images in source folder
    source_images = {}
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        for img in source_path.glob(ext):
            source_images[img.stem.upper()] = img
    
    # Get all images in edited folder and match
    for ext in ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']:
        for img in edited_path.glob(ext):
            stem_upper = img.stem.upper()
            if stem_upper in source_images:
                pairs.append({
                    'name': img.stem,
                    'source': source_images[stem_upper],
                    'edited': img
                })
    
    # Sort by name
    def extract_number(name: str) -> int:
        match = re.search(r'(\d+)', name)
        return int(match.group(1)) if match else 0
    
    pairs.sort(key=lambda x: extract_number(x['name']))
    
    return pairs


def detect_transform(source_path: Path, edited_path: Path, max_dim: int = 1200) -> Optional[np.ndarray]:
    """
    Detect the affine transformation from source to edited image.
    
    Returns:
        2x3 affine transformation matrix, or None if detection failed
    """
    # Load images
    source_img = cv2.imread(str(source_path))
    edited_img = cv2.imread(str(edited_path))
    
    if source_img is None or edited_img is None:
        return None
    
    # Convert to grayscale
    source_gray = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
    edited_gray = cv2.cvtColor(edited_img, cv2.COLOR_BGR2GRAY)
    
    # Get original dimensions
    source_h, source_w = source_gray.shape
    edited_h, edited_w = edited_gray.shape
    
    # Resize for faster feature detection
    source_scale = min(max_dim / max(source_h, source_w), 1.0)
    edited_scale = min(max_dim / max(edited_h, edited_w), 1.0)
    
    if source_scale < 1.0:
        source_gray = cv2.resize(source_gray, None, fx=source_scale, fy=source_scale)
    if edited_scale < 1.0:
        edited_gray = cv2.resize(edited_gray, None, fx=edited_scale, fy=edited_scale)
    
    try:
        # Try ORB first (faster)
        orb = cv2.ORB_create(nfeatures=3000)
        kp1, des1 = orb.detectAndCompute(source_gray, None)
        kp2, des2 = orb.detectAndCompute(edited_gray, None)
        
        use_sift = False
        if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
            use_sift = True
        
        if use_sift:
            # Fall back to SIFT
            sift = cv2.SIFT_create(nfeatures=3000)
            kp1, des1 = sift.detectAndCompute(source_gray, None)
            kp2, des2 = sift.detectAndCompute(edited_gray, None)
            
            if des1 is None or des2 is None or len(kp1) < 20 or len(kp2) < 20:
                return None
            
            # FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            
            # Ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
            matches = good_matches
        else:
            # BFMatcher for ORB
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = bf.knnMatch(des1, des2, k=2)
            
            # Ratio test
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
        
        # Scale back
        src_pts = src_pts / source_scale
        dst_pts = dst_pts / edited_scale
        
        # Estimate affine transformation (source -> edited)
        M, mask = cv2.estimateAffinePartial2D(
            src_pts, dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if M is None:
            return None
        
        # Validate transformation
        angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
        scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        
        # Reject if rotation > 45 degrees or scale is unreasonable
        if abs(angle) > 45 or scale < 0.3 or scale > 3.0:
            return None
        
        return M
        
    except Exception as e:
        print(f"Error detecting transform: {e}")
        return None


def align_edited_to_source(
    source_path: Path,
    edited_path: Path,
    output_path: Path,
    quality: int = 95,
    max_dim: int = 1200
) -> Dict:
    """
    Align edited image to match source image dimensions and orientation.
    
    The edited image will be transformed to match the source's frame.
    """
    result = {
        'name': edited_path.stem,
        'status': 'unknown',
        'transform': None,
        'error': None
    }
    
    try:
        # Load images to get dimensions
        source_img = cv2.imread(str(source_path))
        edited_img = cv2.imread(str(edited_path))
        
        if source_img is None:
            result['status'] = 'error'
            result['error'] = f"Could not load source: {source_path}"
            return result
        
        if edited_img is None:
            result['status'] = 'error'
            result['error'] = f"Could not load edited: {edited_path}"
            return result
        
        source_h, source_w = source_img.shape[:2]
        edited_h, edited_w = edited_img.shape[:2]
        
        # Detect transformation (source -> edited)
        M = detect_transform(source_path, edited_path, max_dim)
        
        if M is None:
            # No transform detected - check if sizes match
            if source_h == edited_h and source_w == edited_w:
                # Same size, just copy
                cv2.imwrite(str(output_path), edited_img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                result['status'] = 'copied'
            else:
                # Different sizes but no transform found - resize to match
                resized = cv2.resize(edited_img, (source_w, source_h), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(str(output_path), resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
                result['status'] = 'resized'
                result['transform'] = {
                    'type': 'resize',
                    'from': [edited_w, edited_h],
                    'to': [source_w, source_h]
                }
        else:
            # Calculate inverse transform (edited -> source frame)
            # M transforms source -> edited, we need edited -> source
            M_inv = cv2.invertAffineTransform(M)
            
            # Apply inverse transform to edited image
            aligned = cv2.warpAffine(
                edited_img,
                M_inv,
                (source_w, source_h),
                flags=cv2.INTER_LANCZOS4,
                borderMode=cv2.BORDER_REFLECT_101
            )
            
            # Save aligned image
            cv2.imwrite(str(output_path), aligned, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Extract transform info
            angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi
            scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            
            result['status'] = 'aligned'
            result['transform'] = {
                'type': 'affine',
                'rotation_degrees': float(angle),
                'scale': float(scale),
                'tx': float(M[0, 2]),
                'ty': float(M[1, 2])
            }
    
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
    
    return result


def process_pair(pair: Dict, output_path: Path, quality: int, max_dim: int) -> Dict:
    """Process a single image pair."""
    output_file = output_path / pair['edited'].name
    return align_edited_to_source(
        pair['source'],
        pair['edited'],
        output_file,
        quality,
        max_dim
    )


def main():
    args = parse_args()
    
    source_path = Path(args.source)
    edited_path = Path(args.edited)
    output_path = Path(args.output)
    
    # Validate paths
    if not source_path.exists():
        print(f"ERROR: Source path does not exist: {source_path}")
        return
    
    if not edited_path.exists():
        print(f"ERROR: Edited path does not exist: {edited_path}")
        return
    
    print("="*60)
    print("PAIRED DATASET ALIGNMENT")
    print("="*60)
    print(f"Source (reference): {source_path}")
    print(f"Edited (to align):  {edited_path}")
    print(f"Output:             {output_path}")
    print(f"Quality:            {args.quality}")
    print(f"Workers:            {args.workers}")
    print("="*60)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find image pairs
    pairs = get_image_pairs(source_path, edited_path)
    print(f"\nFound {len(pairs)} matching image pairs")
    
    if len(pairs) == 0:
        print("No matching pairs found! Check that filenames match between folders.")
        return
    
    # Process pairs
    results = {
        'aligned': 0,
        'copied': 0,
        'resized': 0,
        'error': 0,
        'details': []
    }
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pair, pair, output_path, args.quality, args.max_dim): pair
            for pair in pairs
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Aligning"):
            result = future.result()
            status = result['status']
            results[status] = results.get(status, 0) + 1
            results['details'].append(result)
    
    # Print summary
    print("\n" + "="*60)
    print("ALIGNMENT COMPLETE")
    print("="*60)
    print(f"Total pairs processed: {len(pairs)}")
    print(f"  Aligned (transform applied): {results['aligned']}")
    print(f"  Copied (already aligned):    {results['copied']}")
    print(f"  Resized (size mismatch):     {results['resized']}")
    print(f"  Errors:                      {results['error']}")
    print(f"\nAligned images saved to: {output_path}")
    
    # Save report
    report_path = output_path / args.report
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Report saved to: {report_path}")
    
    # Show alignment statistics
    aligned_details = [r for r in results['details'] if r['status'] == 'aligned' and r.get('transform')]
    if aligned_details:
        rotations = [abs(r['transform']['rotation_degrees']) for r in aligned_details]
        scales = [r['transform']['scale'] for r in aligned_details]
        print(f"\nAlignment statistics:")
        print(f"  Rotation - Min: {min(rotations):.2f}°, Max: {max(rotations):.2f}°, Avg: {sum(rotations)/len(rotations):.2f}°")
        print(f"  Scale    - Min: {min(scales):.3f}, Max: {max(scales):.3f}, Avg: {sum(scales)/len(scales):.3f}")
    
    # Show errors if any
    errors = [r for r in results['details'] if r['status'] == 'error']
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for err in errors[:5]:
            print(f"  - {err['name']}: {err['error']}")
        if len(errors) > 5:
            print(f"  ... and {len(errors)-5} more (see report)")


if __name__ == "__main__":
    main()

