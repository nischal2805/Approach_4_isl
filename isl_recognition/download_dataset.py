#!/usr/bin/env python3
"""
ISL Dataset Download Script

Downloads the INCLUDE-50 dataset from Kaggle or Zenodo.
Supports multiple sources for redundancy.

Usage:
    python download_dataset.py --source kaggle --output data/INCLUDE50
    python download_dataset.py --source zenodo --output data/INCLUDE50
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
from pathlib import Path
import urllib.request
import shutil

# Dataset sources
KAGGLE_DATASET = "sttaseen/include-50-isl-dataset"  # Update if different slug
ZENODO_URL = "https://zenodo.org/record/4010759/files/INCLUDE.zip"

def download_kaggle(output_dir: str):
    """Download INCLUDE-50 from Kaggle."""
    print("=" * 60)
    print("Downloading INCLUDE-50 from Kaggle...")
    print("=" * 60)
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.run([sys.executable, "-m", "pip", "install", "kaggle"], check=True)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download dataset
    cmd = f"kaggle datasets download -d {KAGGLE_DATASET} -p {output_dir} --unzip"
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        print("\nMake sure you have:")
        print("1. Created a Kaggle account")
        print("2. Downloaded your API key from https://www.kaggle.com/settings")
        print("3. Placed kaggle.json in ~/.kaggle/ (Linux) or C:\\Users\\<user>\\.kaggle\\ (Windows)")
        return False
    
    print(f"Dataset downloaded to: {output_dir}")
    return True


def download_zenodo(output_dir: str):
    """Download INCLUDE dataset from Zenodo."""
    print("=" * 60)
    print("Downloading INCLUDE from Zenodo...")
    print("=" * 60)
    
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "INCLUDE.zip")
    
    # Download with progress
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
    
    print(f"Downloading from: {ZENODO_URL}")
    try:
        urllib.request.urlretrieve(ZENODO_URL, zip_path, progress_hook)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False
    
    # Extract
    print("Extracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_dir)
        os.remove(zip_path)
        print(f"Extracted to: {output_dir}")
        return True
    except Exception as e:
        print(f"Error extracting: {e}")
        return False


def download_manual_instructions():
    """Print manual download instructions."""
    print("=" * 60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("""
If automatic download fails, you can manually download the dataset:

1. INCLUDE-50 (Kaggle):
   - Go to: https://www.kaggle.com/datasets/sttaseen/include-50-isl-dataset
   - Click "Download" button
   - Extract to: data/INCLUDE50/

2. INCLUDE (Zenodo):
   - Go to: https://zenodo.org/record/4010759
   - Download the INCLUDE.zip file
   - Extract to: data/INCLUDE50/

3. Direct Download Script (Zenodo):
   - Run: http://bit.ly/include_dl
   - This is a bash script that downloads all parts

Expected folder structure after extraction:
data/INCLUDE50/
├── Class_1/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
├── Class_2/
│   └── ...
└── ...
""")


def verify_dataset(dataset_dir: str) -> dict:
    """Verify downloaded dataset and show statistics."""
    print("=" * 60)
    print("Verifying Dataset...")
    print("=" * 60)
    
    dataset_path = Path(dataset_dir)
    
    if not dataset_path.exists():
        print(f"Error: {dataset_dir} does not exist")
        return None
    
    # Find all video files
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    classes = {}
    
    for item in dataset_path.iterdir():
        if item.is_dir():
            class_name = item.name
            videos = [f for f in item.iterdir() 
                     if f.is_file() and f.suffix.lower() in video_extensions]
            if videos:
                classes[class_name] = len(videos)
    
    if not classes:
        # Try one level deeper
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                for item in subdir.iterdir():
                    if item.is_dir():
                        class_name = item.name
                        videos = [f for f in item.iterdir() 
                                 if f.is_file() and f.suffix.lower() in video_extensions]
                        if videos:
                            classes[class_name] = len(videos)
    
    if not classes:
        print("No video classes found!")
        print("Expected structure: dataset_dir/class_name/video.mp4")
        return None
    
    total_videos = sum(classes.values())
    print(f"Found {len(classes)} classes with {total_videos} total videos")
    print(f"\nSample classes (first 10):")
    for i, (cls, count) in enumerate(sorted(classes.items())[:10]):
        print(f"  {cls}: {count} videos")
    
    if len(classes) > 10:
        print(f"  ... and {len(classes) - 10} more classes")
    
    # Check class balance
    counts = list(classes.values())
    min_count = min(counts)
    max_count = max(counts)
    avg_count = sum(counts) / len(counts)
    
    print(f"\nClass balance:")
    print(f"  Min videos per class: {min_count}")
    print(f"  Max videos per class: {max_count}")
    print(f"  Avg videos per class: {avg_count:.1f}")
    
    return {
        "num_classes": len(classes),
        "total_videos": total_videos,
        "classes": classes,
        "min_per_class": min_count,
        "max_per_class": max_count,
    }


def main():
    parser = argparse.ArgumentParser(description="Download ISL dataset")
    parser.add_argument("--source", choices=["kaggle", "zenodo", "manual"], 
                       default="kaggle", help="Download source")
    parser.add_argument("--output", default="data/INCLUDE50", 
                       help="Output directory")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing dataset")
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.output)
        return
    
    success = False
    
    if args.source == "kaggle":
        success = download_kaggle(args.output)
    elif args.source == "zenodo":
        success = download_zenodo(args.output)
    else:
        download_manual_instructions()
        return
    
    if success:
        verify_dataset(args.output)
    else:
        print("\nAutomatic download failed. Trying alternative method...")
        download_manual_instructions()


if __name__ == "__main__":
    main()
