"""
Debug utility to download the LFW dataset with detailed progress reporting.
"""

import os
import urllib.request
import tarfile
import random
import shutil
import time

# TODO: PLEASE REVIEW THIS
# This script was added to debug the download process for the LFW dataset

def reporthook(count, block_size, total_size):
    """
    Report hook for urllib to show download progress.
    """
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration)) if duration > 0 else 0
    percent = min(int(count * block_size * 100 / total_size), 100)
    
    # Calculate estimated time remaining
    if speed > 0:
        eta = (total_size - progress_size) / (speed * 1024)
    else:
        eta = 0
    
    # Convert bytes to MB for better readability
    progress_mb = progress_size / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    
    print(f"\rProgress: {percent}% ({progress_mb:.2f} MB / {total_mb:.2f} MB) Speed: {speed} KB/s ETA: {eta:.0f} sec", end="")

def download_and_extract_lfw_dataset(target_dir="../data/datasets/lfw", sample_size=None):
    """
    Download and extract a sample of the LFW dataset with detailed progress reporting.
    
    Args:
        target_dir (str): Directory to save the dataset
        sample_size (int, optional): Number of people to include (None for all)
        
    Returns:
        bool: True if successful, False otherwise
    """
    print("Starting LFW dataset download with detailed progress reporting...")
    
    # Ensure target directory exists
    os.makedirs(target_dir, exist_ok=True)
    print(f"Target directory: {os.path.abspath(target_dir)}")
    
    # URL for the LFW dataset
    lfw_url = "https://ndownloader.figshare.com/files/5976018"
    tgz_file = os.path.join(target_dir, "lfw.tgz")
    extract_dir = os.path.join(target_dir, "lfw")
    
    try:
        # Download the dataset if not already downloaded
        if not os.path.exists(tgz_file):
            print(f"Downloading LFW dataset from {lfw_url}...")
            print(f"Destination file: {os.path.abspath(tgz_file)}")
            
            # Check if target directory is writable
            try:
                test_file = os.path.join(target_dir, "test_write.tmp")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print("Target directory is writable.")
            except Exception as e:
                print(f"WARNING: Target directory may not be writable: {e}")
            
            try:
                urllib.request.urlretrieve(lfw_url, tgz_file, reporthook)
                print("\nDownload complete!")
            except Exception as e:
                print(f"\nError during download: {e}")
                return False
        else:
            print(f"File already exists: {os.path.abspath(tgz_file)}")
        
        # Verify the downloaded file
        if os.path.exists(tgz_file):
            file_size = os.path.getsize(tgz_file)
            print(f"Downloaded file size: {file_size / (1024*1024):.2f} MB")
            
            if file_size < 1000:  # Less than 1 KB is definitely an error
                print("Warning: Downloaded file is too small and might be corrupted.")
                os.remove(tgz_file)
                print("Deleted potential corrupted file. Please try again.")
                return False
        
        # Extract the dataset if not already extracted
        if not os.path.exists(extract_dir):
            print(f"Extracting dataset to {os.path.abspath(extract_dir)}...")
            try:
                with tarfile.open(tgz_file) as tar:
                    members = tar.getmembers()
                    num_members = len(members)
                    print(f"Archive contains {num_members} files")
                    
                    for i, member in enumerate(members):
                        if i % 500 == 0 or i == num_members - 1:
                            print(f"\rExtracting: {i+1}/{num_members} files ({(i+1)/num_members*100:.1f}%)", end="")
                        tar.extract(member, path=target_dir)
                    print("\nExtraction complete!")
            except Exception as e:
                print(f"\nError during extraction: {e}")
                return False
        else:
            print(f"Extraction directory already exists: {os.path.abspath(extract_dir)}")
            
        # Verify extraction
        if os.path.exists(extract_dir):
            try:
                dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
                print(f"Extracted directory contains {len(dirs)} person directories")
                if len(dirs) == 0:
                    print("Warning: No person directories found in the extracted dataset.")
                    return False
            except Exception as e:
                print(f"Error verifying extraction: {e}")
                return False
                
        # If sample_size is specified, create a random sample
        if sample_size is not None:
            print(f"Creating sample of {sample_size} people...")
            sample_dir = os.path.join(target_dir, "lfw_sample")
            
            try:
                # Get all person directories
                person_dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
                print(f"Found {len(person_dirs)} person directories in dataset")
                
                # Select a random sample
                if sample_size > len(person_dirs):
                    print(f"Warning: Requested sample size {sample_size} is larger than available persons {len(person_dirs)}")
                    sample_size = len(person_dirs)
                    print(f"Adjusted sample size to {sample_size}")
                
                selected_persons = random.sample(person_dirs, sample_size)
                print(f"Selected {len(selected_persons)} people for sample")
                
                # Create sample directory
                if os.path.exists(sample_dir):
                    print(f"Removing existing sample directory: {os.path.abspath(sample_dir)}")
                    shutil.rmtree(sample_dir)
                os.makedirs(sample_dir)
                print(f"Created sample directory: {os.path.abspath(sample_dir)}")
                
                # Copy selected person directories to sample directory
                for i, person in enumerate(selected_persons):
                    if i % 10 == 0 or i == len(selected_persons) - 1:
                        print(f"\rCopying person directories: {i+1}/{len(selected_persons)} ({(i+1)/len(selected_persons)*100:.1f}%)", end="")
                    
                    src_dir = os.path.join(extract_dir, person)
                    dst_dir = os.path.join(sample_dir, person)
                    shutil.copytree(src_dir, dst_dir)
                
                print(f"\nSample dataset created at {os.path.abspath(sample_dir)}")
            except Exception as e:
                print(f"\nError creating sample: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Run the download with a sample size of 10 people
    success = download_and_extract_lfw_dataset(sample_size=10)
    if success:
        print("Download and extraction completed successfully.")
    else:
        print("Download or extraction failed.")
