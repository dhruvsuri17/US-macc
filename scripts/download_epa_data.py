import os
import requests
import zipfile
from pathlib import Path
import time
import urllib3
import certifi

# Disable SSL warnings (only use this for trusted government sites)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define URLs for the two files
base_url = "https://gaftp.epa.gov/Air/emismod/2022/v1/2022emissions/"
point_inventory_url = base_url + "2022hc_point_inventory_2022v1_13sep2024_v0.zip"
cem_inventory_url = base_url + "2022hc_cem_inventory_2022v1_08jul2024_v0.zip"

# Create data/raw directory if it doesn't exist
# Using Path for better cross-platform compatibility
data_dir = Path("../data/raw")
data_dir.mkdir(parents=True, exist_ok=True)

# Function to download and extract a file
def download_and_extract(url, directory):
    # Get filename from URL
    filename = url.split("/")[-1]
    filepath = directory / filename
    
    print(f"Downloading {filename}...")
    start_time = time.time()
    
    # Stream the download to handle large files efficiently
    # Added verify=False to bypass SSL certificate verification
    response = requests.get(url, stream=True, verify=False)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                # Print progress
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(f"\rProgress: {percent:.1f}% ({downloaded/(1024*1024):.1f} MB / {total_size/(1024*1024):.1f} MB)", end="")
    
    elapsed = time.time() - start_time
    print(f"\nDownloaded {filename} in {elapsed:.1f} seconds")
    
    # Extract the zip file
    print(f"Extracting {filename}...")
    extract_start = time.time()
    
    # Create a directory for extraction with the same name as the zip (without .zip)
    extract_dir = directory / filename.replace(".zip", "")
    extract_dir.mkdir(exist_ok=True)
    
    with zipfile.ZipFile(filepath, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    extract_elapsed = time.time() - extract_start
    print(f"Extracted to {extract_dir} in {extract_elapsed:.1f} seconds")
    
    return extract_dir

# Download and extract both files
point_dir = download_and_extract(point_inventory_url, data_dir)
cem_dir = download_and_extract(cem_inventory_url, data_dir)

print("\nDownload and extraction complete!")
print(f"Point inventory files extracted to: {point_dir}")
print(f"CEM inventory files extracted to: {cem_dir}")