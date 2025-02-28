import zipfile
import os

# Define the directory containing the zip files
zip_dir = '../data/raw'

# Define directories for extraction
cem_extract_dir = '../data/raw/cem'
point_extract_dir = '../data/raw/point'

# Create the extraction directories if they don't exist
os.makedirs(cem_extract_dir, exist_ok=True)
os.makedirs(point_extract_dir, exist_ok=True)

# List all files in the directory
files = os.listdir(zip_dir)
print("Files in directory:", files)  # Debugging line

for file_name in files:
    # Check if the file is a zip file
    if file_name.endswith('.zip'):
        print(f"Processing {file_name}")  # Debugging line
        # Construct full file path
        file_path = os.path.join(zip_dir, file_name)
        
        # Open the zip file
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            # List contents of the zip file
            print("Contents of the zip file:", zip_ref.namelist())  # Debugging line
            
            # Determine the extraction directory based on the file name
            if 'cem' in file_name:
                extract_dir = cem_extract_dir
            elif 'point' in file_name:
                extract_dir = point_extract_dir
            else:
                print(f"Unknown file type for {file_name}, skipping.")
                continue
            
            # Extract all files to the corresponding directory
            for member in zip_ref.namelist():
                zip_ref.extract(member, extract_dir)
                print(f'Extracted {member} to {extract_dir}')