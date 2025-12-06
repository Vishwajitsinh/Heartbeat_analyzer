import os
import requests
import zipfile

# Determine save path
dataset_dir = 'training_data'
os.makedirs(dataset_dir, exist_ok=True)

# URL for a subset of the PhysioNet 2016 Challenge (Training Set A)
# This is a zip file containing ~400 heart sound recordings with labels.
zip_url = "https://physionet.org/static/published-projects/challenge-2016/training-a.zip"
zip_path = os.path.join(dataset_dir, "training-a.zip")

print(f"Downloading Training Data from {zip_url}...")
try:
    r = requests.get(zip_url, stream=True)
    if r.status_code == 200:
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download complete.")
        
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
        print("Extraction complete.")
        
    else:
        print(f"Failed to download. Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
