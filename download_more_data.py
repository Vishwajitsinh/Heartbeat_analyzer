import requests
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Robust Session
session = requests.Session()
retry = Retry(connect=3, backoff_factor=1)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

download_dir = 'medical_data'
os.makedirs(download_dir, exist_ok=True)

# List of training sets available in the PhysioNet 2016 challenge
training_sets = ['training-a', 'training-b', 'training-c', 'training-d', 'training-e', 'training-f']

base_url_template = "https://physionet.org/files/challenge-2016/1.0.0/{}/"

for dataset in training_sets:
    current_base_url = base_url_template.format(dataset)
    print(f"\nScanning {dataset}...")

    # 1. Download REFERENCE.csv for this set
    try:
        r = session.get(current_base_url + "RECORDS", timeout=10)
        if r.status_code != 200:
            print(f"Skipping {dataset} (Records not found)")
            continue
            
        records = r.text.strip().split('\n')
        print(f"Found {len(records)} records in {dataset}.")
        
        # Merge References Safely
        # We read the content and append it to our master file immediately
        # We use a try-except block specifically for the file write
        ref_r = session.get(current_base_url + "REFERENCE.csv", timeout=10)
        if ref_r.status_code == 200:
            ref_content = ref_r.text.strip()
            
            # Helper to append safely
            ref_path = os.path.join(download_dir, "REFERENCE_ALL.csv")
            
            try:
                # If file doesn't exist, create it. If it does, append.
                # Ensure we add a newline separator
                if not os.path.exists(ref_path):
                    with open(ref_path, 'w') as f:
                        f.write(ref_content + "\n")
                else:
                    # Read existing to ensure we don't start on a weird line? 
                    # Simpler: Just open in 'a' mode and ensure leading newline if needed.
                    with open(ref_path, 'a') as f:
                        f.write("\n" + ref_content + "\n")
            except PermissionError:
                print(f"Warning: Could not write to REFERENCE_ALL.csv (Permission Denied). Is the file open?")
            except Exception as file_err:
                print(f"Warning: Failed to update REFERENCE_ALL.csv: {file_err}")

    except Exception as e:
        print(f"Error fetching metadata for {dataset}: {e}")
        continue

    # 2. Download Audio Files
    count = 0
    total = len(records)
    
    for record in records:
        record = record.strip()
        if not record: continue
        
        wav_name = record + ".wav"
        hea_name = record + ".hea"
        
        # Check if exists to skip
        if os.path.exists(os.path.join(download_dir, wav_name)):
            count += 1
            # print(f"[{dataset}] Skipping {record} (Exists)", end='\r')
            continue
            
        # Download WAV
        try:
            r_wav = session.get(current_base_url + wav_name, timeout=10)
            if r_wav.status_code == 200:
                with open(os.path.join(download_dir, wav_name), 'wb') as f:
                    f.write(r_wav.content)
        except:
             pass
                
        # Download HEA
        try:
            r_hea = session.get(current_base_url + hea_name, timeout=10)
            if r_hea.status_code == 200:
                with open(os.path.join(download_dir, hea_name), 'wb') as f:
                    f.write(r_hea.content)
        except:
            pass
                
        count += 1
        print(f"[{dataset}] Downloaded {count}/{total}: {record}", end='\r')

print(f"\nAll downloads complete!")
