import requests
import os
import io

download_dir = 'medical_data'
os.makedirs(download_dir, exist_ok=True)
base_url = "https://physionet.org/files/challenge-2016/1.0.0/training-a/"

print("Downloading Full Dataset List...")
r = requests.get(base_url + "RECORDS")
records = r.text.strip().split('\n')
print(f"Found {len(records)} records to download.")

# Download Reference
print("Downloading Labels...")
r = requests.get(base_url + "REFERENCE.csv")
with open(os.path.join(download_dir, "REFERENCE.csv"), 'wb') as f:
    f.write(r.content)

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
        continue
        
    # Download WAV
    r_wav = requests.get(base_url + wav_name)
    if r_wav.status_code == 200:
        with open(os.path.join(download_dir, wav_name), 'wb') as f:
            f.write(r_wav.content)
            
    # Download HEA
    r_hea = requests.get(base_url + hea_name)
    if r_hea.status_code == 200:
        with open(os.path.join(download_dir, hea_name), 'wb') as f:
            f.write(r_hea.content)
            
    count += 1
    print(f"Downloaded {count}/{total}: {record}", end='\r')

print(f"\nDownload Complete! {count} files ready.")
