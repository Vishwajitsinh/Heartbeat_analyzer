import wfdb
import os
import requests

download_dir = 'medical_data'
os.makedirs(download_dir, exist_ok=True)

print("Attempting to download sample medical data from PhysioNet...")

try:
    # Attempt 1: Using WFDB to download MIT-BIH Arrhythmia Database (ECG) - Famous Standard
    # This is ECG, not Heart Sound, but it validates the WFDB connection.
    print("Downloading MIT-BIH Record 100 (ECG)...")
    wfdb.dl_files('mitdb', download_dir, ['100.dat', '100.hea'])
    print(" - Downloaded mitdb/100")

except Exception as e:
    print(f"WFDB Download Error: {e}")

try:
    # Attempt 2: Direct download of Heart Sound (PCG) from PhysioNet Challenge 2016
    # These are usually .wav files which my app supports natively!
    print("Downloading Heart Sound Record a0001 (PCG)...")
    
    base_url = "https://physionet.org/files/challenge-2016/1.0.0/training-a/"
    files = ["a0001.wav", "a0001.hea"]
    
    for filename in files:
        url = base_url + filename
        print(f" - Fetching {url}...")
        r = requests.get(url)
        if r.status_code == 200:
            with open(os.path.join(download_dir, filename), 'wb') as f:
                f.write(r.content)
            print(f" - Saved {filename}")
        else:
            print(f" - Failed to fetch {filename} (Status: {r.status_code})")
            
except Exception as e:
    print(f"Direct Download Error: {e}")

print(f"All operations complete. Check directory: {os.path.abspath(download_dir)}")
