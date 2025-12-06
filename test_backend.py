import requests
import os

url = 'http://127.0.0.1:5000/analyze'
files = []

# Try to find the sample files generated earlier
sample_dir = 'samples'
sample_file = os.path.join(sample_dir, 'normal_heartbeat.wav')

if os.path.exists(sample_file):
    print(f"Testing with {sample_file}...")
    files.append(('audio', ('normal_heartbeat.wav', open(sample_file, 'rb'), 'audio/wav')))
else:
    print("Sample file not found, creating a dummy file...")
    with open('test.wav', 'wb') as f:
        f.write(b'RIFF....WAVEfmt ...') # Invalid wav but enough to test upload
    files.append(('audio', ('test.wav', open('test.wav', 'rb'), 'audio/wav')))

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
