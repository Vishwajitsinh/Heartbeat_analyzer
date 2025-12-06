import numpy as np
import soundfile as sf
import os
import scipy.signal as signal

# Create samples directory
output_dir = 'samples'
os.makedirs(output_dir, exist_ok=True)

def generate_heartbeat(duration_sec, bpm, irregularity=0.0, sr=22050):
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    audio = np.zeros_like(t)
    
    # Calculate intervals
    bps = bpm / 60.0
    avg_interval = 1.0 / bps
    
    current_time = 0
    beat_times = []
    
    while current_time < duration_sec:
        # Add some randomness for irregularity
        interval = avg_interval + np.random.normal(0, irregularity * avg_interval)
        if current_time + interval > duration_sec:
            break
        current_time += interval
        beat_times.append(current_time)
        
    # Create "Lub-Dub" sound for each beat
    # S1 (Lub) is longer and lower frequency, S2 (Dub) is shorter and slightly higher
    for beat_t in beat_times:
        # Indices for S1 and S2
        s1_start = int(beat_t * sr)
        s2_start = int((beat_t + 0.3) * sr) # S2 comes shortly after S1
        
        # Determine length of sounds (approx 0.1s)
        len_s = int(0.1 * sr)
        
        if s1_start + len_s < len(audio):
            # Create a simple low-freq thud (damped sine wave)
            t_wave = np.linspace(0, 0.1, len_s)
            s1 = np.sin(2 * np.pi * 50 * t_wave) * np.exp(-t_wave * 20)
            audio[s1_start:s1_start+len_s] += s1 * 0.8 # Amplitude
            
        if s2_start + len_s < len(audio):
            t_wave = np.linspace(0, 0.1, len_s)
            s2 = np.sin(2 * np.pi * 70 * t_wave) * np.exp(-t_wave * 30)
            audio[s2_start:s2_start+len_s] += s2 * 0.6 # S2 is usually quieter/sharper

    # Normalize
    audio = audio / np.max(np.abs(audio))
    return audio, sr

print("Generating synthetic heartbeats...")

# 1. Normal Heartbeat (70 BPM, very steady)
print("Creating: Normal Heartbeat (70 BPM)")
normal_audio, sr = generate_heartbeat(duration_sec=10, bpm=70, irregularity=0.02)
sf.write(os.path.join(output_dir, 'normal_heartbeat.wav'), normal_audio, sr)

# 2. Tachycardia (120 BPM, steady)
print("Creating: Tachycardia (120 BPM)")
fast_audio, sr = generate_heartbeat(duration_sec=10, bpm=120, irregularity=0.02)
sf.write(os.path.join(output_dir, 'tachycardia.wav'), fast_audio, sr)

# 3. Irregular / Arrhythmia (70 BPM avg, high variability)
print("Creating: Irregular Arrhythmia")
irr_audio, sr = generate_heartbeat(duration_sec=10, bpm=70, irregularity=0.4) # 40% variance
sf.write(os.path.join(output_dir, 'arrhythmia.wav'), irr_audio, sr)

print(f"Done! Files saved to {os.path.abspath(output_dir)}")
