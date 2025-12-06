from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import os
import json
import wfdb
import soundfile as sf
import scipy.signal as signal

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def analyze_heartbeat(file_path):
    print(f"Analyzing file: {file_path}")
    ext = os.path.splitext(file_path)[1].lower()
    
    y = None
    sr = 22050
    
    try:
        if ext == '.dat' or ext == '.hea':
            # WFDB Handling
            # Verify if paired file exists
            base_path = os.path.splitext(file_path)[0]
            if os.path.exists(base_path + '.dat') and os.path.exists(base_path + '.hea'):
                try:
                    record = wfdb.rdrecord(base_path)
                    # Use the first channel
                    signal_data = record.p_signal[:, 0]
                    sr = record.fs
                    
                    # Normalize to -1 to 1
                    y = signal_data / np.max(np.abs(signal_data))
                    
                    # If sampling rate is too low (e.g. 360Hz for ECG), we might need to resample 
                    # for it to "sound" right if played, but for analysis, we keep it.
                    # However, librosa functions expect audio-like SR. 
                    # Let's resample to 22050 only if we want to use librosa onset strength?
                    # Actually standard heart sounds are low freq. 
                    # Let's assume we proceed with the data as is, but convert to float32
                    y = y.astype(np.float32)

                    # For playback, we might want to save a temporary WAV version
                    # But the frontend just receives data. The frontend is playing 'fileURL'.
                    # This is tricky because the browser can't play .dat.
                    # We might need to convert it to wav and return a URL or base64?
                    # For now, let's focus on analysis.
                    
                except Exception as w_err:
                    raise ValueError(f"WFDB Error: {str(w_err)}")
            else:
                # This is the specific error that user likely hits when uploading only one file
                raise ValueError("For .dat files, you MUST upload both the .dat and .hea files together.")
        else:
            # Standard Audio Loading
            try:
                y, sr = librosa.load(file_path, sr=22050, duration=30)
            except Exception as load_err:
                 raise ValueError(f"Audio Format Error: {str(load_err)}. Try a standard WAV file.")
    except Exception as e:
        # Re-raise to be caught by the route handler
        raise e

    if y is None:
        raise ValueError("Failed to load signal data.")

    # Detect onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    # --- AUTOCORRELATION METHOD (Self-Matching) ---
    # This is superior for heartbeats because it finds the fundamental repeating period
    # regardless of whether the beat has 1 sound (Lub) or 2 sounds (Lub-Dub).
    # It avoids the "Double Counting" error.
    
    # Calculate autocorrelation
    # Max lag: 4 seconds (corresponds to 15 BPM min) - broad enough for bradycardia
    max_lag = int(4 * sr / 512)
    ac = librosa.autocorrelate(onset_env, max_size=max_lag)
    
    # Find the first major peak after the immediate self-match (lag 0)
    # We look for a peak in the realistic heart rate range: 40 BPM to 200 BPM
    # 200 BPM = 0.3s lag
    # 40 BPM = 1.5s lag
    min_lag = int(0.3 * sr / 512)
    max_lag_search = int(1.5 * sr / 512)
    
    if len(ac) > max_lag_search:
        # Search window
        search_region = ac[min_lag:max_lag_search]
        if len(search_region) > 0:
            peak_lag_relative = np.argmax(search_region)
            peak_lag = min_lag + peak_lag_relative
            
            # Convert lag to BPM
            beat_duration = librosa.frames_to_time(peak_lag, sr=sr)
            avg_bpm = 60 / beat_duration
        else:
            avg_bpm = 0
    else:
        avg_bpm = 0

    # --- HRV Calculation via Peak Refinement ---
    # Now that we know the TRUE BPM (e.g. 78, not 156), we can look for peaks
    # that are roughly `beat_duration` apart.
    
    # Expected distance between beats (in frames)
    if avg_bpm > 0:
        expected_dist = peak_lag
        # We allow some tolerance (e.g. +/- 30% for arrhythmia)
        min_dist = int(expected_dist * 0.6) 
    else:
        min_dist = int(0.3 * sr / 512)

    # Find peaks with this smarter distance constraint
    smoothed_env = np.convolve(onset_env, np.ones(5)/5, mode='same') # Slight smooth
    peaks, _ = signal.find_peaks(smoothed_env, height=np.mean(smoothed_env), distance=min_dist)
    
    peak_times = librosa.frames_to_time(peaks, sr=sr)
    rr_intervals = np.diff(peak_times) * 1000 # ms
    
    if len(rr_intervals) > 2:
        hrv_sdnn = np.std(rr_intervals)
        # If autocorrelation gave us a solid BPM, use it. 
        # Otherwise fall back to peak average if sensible.
        if avg_bpm == 0:
            avg_bpm = 60000 / np.mean(rr_intervals)
    else:
        hrv_sdnn = 0
        if avg_bpm == 0: avg_bpm = 0

    # --- MACHINE LEARNING DIAGNOSIS ---
    try:
        import joblib
        
        # Paths
        MODEL_PATH = 'heartbeat_model.pkl'
        SCALER_PATH = 'scaler.pkl'
        
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            # Load Model
            clf = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            
            # --- SEGMENTED INFERENCE (Matches new training logic) ---
            # Instead of one prediction for the whole file, we split it into 2s chunks,
            # predict on each, and take the average probability.
            
            CHUNK_DURATION = 2.0
            OVERLAP = 0.5
            
            total_duration = librosa.get_duration(y=y, sr=sr)
            if total_duration < CHUNK_DURATION:
                y = librosa.util.fix_length(y, size=int(CHUNK_DURATION*sr))
                
            chunk_length = int(CHUNK_DURATION * sr)
            step = int(chunk_length * (1 - OVERLAP))
            
            chunk_probs = []
            
            # Extract features for each chunk
            for start in range(0, len(y) - chunk_length + 1, step):
                y_chunk = y[start : start + chunk_length]
                
                # Copy-paste feature extraction from training script
                # 1. MFCCs + Deltas
                mfccs = librosa.feature.mfcc(y=y_chunk, sr=sr, n_mfcc=20)
                mfccs_mean = np.mean(mfccs.T, axis=0)
                mfccs_std = np.std(mfccs.T, axis=0)
                
                delta_mfccs = librosa.feature.delta(mfccs)
                delta_mean = np.mean(delta_mfccs.T, axis=0)
                delta_std = np.std(delta_mfccs.T, axis=0)
                
                # 2. Spectral
                cent = librosa.feature.spectral_centroid(y=y_chunk, sr=sr)
                cent_mean = np.mean(cent)
                cent_std = np.std(cent)
                
                rolloff = librosa.feature.spectral_rolloff(y=y_chunk, sr=sr)
                rolloff_mean = np.mean(rolloff)
                rolloff_std = np.std(rolloff)
                
                contrast = librosa.feature.spectral_contrast(y=y_chunk, sr=sr)
                contrast_mean = np.mean(contrast.T, axis=0)
                
                # 3. Time/Rhythm
                zcr = librosa.feature.zero_crossing_rate(y_chunk)
                zcr_mean = np.mean(zcr)
                zcr_std = np.std(zcr)
                
                rms = librosa.feature.rms(y=y_chunk)
                rms_mean = np.mean(rms)
                rms_std = np.std(rms)
                
                # Tempo (try/catch for short chunks)
                try:
                    onset_env_c = librosa.onset.onset_strength(y=y_chunk, sr=sr)
                    tempo_c = librosa.feature.tempo(onset_envelope=onset_env_c, sr=sr)
                    if isinstance(tempo_c, np.ndarray): tempo_c = tempo_c[0]
                except:
                    tempo_c = 0
                
                feat_vector = np.hstack([
                    mfccs_mean, mfccs_std, 
                    delta_mean, delta_std,
                    cent_mean, cent_std, 
                    rolloff_mean, rolloff_std, 
                    zcr_mean, zcr_std,
                    contrast_mean,
                    rms_mean, rms_std,
                    tempo_c
                ])
                
                feat_scaled = scaler.transform([feat_vector])
                prob = clf.predict_proba(feat_scaled)[0] # [prob_normal, prob_abnormal]
                chunk_probs.append(prob)
            
            # Aggregate Results
            if len(chunk_probs) > 0:
                avg_probs = np.mean(chunk_probs, axis=0) # [avg_normal, avg_abnormal]
                prob_abnormal = avg_probs[1]
                ml_pred = 1 if prob_abnormal > 0.55 else 0 # 0.55 threshold for safety
                ml_conf_score = max(avg_probs)
            else:
                ml_pred = 0
                prob_abnormal = 0
                ml_conf_score = 0
            
            # --- HYBRID VOTING SYSTEM ---
            
            # Rule-Based Flags
            rule_abnormal = False
            rule_reasons = []
            
            if avg_bpm > 100: 
                rule_abnormal = True
                rule_reasons.append("Tachycardia")
            if avg_bpm < 50 and avg_bpm > 0: 
                rule_abnormal = True
                rule_reasons.append("Bradycardia")
            if hrv_sdnn > 120: # Relaxed threshold because model is now trusted
                rule_abnormal = True
                rule_reasons.append("High Variability")

            # Final Decision Logic
            if ml_pred == 1:
                # ML Detected Abnormality
                if rule_abnormal:
                    diagnosis = "Abnormal Heart Sound Detected"
                    description = f"High Confidence ({int(prob_abnormal*100)}%). Both AI texture analysis and rhythm metrics ({', '.join(rule_reasons)}) indicate pathology."
                    status = "danger"
                else:
                    diagnosis = "Abnormal Texture / Murmur"
                    description = f"Rhythm is normal, but AI detects abnormal sound quality (e.g. Murmur, Valve issue) with {int(prob_abnormal*100)}% confidence."
                    status = "warning"
            else:
                # ML Says Normal
                if rule_abnormal:
                     diagnosis = "Arrhythmia (Rhythm Only)"
                     description = f"Sound quality is healthy, but rhythm is irregular: {', '.join(rule_reasons)}."
                     status = "warning"
                else:
                    diagnosis = "Normal Sinus Rhythm"
                    description = f"Healthy heartbeat confirmed. AI Analysis (Normal) and Rhythm Metrics are all within healthy ranges."
                    status = "healthy"
                
        else:
            # Fallback if model not found
            if hrv_sdnn > 100:
                diagnosis = "Irregular Rhythm / Signal Noise"
                description = "High variability detected (fallback mode)."
                status = "warning"
                
    except Exception as ml_err:
        print(f"ML Error: {ml_err}")
        # Keep default safe fallback
        if avg_bpm > 100 or avg_bpm < 50:
            status = "warning"
            
    # Normalize waveform
    target_points = 200
    step = max(1, len(y) // target_points)
    waveform = y[::step].tolist()

    return {
        "bpm": round(float(avg_bpm), 1),
        "hrv_sdnn": round(float(hrv_sdnn), 1),
        "diagnosis": diagnosis,
        "description": description,
        "status": status,
        "waveform": waveform
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files and 'audio[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('audio')
    if not files:
        files = request.files.getlist('audio[]')
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No selected file'}), 400

    saved_files = []
    target_file = None
    
    try:
        # Save all files
        for file in files:
            if file and file.filename:
                filepath = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(filepath)
                saved_files.append(filepath)
                
                # Pick the main file to analyze (.dat if present, otherwise the audio)
                ext = os.path.splitext(file.filename)[1].lower()
                if ext == '.dat' or (ext != '.hea' and target_file is None):
                    target_file = filepath
                    
        if not target_file:
            target_file = saved_files[0] # Fallback
            
        result = analyze_heartbeat(target_file)
        
        # Cleanup
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
                
        return jsonify(result)
        
    except Exception as e:
        # Cleanup on error
        for f in saved_files:
            if os.path.exists(f):
                os.remove(f)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
