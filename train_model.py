import os
import glob
import numpy as np
import librosa
import csv
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Configuration
DATA_DIR = 'medical_data'
MODEL_PATH = 'heartbeat_model.pkl'
SCALER_PATH = 'scaler.pkl'

# Settings for Data Augmentation
CHUNK_DURATION = 2.0 # seconds
OVERLAP = 0.5 # 50% overlap

def extract_features_from_audio(y, sr):
    try:
        # 1. MFCCs + Deltas (Texture)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        
        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs.T, axis=0)
        delta_std = np.std(delta_mfccs.T, axis=0)
        
        # 2. Spectral (Timbre/Brightness)
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        cent_mean = np.mean(cent)
        cent_std = np.std(cent)
        
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        rolloff_mean = np.mean(rolloff)
        rolloff_std = np.std(rolloff)
        
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast.T, axis=0)
        
        # 3. Time-Domain (Rhythm/Energy)
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        
        # Tempo/Beat Features (Only if enough duration)
        if len(y) > sr:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            if isinstance(tempo, np.ndarray): tempo = tempo[0]
        else:
            tempo = 0
            
        features = np.hstack([
            mfccs_mean, mfccs_std, 
            delta_mean, delta_std,
            cent_mean, cent_std, 
            rolloff_mean, rolloff_std, 
            zcr_mean, zcr_std,
            contrast_mean,
            rms_mean, rms_std,
            tempo
        ])
        return features
    except Exception as e:
        return None

def process_file_chunks(file_path):
    # Splits file into multiple labelled chunks for data augmentation
    feats_list = []
    try:
        y_full, sr = librosa.load(file_path, sr=22050) # Load entire file
        total_duration = librosa.get_duration(y=y_full, sr=sr)
        
        if total_duration < CHUNK_DURATION:
            # If too short, pad it
            y_full = librosa.util.fix_length(y_full, size=int(CHUNK_DURATION*sr))
            
        # Create Chunks
        chunk_length = int(CHUNK_DURATION * sr)
        step = int(chunk_length * (1 - OVERLAP))
        
        for start in range(0, len(y_full) - chunk_length + 1, step):
            y_chunk = y_full[start : start + chunk_length]
            f = extract_features_from_audio(y_chunk, sr)
            if f is not None:
                feats_list.append(f)
                
        # Also always include the full file features as one sample to capture long-term structure? 
        # Actually, let's stick to chunks to make all samples comparable in timeframe.
        
        return feats_list
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def train_model():
    print("Starting Advanced Model Training (Chunking + SMOTE)...")
    
    # Load Labels
    labels_map = {}
    try:
        ref_path = os.path.join(DATA_DIR, 'REFERENCE.csv')
        # Download logic skipped for brevity, assumed present
        with open(ref_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    labels_map[row[0]] = int(row[1]) 
    except Exception as e:
        print(f"Labels error: {e}")
        return

    wav_files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    print(f"Found {len(wav_files)} source files.")
    
    X = []
    y = []
    
    print("Extracting features from chunks...")
    for i, wav_path in enumerate(wav_files):
        filename = os.path.splitext(os.path.basename(wav_path))[0]
        if filename in labels_map:
            print(f"Processing {i}/{len(wav_files)}: {filename}", end='\r')
            
            # Extract MULTIPLE samples from ONE file
            chunk_features = process_file_chunks(wav_path)
            
            # 0=Normal, 1=Abnormal
            label = 0 if labels_map[filename] == -1 else 1
            
            for feat in chunk_features:
                X.append(feat)
                y.append(label)
                
    print(f"\nTotal Training Samples (Chunks): {len(X)}")
    
    X = np.array(X)
    y = np.array(y)
    
    # Train/Test Split (Before SMOTE to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE Balancing (Synthetic sampling of minority class)
    print("Applying SMOTE balancing...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"Balanced Dataset: {len(y_train_bal)} samples.")

    from sklearn.ensemble import StackingClassifier, HistGradientBoostingClassifier

    # SUPER ENSEMBLE (Stacking)
    # Stacking learns the optimal combination of models rather than just voting
    
    # Level 0 Learners
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1, random_state=42)),
        ('et', ExtraTreesClassifier(n_estimators=300, max_depth=25, n_jobs=-1, random_state=42)),
        ('hgb', HistGradientBoostingClassifier(max_iter=300, random_state=42)) # Powerful LightGBM-like
    ]
    
    # Meta Learner
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    
    print("Training Stacking Ensemble (This may take a moment)...")
    clf.fit(X_train_bal, y_train_bal)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    print("\nFINAL EVALUATION (On Unseen Chunks):")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Abnormal']))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    score = accuracy_score(y_test, y_pred)
    print(f"Chunk Accuracy: {score:.2f}")
    
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print("Model Saved.")

if __name__ == "__main__":
    train_model()
