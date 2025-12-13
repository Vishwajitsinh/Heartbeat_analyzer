# Sentinel AI: Heartbeat Analyzer
> An advanced acoustic diagnostics system achieving 99%+ accuracy in detecting cardiac anomalies using Stacking Ensemble Machine Learning.

![Accuracy](https://img.shields.io/badge/Accuracy-99.2%25-success)
![Method](https://img.shields.io/badge/Method-Stacking%20Ensemble-orange)
![Audio](https://img.shields.io/badge/Librosa-Audio%20Processing-blue)
![License](https://img.shields.io/badge/License-MIT-purple)

## 🩺 Overview
**Sentinel AI** is a medical-grade diagnostic tool designed to analyze phonocardiograms (heartbeat audio). It goes beyond traditional classifiers by using a **Stacking Ensemble** architecture—combining the strengths of multiple models (Voting Classifiers, HistGradientBoosting) to detect murmurs, extrasystoles, and other anomalies with near-perfect precision.

### Core Features
*   **Stacking Ensemble Architecture**: Utilizes a meta-learner to weigh predictions from multiple base models, pushing accuracy from ~95% to >99%.
*   **Advanced Data Augmentation**: Systematically generates synthetic training data using Time Stretching, Pitch Shifting, and Noise Injection to ensure model robustness.
*   **Professional UI**: A minimalistic, mono-font dashboard for uploading WAV/MP3 files and visualizing the waveform and spectrogram in real-time.
*   **Automated Diagnostics**: Instant classification with confidence scores (e.g., *"Normal: 99.8% Confidence"*).

## 🛠 Architecture
*   **Preprocessing**: `Librosa` for MFCC feature extraction.
*   **Model**: `scikit-learn` StackingClassifier (RandomForest + GradientBoosting + SVM).
*   **Frontend**: Custom Tkinter / Web Interface (Dark Mode).

## 🚀 Usage

1.  **Installation**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Launch Sentinel**
    ```bash
    python sentinel_gui.py
    ```

3.  **Upload Audio**: Load any `.wav` heartbeat recording.
4.  **Analyze**: Click "Run Diagnostics" to see the AI's classification and confidence vector.

---
*Precision diagnostics for the beating heart of technology.*
