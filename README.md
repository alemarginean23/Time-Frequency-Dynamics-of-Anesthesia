# Time-Frequency Dynamics of Anesthesia: An Explainable CNN Framework for Local Field Potentials

This repository contains the complete computational pipeline for classifying isoflurane concentration states (depth of anesthesia) from multichannel Local Field Potential (LFP) recordings using Convolutional Neural Networks (CNNs) and Explainable AI (Grad-CAM).

---

## 🔬 Preprocessing & Feature Extraction Pipeline

The raw LFP recordings are processed through a rigorous, leakage-free pipeline designed to extract time-frequency representations optimized for deep learning:

1. **Temporal Segmentation (Sliding Window):**
   * Extracts the first 120 seconds of spontaneous brain activity per recording.
   * Applies a sliding window of **2.0 seconds** with a **0.4-second step** (yielding an 80% temporal overlap).
   * Generates 296 trial-based windows per dataset (totaling 888 trials across the 3 anesthetic depth classes).

2. **Multichannel Selection:**
   * Processes multichannel LFP recordings sampled at **1000 Hz**.
   * Excludes noisy or artifact-prone electrodes (`EXCLUDED_CHANNELS = {1, 22, 33}`), retaining **30 active parallel channels**.

3. **Time-Frequency Analysis (STFT):**
   * Computes the Short-Time Fourier Transform (STFT) using a segment length (`nperseg`) of 256 and an overlap (`noverlap`) of 128.
   * **Frequency Capping:** Restricts the spectrum to a maximum frequency of **300 Hz**, resulting in exactly **77 frequency bins**.
   * **Temporal Steps:** Each 2-second trial yields **14 temporal steps**.
   * **Log-Scaling:** Applies a logarithmic transformation (`log1p`) to compress dynamic range and stabilize variance.

4. **Final Tensor Representation:**
   * Combines all dimensions into a unified 4D input tensor of shape **(Trials, Frequencies, Time Steps, Channels)** $\rightarrow$ **(888, 77, 14, 30)**.

5. **Data Augmentation & Normalization:**
   * **SpecAugment:** Applies dynamic time and frequency masking exclusively to the training subset during cross-validation to prevent data leakage and improve model generalization.
   * **Z-Score Normalization:** Standardizes input features based on training fold statistics.

---

## 🏗️ Project Structure & Architecture
- **Pipeline Script (`train_anesthesia_pipeline.py`)**: Handles automated sliding window generation, STFT feature extraction, Stratified 3-fold Cross-Validation, model training, and evaluation metrics (Confusion Matrix, F1-score).
- **CNN Architecture**: A custom 2D Convolutional Neural Network featuring L2 regularization, Batch Normalization, Dropout, and a designated target layer (`xai_target_conv`) tailored for post-hoc Grad-CAM interpretability visualizations.

---

## 📦 Dependencies & Requirements

To run this project, ensure you have Python installed along with the following packages:

```text
tensorflow>=2.10.0
numpy>=1.22.0
scipy>=1.8.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
