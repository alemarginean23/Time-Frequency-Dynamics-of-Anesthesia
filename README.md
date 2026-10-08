# Time-Frequency Dynamics of Anesthesia: An Explainable CNN Framework for Local Field Potentials

This repository contains the complete computational pipeline for classifying isoflurane concentration states (depth of anesthesia) from multichannel Local Field Potential (LFP) recordings using Convolutional Neural Networks (CNNs) and Explainable AI (Grad-CAM).

---

## Preprocessing & Feature Extraction Pipeline

The raw LFP recordings are processed through a robust pipeline designed to extract time-frequency representations optimized for deep learning:

1. **Temporal Segmentation (Sliding Window):**
   * Extracts the first 120 seconds of spontaneous brain activity per recording session (`SAMPLE_RATE = 1000 Hz`, `DURATION_SEC = 120`).
   * Applies a sliding window of **2.0 seconds** (`WINDOW_SEC = 2.0`) with a step size of **0.9 seconds** (`STEP_SEC = 0.9`).
   * Yields a balanced dataset across the 3 anesthetic depth classes (`0002` - Deep, `0003` - Intermediate, `0004` - Light).

2. **Multichannel Selection:**
   * Processes multichannel LFP recordings.
   * Excludes noisy or artifact-prone electrodes (`EXCLUDED_CHANNELS = {1, 22, 33}`), retaining **30 active parallel channels**.

3. **Time-Frequency Analysis (STFT):**
   * Computes the Short-Time Fourier Transform (STFT) using a segment length (`nperseg`) of 256 and an overlap (`noverlap`) of 128.
   * **Frequency Capping:** Restricts the spectrum to a maximum frequency of **300 Hz**, resulting in exactly **77 frequency bins**.
   * **Log-Scaling:** Applies a logarithmic transformation (`log1p`) to compress dynamic range and stabilize variance.

4. **Tensor Representation:**
   * Combines all dimensions into a unified 3D input tensor per trial of shape **(Frequencies, Time Steps, Channels)** $\rightarrow$ **(77, 14, 30)**.

5. **Data Augmentation & Normalization:**
   * **SpecAugment:** Applies dynamic time masking ($1\text{--}2$ steps) and frequency masking ($1\text{--}5$ bands) exclusively to the training subset during cross-validation to prevent data leakage and improve model generalization (`N_AUG = 2`).
   * **Z-Score Normalization:** Dynamically standardizes input features based on training fold statistics.

---

## Project Structure & Notebooks

- **`cnnAug+testSet.ipynb`**: The primary machine learning pipeline notebook handling automated sliding window generation, STFT feature extraction, SpecAugment data augmentation, 20% external hold-out test set splitting, Stratified 3-fold Cross-Validation, model training with class weighting, and comprehensive evaluation metrics (Confusion Matrices, Validation & Loss curves).
- **`gradcam.ipynb`**: Generates advanced Explainable AI (XAI) visualizations. It combines the background spectrogram on the **Green channel** and the Grad-CAM attention heatmap on the **Red channel** via additive RGB synthesis. In the resulting overlay, background activity appears green, model attention appears red, and true neurophysiological decision drivers appear in sharp **Yellow** (Red + Green).
- **`spec.ipynb`**: A companion notebook dedicated to visual inspection and generation of global 120-second session LFP spectrogram plots utilizing `jet` colormap and percentile-based contrast scaling (`vmin` at 1%, `vmax` at 98%) to highlight high-frequency dynamics.

---

## CNN Model Architecture

The custom 2D Convolutional Neural Network is structured as follows:
* **Block 1:** Conv2D (8 filters, $3 \times 3$ kernel, padding='same', L2 regularization) $\rightarrow$ Batch Normalization $\rightarrow$ MaxPooling2D $(2 \times 2)$ $\rightarrow$ Dropout (0.3).
* **Block 2 (XAI Target Layer):** Conv2D (`xai_target_conv`, 16 filters, $3 \times 3$ kernel, L2 regularization) $\rightarrow$ Batch Normalization $\rightarrow$ MaxPooling2D $(2 \times 2)$ $\rightarrow$ Dropout (0.4).
* **Classification Head:** Global Average Pooling $\rightarrow$ Dense (16 units, ReLU, L2 reg) $\rightarrow$ Dropout (0.4) $\rightarrow$ Dense (3 units, Softmax).
* **Optimizer & Callbacks:** Adam optimizer (`lr = 0.0005`), sparse categorical cross-entropy loss, `EarlyStopping` (patience=15), and `ReduceLROnPlateau` (factor=0.3, patience=5).

---

## Dependencies & Requirements

To run this project, ensure you have Python installed along with the following packages:

```text
tensorflow>=2.10.0
numpy>=1.22.0
scipy>=1.8.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
