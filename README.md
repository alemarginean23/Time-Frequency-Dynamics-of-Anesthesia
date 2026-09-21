# Time-Frequency Dynamics of Anesthesia: An Explainable CNN Framework for Local Field Potentials

This repository contains the complete computational pipeline for classifying isoflurane concentration states (depth of anesthesia) from multichannel Local Field Potential (LFP) recordings using Convolutional Neural Networks (CNNs) and Explainable AI (Grad-CAM).

## Project Structure
- **Notebook / Script**: Implements data loading, zero-phase bandpass filtering (0.5–300 Hz), downsampling, Short-Time Fourier Transform (STFT), CNN architecture training (Stratified 3-fold CV), and Grad-CAM interpretability visualizations.
- **Model**: Uses a custom 2D CNN architecture optimized for 3D time-frequency-channel tensors ($100 \times 1000 \times 31$).

## Note on Data
Due to institutional privacy and ethical guidelines regarding preclinical in vivo research data, the raw `.bin` recording files are **not publicly available** in this repository.
