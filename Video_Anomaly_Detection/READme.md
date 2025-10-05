# Video Anomaly Detection Using Deep Learning

**Research Internship Project | IIIT Allahabad**

A deep learning-based video anomaly detection system that combines Convolutional Neural Networks (CNN) and Recurrent Neural Networks (RNN) to identify anomalous events in video sequences with 91.24% accuracy.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Results](#results)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Future Work](#future-work)

## Overview

This project implements a hybrid CNN-RNN architecture for automated video anomaly detection. The system processes video sequences frame-by-frame, extracting spatial features using convolutional layers and modeling temporal dependencies through recurrent networks to classify normal and anomalous behavior.

**Research Institution:** Indian Institute of Information Technology, Allahabad  
**Domain:** Computer Vision, Deep Learning, Video Analysis

## Key Features

- **Hybrid Architecture**: Combines CNN for spatial feature extraction and RNN for temporal modeling
- **Sequence Processing**: Handles variable-length video sequences (60 frames per sequence)
- **Real-time Capable**: Optimized for efficient inference on video streams
- **High Accuracy**: Achieved 91.24% test accuracy with robust F1 and AUC scores
- **Scalable Design**: Modular architecture supporting multiple anomaly classes

## Technical Architecture

### Model Pipeline

```
Input Video Sequence (60 frames, 240×240×3)
    ↓
TimeDistributed CNN Layers
    • Conv2D (32 filters, 3×3) + ReLU + MaxPooling
    • Conv2D (64 filters, 3×3) + ReLU + MaxPooling
    ↓
Flatten & Reshape
    ↓
SimpleRNN Layer (128 units)
    ↓
Dense Layer (128 units) + Dropout (0.5)
    ↓
Output Layer (Softmax - 10 classes)
```

### Key Components

1. **Spatial Feature Extractor**: TimeDistributed CNN layers process each frame independently
2. **Temporal Encoder**: SimpleRNN captures temporal patterns across frame sequences
3. **Classifier**: Fully connected layers with dropout for regularization
4. **Data Generator**: Custom Keras sequence generator for efficient batch processing

### Training Configuration

- **Optimizer**: Adam (learning rate: 0.0001)
- **Loss Function**: Categorical Cross-Entropy
- **Batch Size**: 8
- **Sequence Length**: 60 frames
- **Frame Resolution**: 240×240×3
- **Callbacks**: 
  - Early Stopping (patience: 5)
  - Learning Rate Reduction (factor: 0.5, patience: 3)

## Results

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 91.24% |
| **Test Loss** | 0.3578 |
| **F1 Score (Macro)** | 0.7258 |
| **AUC Score** | 0.9568 |

### Model Strengths

- High accuracy and AUC score demonstrate excellent classification performance
- Strong generalization with minimal overfitting
- Balanced precision-recall trade-off (F1 score: 0.73)
- Robust performance across multiple anomaly classes

## Dataset

The model is trained on a custom video dataset organized as follows:

```
Dataset/
├── Train/
│   ├── Class_0/
│   │   ├── video_001/
│   │   │   ├── frame_001.jpg
│   │   │   ├── frame_002.jpg
│   │   │   └── ...
│   ├── Class_1/
│   └── ...
└── Test/
    └── (same structure)
```

- **Classes**: 10 distinct anomaly/normal behavior categories
- **Preprocessing**: Frames normalized to [0, 1] range
- **Format**: RGB images resized to 240×240 pixels

## Installation

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
scikit-learn
matplotlib
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/video-anomaly-detection.git
cd video-anomaly-detection

# Prepare dataset
# Place your video frames in Dataset/Train and Dataset/Test directories
```

## Usage

### Training the Model

```bash
python source.py
```

The script will:
1. Load and preprocess video sequences
2. Train the CNN-RNN model
3. Evaluate on test set
4. Generate performance metrics and visualizations

### Configuration

Modify these parameters in `source.py` as needed:

```python
sequence_length = 60      # Number of frames per sequence
frame_size = (240, 240, 3)  # Frame dimensions
batch_size = 8            # Training batch size
num_classes = 10          # Number of anomaly classes
```

## Model Performance

The trained model demonstrates:

- **Convergence**: Stable training with early stopping mechanism
- **Generalization**: Minimal gap between training and validation metrics
- **Robustness**: High AUC score indicates strong discriminative capability
- **Efficiency**: Optimized architecture for reasonable training time

## Technical Contributions

1. **Architecture Design**: Developed hybrid CNN-RNN model tailored for video anomaly detection
2. **Data Pipeline**: Implemented efficient custom data generator for video sequence processing
3. **Performance Optimization**: Applied learning rate scheduling and early stopping strategies
4. **Comprehensive Evaluation**: Multi-metric assessment (Accuracy, F1, AUC) for thorough model validation

## Future Work

- [ ] Implement attention mechanisms for improved temporal modeling
- [ ] Explore 3D CNN architectures (C3D, I3D) for joint spatiotemporal learning
- [ ] Add LSTM/GRU layers for better long-term dependency capture
- [ ] Integrate transfer learning with pre-trained models (ResNet, VGG)
- [ ] Deploy as REST API for real-time anomaly detection
- [ ] Add visualization tools for attention maps and anomaly localization
- [ ] Experiment with unsupervised/semi-supervised approaches
- [ ] Optimize for edge deployment (model quantization, pruning)

## Technologies Used

[TensorFlow]
[Keras]
[Python]
[NumPy]
[scikit-learn]

## Author

**Solanki Dharak Deepak**  
Research Intern | IIIT Allahabad  

## License

This project was developed as part of a research internship at IIIT Allahabad.

## Acknowledgments

- IIIT Allahabad for providing research facilities and guidance
- Department of Computer Science for computational resources
- Research supervisor and mentors for technical guidance

---

**Note**: This project demonstrates practical application of deep learning in computer vision, specifically for video surveillance and security applications.
