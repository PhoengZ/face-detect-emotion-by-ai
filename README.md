# Face Emotion Detection by AI

## Description

This project implements a facial emotion detection system using Deep Learning. It is designed to classify facial expressions into **7 distinct emotions**: anger, disgust, fear, happiness, neutral, sad, and surprised.

The core of the project is a Convolutional Neural Network (CNN) trained on the **FER2013** dataset, capable of processing 48x48 grayscale images to predict the subject's emotion.

## Tech Stack

- **Language**: Python
- **Deep Learning Framework**: PyTorch
- **Computer Vision**: Torchvision
- **Data Management**: Hugging Face Datasets
- **Utilities**: Tqdm (for progress tracking)

## Training Technique

The project uses a custom CNN architecture (`EmotionCNN`) trained from scratch.

### Model Architecture

- **Convolutional Layers**: 3 layers with increasing channel depth (32 -> 64 -> 128), each followed by Batch Normalization and ReLU activation.
- **Pooling**: Max Pooling layers to downsample feature maps.
- **Fully Connected Layers**: Two linear layers with Dropout (50%) to prevent overfitting.

### Training Details

- **Dataset**: FER2013 (via `clip-benchmark/wds_fer2013`).
- **Preprocessing**:
  - Resize to 48x48 pixels.
  - Convert to Grayscale (1 channel).
  - Normalization (mean=0.5, std=0.5).
- **Hyperparameters**:
  - **Loss Function**: CrossEntropyLoss.
  - **Optimizer**: Adam (Learning Rate = 0.001).
  - **Epochs**: 50.

## How to Clone

You can clone this repository to your local machine using the following command:

```bash
git clone https://github.com/PhoengZ/face-detect-emotion-by-ai.git
cd face-detect-emotion-by-ai
```

## How to Use

### 1. Prerequisites

Make sure you have Python installed. You will need to install the following dependencies:

```bash
pip install torch torchvision datasets tqdm
```

### 2. Running the Project

The project is structured as a Jupyter Notebook.

1.  Open `detect_image.ipynb` in your preferred notebook environment (VS Code, Jupyter Lab, or Google Colab).
2.  Run the cells sequentially to:
    - Load and preprocess the dataset.
    - Define and initialize the `EmotionCNN` model.
    - Train the model for 50 epochs.
    - Evaluate the model accuracy on the test set.

### 3. Output

After training, the model weights will be saved to `emotion_model.pth` in the current directory.
