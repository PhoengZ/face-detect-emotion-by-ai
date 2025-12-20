# Face Emotion Detection by AI

## 1. Description of this project

This project is an AI-based system designed to detect human emotions from facial expressions. It utilizes a deep learning approach, specifically a Convolutional Neural Network (CNN), trained on the **FER2013** dataset (Face Expression Recognition). The model classifies images into one of 7 emotion categories.

## 2. Tech Stack

- **Language**: Python
- **Deep Learning Framework**: PyTorch
- **Vision Libraries**: Torchvision
- **Data Handling**: Hugging Face Datasets (`datasets` library)
- **Utilities**: Tqdm (for progress bars)

## 3. Techniques used to train emotion_detect model

The model is trained using several key Deep Learning and Computer Vision techniques:

- **Convolutional Neural Network (CNN)**: The core architecture used to automatically learn spatial hierarchies of features from input images.
- **Data Preprocessing & Normalization**:
  - Images are resized to `48x48` pixels.
  - Converted to **Grayscale** (1 channel) as color is less critical for feature patterns in this specific task.
  - Normalized with mean `0.5` and standard deviation `0.5`.
- **Data Augmentation**: To prevent overfitting and improve generalization:
  - `RandomHorizontalFlip`: Randomly flips images horizontally.
  - `RandomRotation`: Rotates images by up to 15 degrees.
- **Regularization**:
  - **Batch Normalization**: Applied after convolution layers to stabilize and accelerate training.
  - **Dropout**: Applied in the fully connected layers (p=0.5) to prevent the model from relying too heavily on specific neurons.
- **Optimization**:
  - Optimizer: **Adam** with a learning rate of `0.001`.
  - Loss Function: **CrossEntropyLoss** (standard for multi-class classification).

## 4. Components of ML Pipeline

The project is structured into modular components:

- **`src/data_loader.py`**:
  - **Role**: Data Ingestion & Transformation.
  - **Description**: Loads the FER2013 dataset from Hugging Face. It defines the `FER2013Dataset` class and creates training and testing `DataLoader` instances with the defined transformations.
- **`src/model.py`**:
  - **Role**: Model Architecture Definition.
  - **Description**: Defines the `EmotionCNN` class. It specifies the 3 convolutional layers, max-pooling layers, batch normalization, and fully connected layers.
- **`src/train.py`**:
  - **Role**: Training Orchestrator.
  - **Description**: Contains the training loop. It iterates through epochs, calculates loss, performs backpropagation (`loss.backward()`), updates weights, and evaluates the model on the test set. It saves the best model as `best_model.pth` based on accuracy.

## 5. How does it work

1.  **Input**: The system takes a 48x48 pixel grayscale image of a face.
2.  **Feature Extraction**: The image passes through 3 blocks of **Convolutional Layers**. Each layer applies filters (kernels) to detect features like edges, textures, and curves.
3.  **Downsampling**: **Max Pooling** layers reduce the spatial dimensions (e.g., 48x48 -> 24x24 -> 12x12 -> 6x6) while retaining important information.
4.  **Activation**: **ReLU** (Rectified Linear Unit) activation functions are used to introduce non-linearity, allowing the model to learn complex patterns.
5.  **Classification**: The flattened features are passed through **Fully Connected (Linear) Layers**.
6.  **Output**: The final layer outputs a probability distribution across the 7 emotion classes (e.g., angry, disgust, fear, happy, sad, surprise, neutral). The class with the highest score is the predicted emotion.

## 6. How to clone this project

To download this project to your local machine, run the following command in your terminal:

```bash
git clone <repository_url>
cd face-detect-emotion-by-ai
```

_(Note: Replace `<repository_url>` with the actual URL of this repository)_

## 7. How to use this project

1.  **Set up the environment**:
    Ensure you have Python installed. It is recommended to create a virtual environment.

2.  **Install Dependencies**:

    ```bash
    pip install -r requirement.txt
    ```

3.  **Train the Model**:
    Run the training script to start training the model from scratch.

    ```bash
    python src/train.py
    ```

    This will download the dataset, train the model for 100 epochs (default), and save the weight with the highest accuracy to `best_model.pth`.

4.  **Inference / Testing**:
    You can use the Jupyter Notebook `detect_image.ipynb` to load the trained model and test it on new images.

## 8. What did I learn from doing this project

Through the development of this project, several key Deep Learning concepts were reinforced:

- **CNN Mechanics**: Understanding how **Kernel Size** (frame for moving over pixels), **Stride**, and **Padding** work together to process images and maintain or reduce dimensions.
- **Layer Functions**:
  - **Conv2d**: How to extract features from 2D image data.
  - **MaxPool2d**: How to downsample images (decompose 48x48 to 24x24) to reduce computation and focus on dominant features.
  - **Linear (Fully Connected)**: How to map extracted features to final class scores.
- **Activation & Regularization**:
  - **ReLU**: Importance of non-linearity in deep networks.
  - **Dropout**: The concept of "forgetting" or zeroing out random elements to force the model to learn more robust features (preventing overfitting).
- **Training Loop Implementation**: constructing a manual training loop in PyTorch, managing batches, computing gradients, and updating parameters.

### 9.Note

Dataset based on FER-2013 (Goodfellow et al., 2013), accessed via clip-benchmark/wds_fer2013.
Dataset Source: [clip-benchmark/wds_fer2013](https://huggingface.co/datasets/clip-benchmark/wds_fer2013)
