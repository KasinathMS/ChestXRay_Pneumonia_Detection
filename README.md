# Pneumonia Detection using MobileNetV2

This project implements a deep learning solution for detecting pneumonia from chest X-ray images. It leverages transfer learning with a pre-trained MobileNetV2 model, fine-tuned on a custom dataset of chest X-rays, to achieve high accuracy in classification.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Streamlit Application](#streamlit-application)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)

## Project Overview
Pneumonia is a serious lung infection, and early diagnosis is crucial for effective treatment. This project aims to automate the detection of pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture. The model is trained to classify X-ray images as either 'NORMAL' or 'PNEUMONIA'.

## Features
- **Transfer Learning**: Utilizes MobileNetV2, a state-of-the-art pre-trained model, as a feature extractor.
- **Custom Classification Head**: A custom classification layer is added on top of MobileNetV2 for binary classification.
- **Data Augmentation**: Employs `ImageDataGenerator` for real-time image augmentation to prevent overfitting and improve model generalization.
- **Early Stopping**: Implements `EarlyStopping` callback to monitor validation loss and stop training when performance plateaus, restoring the best model weights.
- **Comprehensive Evaluation**: Provides a detailed classification report and confusion matrix to assess model performance.

## Dataset
The project uses a dataset of chest X-ray images categorized into 'NORMAL' and 'PNEUMONIA' classes. The dataset is organized into `train`, `test`, and `val` (validation) directories. Each directory contains `NORMAL` and `PNEUMONIA` subdirectories.

**Data Source**: [(Kaggle Chest X-Ray Images (Pneumonia) dataset)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Model Architecture
The model consists of:
1.  **MobileNetV2 Base**: Loaded without its top classification layers, with weights pre-trained on ImageNet. The layers of this base model are frozen during initial training.
2.  **Custom Layers**:
    - A `Flatten` layer to convert the 3D output of MobileNetV2 into a 1D vector.
    - A `Dropout` layer (with a 0.5 dropout rate) for regularization.
    - A `Dense` output layer with a single neuron and `sigmoid` activation for binary classification.

The model is compiled with the `adam` optimizer and `binary_crossentropy` loss function, monitoring `accuracy`.

## Results
The trained model achieved the following performance metrics on the test dataset:
- **Accuracy**: Approximately 90%
- **Precision (NORMAL)**: 0.87
- **Recall (NORMAL)**: 0.88
- **F1-Score (NORMAL)**: 0.87
- **Precision (PNEUMONIA)**: 0.93
- **Recall (PNEUMONIA)**: 0.92
- **F1-Score (PNEUMONIA)**: 0.92

**Confusion Matrix:**
```
[[206  28]
 [ 32 358]]
```

These results indicate that the model performs well in distinguishing between normal and pneumonia cases, with a slightly better performance in detecting pneumonia.

## Streamlit Application
A Streamlit application has been developed to provide an interactive interface for performing pneumonia detection. Users can upload a chest X-ray image, and the application will use the trained model to predict whether the image shows signs of pneumonia or not.

**To run the Streamlit app:**
1.  Ensure you have Streamlit installed (`pip install streamlit`).
2.  Navigate to the directory containing your `streamlit_app.py` (or similar) file.
3.  Run the command: `streamlit run your_app_name.py`

## Setup and Installation
To set up and run this project, follow these steps:

1.  **Clone the repository** (if applicable):
    ```bash
    git clone https://github.com/KasinathMS/ChestXRay_Pneumonia_Detection.git
    cd ChestXRay_Pneumonia_Detection
    ```
2.  **Install dependencies**:
    ```bash
    pip install tensorflow keras numpy scikit-learn matplotlib seaborn
    ```
3.  **Download the Dataset**:
    Place the 'chest_xray' dataset (containing 'train', 'test', 'val' folders) in the specified `base_path` (e.g., `/content/drive/MyDrive/Pneumonia/chest_xray` if using Google Drive in Colab, or adjust the `base_path` variable accordingly).

## Usage
1.  **Open the Jupyter/Colab Notebook**: Run the notebook cells sequentially.
2.  **Data Loading and Preprocessing**: The notebook will load images using `ImageDataGenerator` and apply augmentations.
3.  **Model Training**: The model will be trained with early stopping.
4.  **Evaluation**: Review the training history plots, model evaluation metrics, confusion matrix, and classification report.
5.  **Save Model**: The trained model will be saved as `mobilenetv2.keras` for future use.
6.  **Run Streamlit App**: Use the Streamlit application to test the model with new images.
