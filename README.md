# 🧠 Brain Tumor Detector

A deep learning web application that detects brain tumors from MRI images using a Convolutional Neural Network (CNN).

## 🌐 Live Demo
👉 [Click here to try the app](https://saniya354-brain-tumor-dectector.hf.space)

## 📌 About the Project
This project uses a CNN model trained on brain MRI images to classify whether a tumor is present or not.

- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Deployment:** Streamlit + Hugging Face Spaces
- **Test Accuracy:** 97.8%

## 🗂️ Dataset
- Training images: 1600 (800 tumor, 800 no tumor)
- Testing images: 1600 (800 tumor, 800 no tumor)
- Image size: 128x128 pixels

## 🏗️ Model Architecture
- 3 Convolutional layers (32, 64, 128 filters)
- 3 MaxPooling layers
- 1 Flatten layer
- 1 Dense layer (128 neurons, ReLU)
- 1 Output layer (Sigmoid activation)

## 📊 Training Results
| Epoch | Train Accuracy | Val Accuracy |
|-------|---------------|--------------|
| 1     | 65.7%         | 92.1%        |
| 5     | 93.0%         | 92.9%        |
| 10    | 97.5%         | 97.8%        |


## 📦 Requirements
```
streamlit
tensorflow
numpy
Pillow
gdown
```
- Hugging Face: [saniya354](https://huggingface.co/saniya354)
- GitHub: [saniya3540](https://github.com/saniya3540)
