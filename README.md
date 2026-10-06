# AI-Assisted Wound Image Classification

## Project Description

This project presents an AI-assisted wound image classification system developed using Digital Image Processing, Machine Learning, and Deep Learning techniques. The system uses a Convolutional Neural Network (CNN) to classify wound-related images into two categories:

- Ulcer
- Normal/Healthy Skin

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit

## Dataset

The dataset contains 1,055 wound-related images:

- Ulcer: 512 images
- Normal/Healthy Skin: 543 images

The images were used to train and validate the CNN classification model.

## Model

A Convolutional Neural Network (CNN) was developed using TensorFlow and Keras.

Image Input Size: 128 × 128 pixels  
Batch Size: 32  
Training Epochs: 10  
Optimizer: Adam  
Loss Function: Binary Cross-Entropy

The highest validation accuracy achieved during training was 93.33%.

## Application

A Streamlit-based user interface was developed for the project. Users can upload an image and obtain the predicted classification along with the confidence value.

## Project Files

- `train_model.py` – CNN model training code
- `app.py` – Streamlit application code
- `requirements.txt` – Required Python packages

## Disclaimer

This project is developed for educational and research purposes. It is an AI-assisted image classification system and should not be considered a replacement for professional medical diagnosis.
