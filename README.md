# Traffic Light Sign Recognition

A deep learning based traffic light sign recognition system using
EfficientNetB0 transfer learning with TensorFlow/Keras.

## Overview

This project classifies traffic light/sign images into their respective
categories using a pretrained EfficientNetB0 model.

## Model Architecture

Input Image
↓
Image Preprocessing
↓
Data Augmentation
↓
EfficientNetB0
↓
Global Average Pooling
↓
Dense Layer
↓
Dropout
↓
Output Layer
↓
Predicted Class

## Approach

Transfer Learning

A pretrained EfficientNetB0 network is used as the feature extractor,
followed by custom classification layers for traffic light/sign
classification.

## Dataset

cropped-LISA-traffic-light-dataset

## Technologies

- Python
- TensorFlow
- Keras
- EfficientNetB0
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Features

- Traffic light/sign image classification
- Image preprocessing
- Data augmentation
- Transfer learning
- Model evaluation
- Confusion matrix
- Prediction on new images

## Training

The complete implementation is available in:

`traffic_light_sign_recognition.ipynb`

## Results

Add your final test accuracy here.

Example:

Test Accuracy: XX%

## Trained Model

The trained model can be accessed here:

[View Trained Model](YOUR_HUGGING_FACE_LINK)

## Project Structure

traffic-light-sign-recognition/
│
├── README.md
├── requirements.txt
├── .gitignore
└── traffic_light_sign_recognition.ipynb
