# AI Facial Emotion Detection System










 ## 📌 Overview

This project is a complete AI-powered Facial Emotion Detection system built using Deep Learning, Computer Vision, and a custom CNN model.
The system can recognize emotions in real-time from webcam input and classify them into:

(Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise)

It includes:

 >Dataset loading & preprocessing

 >CNN model training

 >Data augmentation module

 >Real-time detection GUI

## 📖 Table of Contents

>Features

>Screenshots

>Model Architecture

>Tech Stack

>Project Structure

>Installation

### Usage

Results

Future Improvements

License

## ✨ Features

 >Real-time emotion detection

 >Custom CNN trained on augmented dataset

 >Confusion matrix & accuracy visualization

 >GUI interface using Tkinter

 >Data augmentation pipeline

 >Per-class accuracy calculations

## 📸 Screenshots

Before augmentation:
<img width="1189" height="690" alt="image" src="https://github.com/user-attachments/assets/b4b0855f-fc5c-4b30-99af-b051c33f7a79" />

After augmentation:
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/f7dfebf1-f0bc-428e-9ac1-98a070c05a88" />


## 🧠 Model Architecture

>Conv2D + ReLU

>MaxPooling

>Batch Normalization

>Dropout

>Dense + Softmax

>Optimizers & callbacks:

>Adam

>EarlyStopping

>ReduceLROnPlateau

## 🛠 Tech Stack

>Python

>TensorFlow / Keras

>OpenCV

>NumPy

>scikit-learn

>Matplotlib

>Tkinter

## ▶️ Usage
>Generate augmented data: 
 [ 01_Data Augmentation.ipynb ]

>Train the model: 
 [ 02_AI face emotion detaction training.ipynb ]

>Run real-time GUI: 
 [ 03_GUI.ipynb ]


## 📊 Results

 >Model Accuracy: 83.02%
<img width="852" height="690" alt="image" src="https://github.com/user-attachments/assets/9e59c019-58c8-430a-8f39-84aec7a21235" />

<img width="865" height="690" alt="image" src="https://github.com/user-attachments/assets/884bcf61-2832-423a-8dfa-31512e59997d" />

<img width="1189" height="390" alt="image" src="https://github.com/user-attachments/assets/1a484599-223f-4df5-ae9f-9979908ab61e" />

> Accuracy 76.08% for (angry)

> Accuracy 96.82% for (disgust)

> Accuracy 72.00% for (fear)

> Accuracy 92.67% for (happy)

> Accuracy 80.90% for (neutral)

> Accuracy 71.72% for (sad)

> Accuracy 91.41% for (surprise)


## 🚀 Future Improvements

>Add real-time webcam emotion detection

>Use MobileNetV2 or EfficientNet

>Deploy as desktop/web application

>Convert to TensorFlow Lite

>Multi-face emotion detection


## 📄 License

>This project is licensed under the MIT License.
