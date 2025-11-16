AI Face Emotion Detection System

This project is a complete Facial Emotion Recognition System built using Deep Learning, OpenCV, and a custom CNN model.
It includes:

A full training pipeline

Dataset augmentation tool

A real-time GUI application for emotion detection

The system predicts emotions such as:
Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise

🚀 Features
✔ 1. Deep Learning Model

Custom CNN built with TensorFlow/Keras

Trained on an augmented dataset of 48×48 grayscale faces

Uses:

Conv2D layers

MaxPooling

Dropout

Batch Normalization

Softmax classifier

✔ 2. Data Augmentation Tool

Automatically generates images (flip, rotate, zoom, shift)

Saves augmented dataset to a folder

✔ 3. Real-Time Emotion Detection GUI

Built with Tkinter

Uses OpenCV to capture webcam video

Loads trained model

Displays the live emotion prediction on screen

📁 Project Structure
📁 Emotion-Detection-Project
 ├── 01_training.ipynb        # Model training pipeline
 ├── 02_augmentation.ipynb    # Data augmentation tool
 ├── 03_gui_app.ipynb         # Tkinter real-time detection app
 ├── model.h5                 # Saved trained model (optional)
 ├── augmented_dataset/        # Auto-generated images
 ├── README.md

🧠 Model Training Workflow

Load dataset from folders

Preprocess images

Encode emotion labels

Build and compile CNN model

Train using:

EarlyStopping

ReduceLROnPlateau

Evaluate performance

Generate:

Accuracy/Loss curves

Confusion matrix

Per-class accuracy results

📊 Evaluation Metrics

Your notebook computes:

Overall accuracy

Per-class accuracy

Confusion matrix

Training accuracy & loss curves

These visualizations help verify training quality.

🛠️ Technologies Used

Python

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

Tkinter (GUI)

scikit-learn

▶️ How to Run
1️⃣ Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow

2️⃣ Train the model

Run:

01_training.ipynb

3️⃣ Generate augmented images

Run:

02_augmentation.ipynb

4️⃣ Launch the GUI real-time detector

Run:

03_gui_app.ipynb

📝 Future Improvements

Replace CNN with MobileNetV2 for higher accuracy

Add face alignment for better detection

Export model to TensorFlow Lite

Build a full desktop or web app

📄 License

This project is open source — feel free to modify and use it for learning or development.
