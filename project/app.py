import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pandas as pd
import time

# ==============================
# Page Config
# ==============================
st.set_page_config(
    page_title="AI Emotion Detection",
    page_icon="🎭",
    layout="wide"
)

# ==============================
# Custom CSS
# ==============================
st.markdown("""
<style>
.big-title {
    font-size:40px !important;
    font-weight:700;
    color:#4A90E2;
}
.emotion-box {
    padding:15px;
    border-radius:10px;
    background-color:#F0F2F6;
    text-align:center;
    font-size:20px;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Load Model (Optimized)
# ==============================
@st.cache_resource
def load_emotion_model():
    return load_model("best_emotion_model.keras")

model = load_emotion_model()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
emoji_dict = {
    "Angry": "😠",
    "Disgust": "🤢",
    "Fear": "😨",
    "Happy": "😄",
    "Sad": "😢",
    "Surprise": "😲",
    "Neutral": "😐"
}

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ==============================
# Sidebar
# ==============================
st.sidebar.title("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)
st.sidebar.markdown("---")
st.sidebar.info("Built by Mohamed Alaa 🚀\nAI & Machine Learning Developer")

# ==============================
# Main UI
# ==============================
st.markdown('<p class="big-title">🎭 AI Face Emotion Detection</p>', unsafe_allow_html=True)
st.write("Upload an image and let AI detect the emotion in real time.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected!")
    else:
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face)
            confidence = np.max(prediction)
            emotion = emotion_labels[np.argmax(prediction)]

            if confidence >= confidence_threshold:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                cv2.putText(img, emotion, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0,255,0), 2)

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Detected Image", use_column_width=True)

        with col2:
            st.markdown(f"""
            <div class="emotion-box">
            <h2>{emoji_dict[emotion]} {emotion}</h2>
            <p>Confidence: {confidence:.2f}</p>
            </div>
            """, unsafe_allow_html=True)

            df = pd.DataFrame(prediction[0], index=emotion_labels, columns=["Confidence"])
            st.bar_chart(df)