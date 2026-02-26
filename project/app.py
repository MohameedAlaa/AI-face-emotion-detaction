import os
import sys
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import cv2  # type: ignore[import]
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore[import]
from PIL import Image
import pandas as pd

# When executed directly with python, Streamlit lacks a ScriptRunContext which
# causes many warnings. Detect that case and print a helpful message instead.
ctx = get_script_run_ctx()
if ctx is None:
    sys.stderr.write("ERROR: This application must be started with `streamlit run project/app.py`\n")
    sys.stderr.write("For example:\n    streamlit run project/app.py\n")
    sys.exit(1)


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
# Load Model
# ==============================
@st.cache_resource
def load_emotion_model():
     base_dir = os.path.dirname(__file__)
     model_path = os.path.join(base_dir, "best_emotion_model.keras")
     return load_model(model_path)

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

# Load Haar Cascade
cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

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
    img = np.array(image.convert("RGB"))  # Ensure 3 channels

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected!")
    else:
        emotions_detected = []
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face, verbose=0)
            confidence = np.max(prediction)
            emotion = emotion_labels[np.argmax(prediction)]

            if confidence >= confidence_threshold:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{emoji_dict[emotion]} {emotion}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                emotions_detected.append((emotion, confidence))

        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Detected Image", use_column_width=True)

        with col2:
            if emotions_detected:
                top_emotion, top_conf = max(emotions_detected, key=lambda x: x[1])
                st.markdown(f"""
                <div class="emotion-box">
                <h2>{emoji_dict[top_emotion]} {top_emotion}</h2>
                <p>Confidence: {top_conf:.2f}</p>
                </div>
                """, unsafe_allow_html=True)

                df = pd.DataFrame(prediction[0], index=emotion_labels, columns=["Confidence"])
                st.bar_chart(df)