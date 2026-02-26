import os
import sys
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import cv2  # type: ignore[import]
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore[import]
from PIL import Image
import pandas as pd
from datetime import datetime
from datetime import datetime

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
/* Main Title */
.big-title {
    font-size:48px !important;
    font-weight:800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    font-size:18px;
    color: #9CA3AF;
    text-align: center;
    margin-bottom: 30px;
}

/* Emotion Result Box */
.emotion-box {
    padding:30px;
    border-radius:20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    text-align:center;
    font-size:24px;
    color:#FFFFFF;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
    margin-bottom: 20px;
}

.emotion-box h2 {
    font-size: 36px;
    margin: 10px 0;
    font-weight: 700;
}

.emotion-box p {
    font-size: 18px;
    margin: 5px 0;
    opacity: 0.95;
}

/* Chart Title */
.chart-title {
    font-size: 20px;
    font-weight: 600;
    color: #FFFFFF;
    text-align: center;
    margin: 20px 0 10px 0;
}

/* Upload Section */
div[data-testid="stFileUploader"] {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
    padding: 20px;
    border-radius: 15px;
    border: 2px dashed #667eea;
}

/* Image Caption */
div[data-testid="caption"] {
    text-align: center;
    font-size: 16px;
    font-weight: 600;
    color: #667eea;
    margin-top: 10px;
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

# Set confidence threshold
confidence_threshold = 0.5

# ==============================
# Feedback Save Function
# ==============================
def save_feedback(predicted_emotion, confidence, correct_emotion):
    """Save feedback to CSV for model improvement"""
    feedback_file = os.path.join(os.path.dirname(__file__), "feedback_log.csv")
    
    feedback_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_emotion': predicted_emotion,
        'confidence': f"{confidence:.4f}",
        'corrected_emotion': correct_emotion
    }
    
    if os.path.exists(feedback_file):
        df = pd.read_csv(feedback_file)
        df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
    else:
        df = pd.DataFrame([feedback_data])
    
    df.to_csv(feedback_file, index=False)
    return True

# ==============================
# Main UI
# ==============================
st.markdown('<p class="big-title">🎭 AI Face Emotion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI detect emotions in real-time with advanced deep learning</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

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
            st.image(img, caption="✨ Detected Image", use_container_width=True)

        with col2:
            if emotions_detected:
                top_emotion, top_conf = max(emotions_detected, key=lambda x: x[1])
                st.markdown(f"""
                <div class="emotion-box">
                <h2>{emoji_dict[top_emotion]} {top_emotion}</h2>
                <p>Confidence: {top_conf:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Feedback Section
                st.markdown("---")
                st.markdown("<h3 style='text-align: center; color: #667eea;'>Was the result correct?</h3>", unsafe_allow_html=True)
                
                col_yes, col_no = st.columns(2)
                
                with col_yes:
                    if st.button("✅ Yes, Correct!", use_container_width=True, key="correct_btn"):
                        save_feedback(top_emotion, top_conf, top_emotion)
                        st.success("✨ Thank you! Your feedback helps us improve the model.")
                
                with col_no:
                    if st.button("❌ No, Incorrect", use_container_width=True, key="incorrect_btn"):
                        st.session_state.show_feedback = True
                
                # Show correction options if user said it was incorrect
                if st.session_state.get('show_feedback', False):
                    st.markdown("<h4 style='color: #764ba2;'>What is the correct emotion?</h4>", unsafe_allow_html=True)
                    
                    correct_emotion = st.selectbox(
                        "Select the correct emotion:",
                        emotion_labels,
                        key="emotion_correction"
                    )
                    
                    if st.button("💾 Submit Correction", use_container_width=True, key="submit_feedback"):
                        # Save feedback to CSV for model training
                        save_feedback(top_emotion, top_conf, correct_emotion)
                        st.success(f"✨ Thank you! Your correction has been saved.\n\n📊 Correction: {emoji_dict[top_emotion]} {top_emotion} → {emoji_dict[correct_emotion]} {correct_emotion}\n\nThis data will help improve our model!")
                        st.session_state.show_feedback = False
                
                # Show feedback status
                feedback_file = os.path.join(os.path.dirname(__file__), "feedback_log.csv")
                if os.path.exists(feedback_file):
                    feedback_df = pd.read_csv(feedback_file)
                    st.markdown(f"<p style='text-align: center; color: #667eea; font-size: 12px;'>📈 Model improved with {len(feedback_df)} corrections so far</p>", unsafe_allow_html=True)


# To run the app:
# 1) Open PowerShell and navigate to the project directory
# 2) Run: streamlit run project/app.py