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
import hashlib
import time
import urllib.request

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
     model_path = os.path.join(base_dir, "best_emotion_model_retrained.keras")
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

# Face detector files (OpenCV DNN)
base_dir = os.path.dirname(__file__)
dnn_prototxt_path = os.path.join(base_dir, "opencv_face_detector.prototxt")
dnn_model_path = os.path.join(base_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# DNN model download sources
dnn_prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
dnn_model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

# Load face detector (DNN preferred, Haar fallback)
def get_face_detector():
    """Get face detector, preferring DNN if files exist, falling back to Haar."""
    if os.path.exists(dnn_prototxt_path) and os.path.exists(dnn_model_path):
        try:
            net = cv2.dnn.readNetFromCaffe(dnn_prototxt_path, dnn_model_path)
            return "dnn", net
        except Exception:
            pass  # Fall through to Haar if DNN fails

    cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
    haar_cascade = cv2.CascadeClassifier(cascade_path)
    return "haar", haar_cascade

# Load initial detector (will be re-checked after download)
face_detector_type, face_detector = get_face_detector()

# DNN confidence threshold
dnn_confidence_threshold = 0.5

# Emotion confidence threshold (lowered to show more results)
emotion_confidence_threshold = 0.3
def ensure_dnn_files():
    """Download DNN face detector files if missing."""
    if os.path.exists(dnn_prototxt_path) and os.path.exists(dnn_model_path):
        return True

    if st.session_state.get("dnn_download_attempted", False):
        return False

    st.session_state.dnn_download_attempted = True
    try:
        with st.spinner("Downloading face detector files..."):
            if not os.path.exists(dnn_prototxt_path):
                urllib.request.urlretrieve(dnn_prototxt_url, dnn_prototxt_path)
            if not os.path.exists(dnn_model_path):
                urllib.request.urlretrieve(dnn_model_url, dnn_model_path)
        return True
    except Exception:
        return False

# ==============================
# Feedback Save Function
# ==============================
def save_feedback(predicted_emotion, confidence, correct_emotion, face_image):
    """Save feedback and the face image for model improvement"""
    # Calculate hash of the image to check for duplicates
    image_hash = hashlib.md5(face_image.tobytes()).hexdigest()
    
    # Create feedback directory
    feedback_dir = os.path.join(os.path.dirname(__file__), "feedback_images")
    if not os.path.exists(feedback_dir):
        os.makedirs(feedback_dir)
    
    # Check if this image hash already exists in feedback log
    feedback_file = os.path.join(os.path.dirname(__file__), "feedback_log.csv")
    if os.path.exists(feedback_file):
        try:
            df = pd.read_csv(feedback_file)
            if 'image_hash' in df.columns and image_hash in df['image_hash'].values:
                return False  # Duplicate image, skip saving
        except (PermissionError, pd.errors.ParserError):
            pass  # Continue with saving even if we can't read the file
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_filename = f"{timestamp}_{predicted_emotion}_to_{correct_emotion}.png"
    image_path = os.path.join(feedback_dir, image_filename)
    
    # Save face image
    cv2.imwrite(image_path, face_image)
    
    # Save feedback to CSV with image hash (with retry logic)
    feedback_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'predicted_emotion': predicted_emotion,
        'confidence': f"{confidence:.4f}",
        'corrected_emotion': correct_emotion,
        'image_path': image_filename,
        'image_hash': image_hash
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if os.path.exists(feedback_file):
                df = pd.read_csv(feedback_file)
                df = pd.concat([df, pd.DataFrame([feedback_data])], ignore_index=True)
            else:
                df = pd.DataFrame([feedback_data])
            
            df.to_csv(feedback_file, index=False)
            return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.5)  # Wait 500ms before retry
            else:
                st.warning("⚠️ Feedback image saved, but couldn't update log (OneDrive sync issue). Retrying...")
                return True  # Return True since image was saved successfully
    
    return True

# ==============================
# Main UI
# ==============================
st.markdown('<p class="big-title">🎭 AI Face Emotion Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload an image and let AI detect emotions in real-time with advanced deep learning</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

# Reset feedback state when a new file is uploaded
if 'last_uploaded_file' not in st.session_state:
    st.session_state.last_uploaded_file = None
if uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.last_uploaded_file = uploaded_file
    st.session_state.feedback_submitted = False

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = np.array(image.convert("RGB"))  # Ensure 3 channels

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if face_detector_type != "dnn":
        if ensure_dnn_files():
            face_detector_type, face_detector = get_face_detector()

    if face_detector_type == "dnn":
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        (h, w) = img_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(img_bgr, 1.0, (300, 300), (104.0, 177.0, 123.0))
        face_detector.setInput(blob)
        detections = face_detector.forward()
        faces = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < dnn_confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2 - x1, y2 - y1))
    else:
        if not st.session_state.get("dnn_files_missing_warned", False):
            st.info(
                "For better face detection, add these files to the project folder: "
                "opencv_face_detector.prototxt and res10_300x300_ssd_iter_140000.caffemodel. "
                "Using Haar cascade fallback for now."
            )
            st.session_state.dnn_files_missing_warned = True
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected!")
    else:
        emotions_detected = []
        face_images = {}  # Store face images for feedback
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face_for_model = np.reshape(face, (1, 48, 48, 1))

            prediction = model.predict(face_for_model, verbose=0)
            confidence = np.max(prediction)
            emotion = emotion_labels[np.argmax(prediction)]

            if confidence >= emotion_confidence_threshold:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(img, f"{emoji_dict[emotion]} {emotion}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                emotions_detected.append((emotion, confidence))
                # Store the original 48x48 grayscale face for saving
                face_images[emotion] = (face * 255).astype(np.uint8)

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
                
                # Feedback Section - only show if feedback hasn't been submitted yet
                if not st.session_state.get('feedback_submitted', False):
                    st.markdown("---")
                    st.markdown("<h3 style='text-align: center; color: #667eea;'>Was the result correct?</h3>", unsafe_allow_html=True)
                    
                    col_yes, col_no = st.columns(2)
                    
                    with col_yes:
                        if st.button("✅ Yes, Correct!", use_container_width=True, key="correct_btn"):
                            save_feedback(top_emotion, top_conf, top_emotion, face_images.get(top_emotion))
                            st.session_state.feedback_submitted = True
                            st.session_state.is_correction = False
                            st.rerun()
                    
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
                            # Save feedback and image for model training
                            save_feedback(top_emotion, top_conf, correct_emotion, face_images.get(top_emotion))
                            st.session_state.correction_details = f"{emoji_dict[top_emotion]} {top_emotion} → {emoji_dict[correct_emotion]} {correct_emotion}"
                            st.session_state.show_feedback = False
                            st.session_state.feedback_submitted = True
                            st.session_state.is_correction = True
                            st.rerun()
                else:
                    # Show a thank you message based on feedback type
                    if st.session_state.get('is_correction', False):
                        st.markdown("---")
                        if st.session_state.get('correction_details'):
                            st.markdown(f"""
                            <div style='padding: 25px; border-radius: 15px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); text-align: center; color: white; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);'>
                                <h3 style='margin: 0 0 10px 0; font-size: 24px;'>✨ Thank You for Your Feedback!</h3>
                                <p style='margin: 5px 0; font-size: 16px; opacity: 0.95;'>Your correction has been saved and will help improve our AI model.</p>
                                <p style='margin: 10px 0 0 0; font-size: 18px; font-weight: 600;'>📊 {st.session_state.correction_details}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("---")
                        st.markdown("""
                        <div style='padding: 25px; border-radius: 15px; background: linear-gradient(135deg, #10b981 0%, #059669 100%); text-align: center; color: white; box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3);'>
                            <h3 style='margin: 0 0 10px 0; font-size: 24px;'>Thank you for using our app!</h3>
                            <p style='margin: 5px 0; font-size: 16px; opacity: 0.95;'>We appreciate your support and look forward to providing you with even better AI-powered insights.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Show feedback status
                feedback_file = os.path.join(os.path.dirname(__file__), "feedback_log.csv")
                if os.path.exists(feedback_file):
                    feedback_df = pd.read_csv(feedback_file)
                    st.markdown(f"<p style='text-align: center; color: #667eea; font-size: 12px;'>📈 Model improved with {len(feedback_df)} corrections so far</p>", unsafe_allow_html=True)


# To run the app:
# 1) Open PowerShell and navigate to the project directory
# 2) Run: streamlit run project/app.py