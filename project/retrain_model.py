import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

# ==============================
# Configuration
# ==============================
BASE_DIR = os.path.dirname(__file__)
FEEDBACK_CSV = os.path.join(BASE_DIR, "feedback_log.csv")
FEEDBACK_IMAGES_DIR = os.path.join(BASE_DIR, "feedback_images")
MODEL_PATH = os.path.join(BASE_DIR, "best_emotion_model.keras")
RETRAINED_MODEL_PATH = os.path.join(BASE_DIR, "best_emotion_model_retrained.keras")

EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ==============================
# Load Feedback Data
# ==============================
def load_feedback_data():
    """Load feedback data from CSV and images"""
    if not os.path.exists(FEEDBACK_CSV):
        print("❌ No feedback data found. Run the app first to collect feedback.")
        return None, None
    
    df = pd.read_csv(FEEDBACK_CSV)
    
    if len(df) == 0:
        print("❌ Feedback CSV is empty. No data to train on.")
        return None, None
    
    print(f"✅ Found {len(df)} feedback records")
    
    X = []  # Images
    y = []  # Labels
    
    for idx, row in df.iterrows():
        # Handle missing values
        if pd.isna(row.get('image_path')) or pd.isna(row.get('corrected_emotion')):
            print(f"  ⚠️  Skipping row {idx}: Missing image_path or corrected_emotion")
            continue
        
        image_filename = str(row['image_path']).strip()
        correct_emotion = str(row['corrected_emotion']).strip()
        
        image_path = os.path.join(FEEDBACK_IMAGES_DIR, image_filename)
        
        if os.path.exists(image_path):
            # Load image
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img / 255.0  # Normalize
                X.append(img)
                y.append(EMOTION_LABELS.index(correct_emotion))
                print(f"  ✅ Loaded: {image_filename} → {correct_emotion}")
            else:
                print(f"  ❌ Failed to load: {image_filename}")
        else:
            print(f"  ❌ Image not found: {image_path}")
    
    if len(X) == 0:
        print("❌ No valid images loaded.")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape for model input (batch, height, width, channels)
    X = X.reshape(-1, 48, 48, 1)
    
    print(f"\n📊 Data Summary:")
    print(f"   Total samples: {len(X)}")
    print(f"   Shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    return X, y

# ==============================
# Retrain Model
# ==============================
def retrain_model(X_feedback, y_feedback):
    """Fine-tune the existing model with feedback data"""
    
    print("\n🔄 Loading original model...")
    model = load_model(MODEL_PATH)
    
    print("📈 Model Summary:")
    model.summary()
    
    # Compile model with lower learning rate for fine-tuning
    optimizer = Adam(learning_rate=0.0001)  # Lower learning rate for fine-tuning
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Data augmentation to improve generalization
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        fill_mode='nearest'
    )
    
    print("\n🚀 Starting fine-tuning...")
    print(f"   Epochs: 10")
    print(f"   Batch size: 8")
    print(f"   Samples: {len(X_feedback)}")
    
    # Fine-tune with the feedback data
    history = model.fit(
        datagen.flow(X_feedback, y_feedback, batch_size=8),
        epochs=10,
        verbose=1,
        steps_per_epoch=max(1, len(X_feedback) // 8)
    )
    
    # Save the retrained model
    model.save(RETRAINED_MODEL_PATH)
    print(f"\n✅ Retrained model saved to: {RETRAINED_MODEL_PATH}")
    
    # Print metrics
    print(f"\n📊 Training Results:")
    print(f"   Final Loss: {history.history['loss'][-1]:.4f}")
    print(f"   Final Accuracy: {history.history['accuracy'][-1]:.4f}")
    
    return model

# ==============================
# Backup and Replace Original Model
# ==============================
def backup_and_replace_model():
    """Backup old model and replace with retrained one"""
    import shutil
    
    backup_path = os.path.join(BASE_DIR, "best_emotion_model_backup.keras")
    
    if os.path.exists(RETRAINED_MODEL_PATH):
        # Backup original model
        if os.path.exists(MODEL_PATH):
            shutil.copy(MODEL_PATH, backup_path)
            print(f"💾 Original model backed up to: {backup_path}")
        
        # Replace with retrained model
        shutil.copy(RETRAINED_MODEL_PATH, MODEL_PATH)
        print(f"✨ Model updated successfully!")
        print(f"   New model: {MODEL_PATH}")

# ==============================
# Main Execution
# ==============================
def main():
    print("=" * 60)
    print("🤖 AI Emotion Detection - Model Retraining")
    print("=" * 60)
    
    # Load feedback data
    X_feedback, y_feedback = load_feedback_data()
    
    if X_feedback is None or y_feedback is None:
        print("\n❌ Aborting: No valid feedback data to train on.")
        return
    
    # Retrain model
    model = retrain_model(X_feedback, y_feedback)
    
    # Ask user if they want to replace the original model
    print("\n" + "=" * 60)
    print("Replace original model with retrained version? (y/n)")
    response = input("> ").strip().lower()
    
    if response == 'y':
        backup_and_replace_model()
        print("\n✅ Model updated! Your app will now use the improved model.")
    else:
        print("\n⚠️  Original model unchanged.")
        print(f"   Retrained model saved as: {RETRAINED_MODEL_PATH}")
    
    print("\n" + "=" * 60)
    print("✨ Retraining complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
