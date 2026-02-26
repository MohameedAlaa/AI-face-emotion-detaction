# 🤖 AI Face Emotion Detection - Model Retraining Guide

## Overview

This system automatically collects user feedback and allows you to retrain the model to improve accuracy over time.

---

## 📁 File Structure

```
project/
├── app.py                          # Main Streamlit app
├── best_emotion_model.keras        # Current model (updated)
├── best_emotion_model_backup.keras # Backup of previous model
├── retrain_model.py                # Retraining script
├── feedback_log.csv                # Feedback metadata
└── feedback_images/                # Folder with face images
    ├── 20260227_103045_Happy_to_Sad.png
    ├── 20260227_103156_Angry_to_Neutral.png
    └── ...
```

---

## 🔄 Workflow

### Step 1: Collect Feedback (Via App)

1. Run the Streamlit app:
   ```powershell
   cd "c:\Users\moham\OneDrive\Desktop\AI-face-emotion-detaction"
   python -m streamlit run project/app.py
   ```

2. Upload images and provide feedback:
   - Click **"✅ Yes, Correct!"** → Saves as correct prediction
   - Click **"❌ No, Incorrect"** → Select correct emotion and submit

3. Each feedback saves:
   - ✅ Metadata to `feedback_log.csv`
   - 📷 Face image to `feedback_images/`

### Step 2: Retrain Model (Offline Once)

Once you have enough feedback data (5-10+ samples recommended):

1. Open PowerShell in the project folder:
   ```powershell
   cd "c:\Users\moham\OneDrive\Desktop\AI-face-emotion-detaction\project"
   ```

2. Run the retraining script:
   ```powershell
   python retrain_model.py
   ```

3. The script will:
   - Load all feedback images and labels
   - Fine-tune the existing model
   - Ask if you want to replace the original model
   - Save backup of old model

---

## 📊 Feedback Data Format

### feedback_log.csv
| timestamp | predicted_emotion | confidence | corrected_emotion | image_path |
|-----------|-------------------|-----------|------------------|-----------|
| 2026-02-27 10:30:45 | Happy | 0.7500 | Sad | 20260227_103045_000123_Happy_to_Sad.png |
| 2026-02-27 10:35:12 | Angry | 0.6200 | Neutral | 20260227_103156_000456_Angry_to_Neutral.png |

---

## 🚀 How Retraining Works

1. **Data Preparation**: Loads 48×48 grayscale images from `feedback_images/`
2. **Label Mapping**: Uses `corrected_emotion` from CSV as ground truth
3. **Data Augmentation**: Applies rotation, zoom, shifts to improve generalization
4. **Fine-tuning**: Trains model with low learning rate (0.0001) for 10 epochs
5. **Model Update**: Replaces old model with improved version (with backup)

---

## 📈 Expected Improvements

- **With 5-10 samples**: Noticeable improvement
- **With 20+ samples**: Significant accuracy boost
- **With 50+ samples**: Major improvement in model performance

---

## 💡 Tips for Best Results

✅ **DO:**
- Test the model after retraining with new images
- Keep collecting feedback continuously
- Retrain regularly (weekly/monthly)
- Focus on common misclassifications

❌ **DON'T:**
- Delete images from `feedback_images/` unless necessary
- Modify `feedback_log.csv` manually
- Retrain with only 1-2 samples (too little data)

---

## 🔧 Troubleshooting

### No feedback data found
```
❌ Error: No feedback data found
```
→ Run the app first and provide feedback before retraining

### Image not found error
```
❌ Image not found: path/to/image.png
```
→ Don't move/delete feedback image files. Keep them in `feedback_images/` folder

### Model takes too long to train
→ Normal for first retrain. Subsequent runs with more data may take longer.

---

## 📝 Example Retraining Session

```powershell
============================================================
🤖 AI Emotion Detection - Model Retraining
============================================================
✅ Found 15 feedback records
  ✅ Loaded: 20260227_103045_Happy_to_Sad.png → Sad
  ✅ Loaded: 20260227_103156_Angry_to_Neutral.png → Neutral
  ...

📊 Data Summary:
   Total samples: 15
   Shape: (15, 48, 48, 1)
   Labels shape: (15,)

🔄 Loading original model...
📈 Model Summary:
   ...

🚀 Starting fine-tuning...
   Epochs: 10
   Batch size: 8
   Samples: 15

Epoch 1/10
2/2 [==============================] - 0s 120ms/step - loss: 0.8234 - accuracy: 0.6667
...
Epoch 10/10
2/2 [==============================] - 0s 115ms/step - loss: 0.2145 - accuracy: 0.9333

✅ Retrained model saved

📊 Training Results:
   Final Loss: 0.2145
   Final Accuracy: 0.9333

Replace original model? (y/n)
> y

✨ Model updated!
============================================================
```

---

## 🔐 Model Versions

- **best_emotion_model.keras** → Current active model (used by app)
- **best_emotion_model_backup.keras** → Previous version (manual rollback if needed)
- **best_emotion_model_retrained.keras** → Latest retrained version

---

## 🎯 Next Steps

1. ✅ App running and collecting feedback
2. ✅ Retraining script created
3. 🔄 Run `retrain_model.py` periodically
4. 📈 Monitor accuracy improvements
5. 🚀 Deploy improved model

---

**Happy Training! 🎉**
