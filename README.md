# 🎭 DeepFake Detection using EfficientNetB0

A deep learning system that detects AI-generated fake images using a fine-tuned **EfficientNetB0** convolutional neural network — built and trained entirely on Google Colab.

---

## 📌 Problem Statement

With the rapid rise of AI-generated media, deepfakes pose a serious threat to digital trust and authenticity. This project builds a binary image classifier that can distinguish between **real** and **AI-generated fake** images using transfer learning.

---

## 🧠 Model Architecture

**Base Model:** EfficientNetB0 (pre-trained on ImageNet, weights frozen)

**Custom Classification Head:**

```
EfficientNetB0 (frozen base)
        ↓
GlobalAveragePooling2D
        ↓
Dropout (0.5)
        ↓
Dense (64 units, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense (1 unit, Sigmoid) → Real / Fake
```

**Why EfficientNetB0?**
- Lightweight yet highly accurate for image classification
- Pre-trained on 1M+ ImageNet images — strong feature extraction out of the box
- Ideal for limited computational resources like Google Colab free tier

---

## 📊 Dataset

| Detail | Value |
|--------|-------|
| Source | Kaggle — `manjilkarki/deepfake-and-real-images` |
| Classes | Real, Fake |
| Training images | 800 (400 per class) |
| Test images | 200 (100 per class) |
| Total used | 1000 images (subset due to Colab constraints) |
| Image size | 224 × 224 px |

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 10 |
| Batch Size | 32 |
| Early Stopping | Patience = 3 (restores best weights) |

---

## 📈 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | 52.00% |
| Test Images | 200 |
| Classes | Binary (Real / Fake) |

> **Note:** The model was trained on a 1000-image subset due to Google Colab computational constraints. Accuracy is expected to improve significantly with a larger dataset and more epochs. This serves as a working proof-of-concept pipeline.

---

## 🗂️ Project Structure

```
DeepFakeDetection/
├── deepfake_detection.ipynb       # Main Google Colab notebook
├── kaggle.json                    # Kaggle API key (not committed)
├── deepfake_dataset/
│   ├── Real/                      # Real images
│   └── Fake/                      # Deepfake images
└── efficientnet_model.keras       # Saved trained model
```

---

## 🚀 How to Run

### Step 1 — Open in Google Colab
Upload `deepfake_detection.ipynb` to [colab.research.google.com](https://colab.research.google.com)

### Step 2 — Upload Kaggle API key
When prompted, upload your `kaggle.json` file to authenticate with Kaggle

### Step 3 — Run cells sequentially
The notebook handles:
- Dataset download from Kaggle
- Image preprocessing and augmentation
- Model building with EfficientNetB0
- Training with early stopping
- Evaluation and accuracy plots
- Saving model to Google Drive

### Step 4 — Model is saved automatically
Trained model saved to:
```
/content/drive/My Drive/DeepfakeDetection/efficientnet_model.keras
```

---

## 🔬 How It Works

```
Input Image (224×224)
        ↓
EfficientNetB0 extracts visual features
        ↓
Custom head classifies features
        ↓
Sigmoid output → probability score
        ↓
Score > 0.5 → FAKE
Score ≤ 0.5 → REAL
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| TensorFlow / Keras | Deep learning framework |
| EfficientNetB0 | Pre-trained CNN base model |
| Google Colab | Training environment (free GPU) |
| Kaggle API | Dataset download |
| Google Drive | Model storage |
| Matplotlib | Training plots |

---

## 🔮 Future Improvements

- [ ] Train on full dataset (10,000+ images) for better accuracy
- [ ] Add video deepfake detection frame-by-frame
- [ ] Deploy as a web app — upload an image → get Real/Fake verdict
- [ ] Try EfficientNetB3 or B7 for higher accuracy
- [ ] Add Grad-CAM visualisation to highlight which regions look fake

---

## 👥 Contributors

| Name | GitHub |
|------|--------|
| **Mekala Madhu Mitha** | [@MadhuMitha16884](https://github.com/MadhuMitha16884) |
| **Ishrath Tabassum** | [@Azalea2406](https://github.com/Azalea2406) |
| **Srujana Patchala** | [@srujana247](https://github.com/srujana247) |

---

## 📄 License

This project is for academic and research purposes only.
