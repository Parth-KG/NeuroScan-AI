# 🧠 NeuroScan AI — Brain Tumor Detection

> An AI-powered MRI brain scan classifier built with EfficientNet-B0, achieving **95.39% test accuracy** across four diagnostic categories. Includes Grad-CAM explainability and an interactive Streamlit dashboard.

---

## 📌 Project Overview

NeuroScan AI is a deep learning system that classifies brain MRI scans into four categories:

- **Glioma** — a type of tumor originating in the brain or spinal cord
- **Meningioma** — a tumor arising from the meninges
- **Pituitary** — a tumor located at the base of the brain
- **No Tumor** — a healthy scan with no detected abnormality

The model uses transfer learning on a pretrained **EfficientNet-B0** backbone, fine-tuned with heavy data augmentation on a labeled MRI dataset. A **Grad-CAM** heatmap overlay provides visual explainability, highlighting the exact regions of the MRI that influenced the model's prediction.

---

## 🗂️ Project Structure

```
NeuroScan-AI/
│
├── data/
│   ├── Training/          # Training images (organized by class)
│   └── Testing/           # Test images (organized by class)
│
├── models/
│   ├── brain_tumor_efficientnet.pth            # Base trained model
│   ├── brain_tumor_efficientnet_finetuned.pth  # Fine-tuned model
│   └── brain_tumor_efficientnet_augmented.pth  # Final augmented model (used in app)
│
├── train.py               # Initial training from pretrained EfficientNet-B0
├── finetune.py            # Fine-tuning with augmentation on trained model
├── augmentedFinetune.py   # Advanced fine-tuning with clean val split strategy
├── evaluate.py            # Full evaluation with confusion matrix & report
├── predict.py             # Single-image prediction with Grad-CAM visualization
└── app.py                 # Streamlit web application
```

---

## ⚙️ Model Training Pipeline

The model was developed in three progressive stages:

### Stage 1 — `train.py`
- Loads pretrained **EfficientNet-B0** from `timm` (ImageNet weights)
- Replaces the classification head with a 4-class output layer
- Trains for **5 epochs** with AdamW optimizer (LR: 0.001)
- 80/20 train/validation split
- Saves weights to `models/brain_tumor_efficientnet.pth`

### Stage 2 — `finetune.py`
- Loads Stage 1 weights and continues training
- Adds `RandomHorizontalFlip` and `RandomRotation` augmentation
- Runs for **10 more epochs** with a lower LR (0.0001)
- Adds `weight_decay=1e-4` to reduce overfitting
- Saves to `models/brain_tumor_efficientnet_finetuned.pth`

### Stage 3 — `augmentedFinetune.py` *(Final Model)*
- Loads Stage 1 weights and applies **heavy augmentation** to training data:
  - `RandomHorizontalFlip`, `RandomRotation(15)`, `RandomAffine`, `ColorJitter`
- Validation is kept **clean** (no augmentation) for honest evaluation
- Uses a deterministic index-based train/val split via `numpy`
- Saves to `models/brain_tumor_efficientnet_augmented.pth`

---

## 📊 Evaluation — `evaluate.py`

Runs the final augmented model against the held-out **Testing** directory.

**Outputs:**
- Overall test accuracy
- Per-class `classification_report` (precision, recall, F1)
- Confusion matrix heatmap

**Result: ✅ 95.39% Test Accuracy**

---

## 🔍 Grad-CAM Explainability

Both `predict.py` and `app.py` use **Gradient-weighted Class Activation Mapping (Grad-CAM)** to generate heatmaps that visually explain model predictions.

- Target layer: `model.conv_head` (final convolutional layer of EfficientNet)
- Highlights regions in the MRI that most strongly influenced the classification
- Rendered as a color overlay on the original scan

---

## 🖥️ Streamlit App — `app.py`

An interactive web dashboard for non-technical users.

**Features:**
- Upload any MRI scan (JPG/PNG)
- Instant classification with confidence score
- Per-class confidence breakdown with visual progress bars
- Grad-CAM heatmap overlay with explainability section
- Dark-themed, responsive UI

**Run the app:**
```bash
streamlit run app.py
```

---

## 🛠️ Installation

**Requirements:**
```bash
pip install torch torchvision timm streamlit streamlit-lottie \
            pytorch-grad-cam scikit-learn matplotlib pillow requests
```

**Dataset:**  
Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and place it in `./data/` with `Training/` and `Testing/` subdirectories.

**Training from scratch:**
```bash
python train.py
python augmentedFinetune.py
python evaluate.py
```

---

## 🧪 Quick Prediction Test

```bash
python predict.py
```

Randomly selects an image from the test set, runs inference, and displays the original scan alongside the Grad-CAM heatmap — color-coded green (correct) or red (incorrect).

---

## 📈 Results Summary

| Model Version       | Val Accuracy |
|---------------------|-------------|
| Base (train.py)     | ~92–93%     |
| Fine-tuned          | ~94%        |
| Augmented (final)   | **95.39%**  |

---

## 🏗️ Tech Stack

| Component        | Technology                        |
|------------------|------------------------------------|
| Model            | EfficientNet-B0 (`timm`)           |
| Framework        | PyTorch                            |
| Explainability   | Grad-CAM (`pytorch-grad-cam`)      |
| UI               | Streamlit                          |
| Data             | `torchvision.datasets.ImageFolder` |
| Hardware         | Apple MPS / CUDA / CPU             |

---

> ⚠️ **Disclaimer:** NeuroScan AI is a research and educational project. It is not intended for clinical use or to replace professional medical diagnosis.
