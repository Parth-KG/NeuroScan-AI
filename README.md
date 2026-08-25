# 🧠 NeuroScan AI — Brain Tumor Detection

> An MRI brain-scan classifier built on EfficientNet-B0, evaluated on a held-out set of 1,600 scans.
> **95.19% accuracy**, **macro F1 0.9509**, and — on the screening question of *is there a tumor at all* — **100% specificity with zero false positives**.
> Includes Grad-CAM explainability and an interactive Streamlit dashboard.

---

## 📌 Project Overview

NeuroScan AI classifies brain MRI scans into four categories:

- **Glioma** — a tumor originating in the brain or spinal cord
- **Meningioma** — a tumor arising from the meninges
- **Pituitary** — a tumor at the base of the brain
- **No Tumor** — a healthy scan with no detected abnormality

The model uses transfer learning on a pretrained **EfficientNet-B0** backbone, fine-tuned with heavy data augmentation on a labeled MRI dataset. A **Grad-CAM** heatmap overlay provides visual explainability, highlighting the regions of the scan that drove the prediction.

---

## 📊 Results

Evaluated on the held-out `Testing/` directory: **1,600 scans, 400 per class.**

| Metric | Value |
|---|---|
| **Test accuracy** | **95.19%** (1,523 / 1,600) |
| Test loss | 0.4739 |
| Macro F1 | 0.9509 |
| ROC-AUC (macro) | 0.9950 |
| Cohen's κ | 0.9358 |
| Matthews correlation (MCC) | 0.9374 |
| Top-2 accuracy | 96.44% |

Cohen's κ and MCC both sit near 0.94, which matters because they discount agreement that chance alone would produce — the accuracy figure is not being inflated by class balance.

### Screening performance — tumor vs. no tumor

Collapsing the three tumor classes into one and asking only *is this scan abnormal*:

| Metric | Value |
|---|---|
| Sensitivity (recall on tumor) | **97.83%** |
| Specificity (recall on healthy) | **100.00%** |
| False positives | **0** |
| False negatives | 26 |

Every one of the 400 healthy scans was correctly identified as healthy. No patient in this test set would have been flagged for a tumor they do not have. The cost is 26 tumor scans out of 1,200 that were labelled *no tumor* — the failure direction that matters more clinically, and the one discussed under Limitations.

### Per-class performance

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | 0.997 | **0.830** | 0.906 |
| Meningioma | 0.899 | 0.980 | 0.938 |
| No Tumor | 0.939 | 1.000 | 0.969 |
| Pituitary | 0.985 | 0.998 | 0.991 |

### Where the errors actually are

With 400 scans per class, the per-class recall figures resolve the 77 total errors as follows *(derived arithmetically from the table above)*:

| Class | Scans | Misclassified |
|---|---|---|
| Glioma | 400 | **68** |
| Meningioma | 400 | 8 |
| Pituitary | 400 | ~1 |
| No Tumor | 400 | 0 |

**88% of every mistake the model makes is a glioma.** Meningioma's precision of 0.899 is the other side of the same coin: it absorbs roughly 44 false positives, most of them gliomas being reassigned. This is not a model that is uniformly 95% good — it is near-perfect on three classes and materially weaker on one.

### Baseline vs. final

| Model version | Test accuracy | Macro F1 |
|---|---|---|
| Baseline (`train.py`) | 93.06% | 0.9296 |
| **Augmented (`augmentedFinetune.py`) — final** | **95.19%** | **0.9509** |

Heavy augmentation with a clean validation split bought **+2.13 accuracy points and +0.021 macro F1** over the baseline.

---

## ⚠️ Limitations

Stated plainly, because the aggregate numbers hide them.

**Glioma recall is the weak point — 0.830.** Roughly one in six glioma scans receives a different label. Glioma *precision* is 0.997, so when the model does say "glioma" it is almost always right; the problem is entirely one of missed cases, not over-calling. Anyone reusing this model should treat a negative glioma prediction as weak evidence and a positive one as strong.

**All 26 false negatives are tumors called healthy.** The zero-false-positive result is genuinely good, but it should not be read in isolation — a model biased toward "no tumor" will produce exactly this shape of result. The 2.17% miss rate is the number to weigh against it.

**Test loss of 0.4739 is high for 95% accuracy.** That gap suggests the model is assigning high confidence to at least some of its wrong predictions. The confidence scores shown in the Streamlit app should be read as relative rankings, not calibrated probabilities. No calibration (temperature scaling, Platt) has been applied.

**Top-2 accuracy is only 96.44%.** Just 1.25 points above top-1, meaning the correct label usually is not sitting in second place when the model gets it wrong — the errors are confident, not near-misses.

**Single-source dataset, no external validation.** All training and test scans come from one Kaggle dataset. Performance on scans from different scanners, acquisition protocols, or patient populations is untested and should not be assumed.

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
│   ├── brain_tumor_efficientnet_finetuned.pth  # Intermediate experiment
│   └── brain_tumor_efficientnet_augmented.pth  # Final model (used in app)
│
├── train.py               # Initial training from pretrained EfficientNet-B0
├── finetune.py            # Intermediate fine-tuning experiment
├── augmentedFinetune.py   # Final training — heavy aug + clean val split
├── evaluate.py            # Full evaluation with confusion matrix & report
├── predict.py             # Single-image prediction with Grad-CAM
└── app.py                 # Streamlit web application
```

---

## ⚙️ Training Pipeline

### Stage 1 — `train.py` *(baseline)*
- Loads pretrained **EfficientNet-B0** from `timm` (ImageNet weights)
- Replaces the classification head with a 4-class output layer
- Trains for **5 epochs**, AdamW, LR 0.001, 80/20 train/val split
- Saves to `models/brain_tumor_efficientnet.pth`
- **Result: 93.06% test accuracy, macro F1 0.9296**

### Stage 2 — `finetune.py` *(intermediate, not used in final)*
- Continues from Stage 1 weights with light augmentation (`RandomHorizontalFlip`, `RandomRotation`)
- 10 epochs at LR 0.0001, `weight_decay=1e-4`
- Kept in the repo for reference; superseded by Stage 3 and **not re-evaluated** under the current test protocol, so no comparable figure is reported

### Stage 3 — `augmentedFinetune.py` *(final model)*
- Loads Stage 1 weights and applies **heavy augmentation** to training data:
  `RandomHorizontalFlip`, `RandomRotation(15)`, `RandomAffine`, `ColorJitter`
- Validation kept **clean** (no augmentation) for honest evaluation
- Deterministic index-based train/val split via `numpy`
- Saves to `models/brain_tumor_efficientnet_augmented.pth`
- **Result: 95.19% test accuracy, macro F1 0.9509**

---

## 🏋️ Model & Runtime

| Property | Value |
|---|---|
| Architecture | EfficientNet-B0 (`timm`, ImageNet-pretrained) |
| Parameters | 4,012,672 |
| Checkpoint size | 16.35 MB |
| Inference latency | 14.96 ms / image |
| Devices | Apple MPS / CUDA / CPU |

Small enough to run comfortably on CPU — roughly 67 scans per second at the measured latency.

---

## 🔍 Grad-CAM Explainability

Both `predict.py` and `app.py` use **Gradient-weighted Class Activation Mapping** to explain predictions visually.

- Target layer: `model.conv_head` (final convolutional layer of EfficientNet-B0)
- Highlights the regions of the MRI that most influenced the classification
- Rendered as a color overlay on the original scan

Given the glioma recall gap, the heatmaps are worth inspecting on any glioma-suspected scan rather than trusting the label alone.

---

## 🖥️ Streamlit App — `app.py`

An interactive dashboard for non-technical users.

**Features:**
- Upload any MRI scan (JPG/PNG)
- Instant classification with confidence score
- Per-class confidence breakdown with visual progress bars
- Grad-CAM heatmap overlay with explainability section
- Dark-themed, responsive UI

```bash
streamlit run app.py
```

> Confidence scores are uncalibrated — see Limitations.

---

## 📊 Evaluation — `evaluate.py`

Runs the final augmented model against the held-out `Testing/` directory and reports overall accuracy, per-class `classification_report` (precision / recall / F1), and a confusion matrix heatmap.

---

## 🛠️ Installation

```bash
pip install torch torchvision timm streamlit streamlit-lottie \
            pytorch-grad-cam scikit-learn matplotlib pillow requests
```

**Dataset:**
Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and place it in `./data/` with `Training/` and `Testing/` subdirectories.

**Train from scratch:**
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

Selects a random image from the test set, runs inference, and displays the original scan alongside the Grad-CAM heatmap — color-coded green (correct) or red (incorrect).

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Model | EfficientNet-B0 (`timm`) |
| Framework | PyTorch |
| Explainability | Grad-CAM (`pytorch-grad-cam`) |
| UI | Streamlit |
| Data | `torchvision.datasets.ImageFolder` |
| Hardware | Apple MPS / CUDA / CPU |

---

> ⚠️ **Disclaimer:** NeuroScan AI is a research and educational project. It is not intended for clinical use and does not replace professional medical diagnosis. The limitations above are not boilerplate — a 17% glioma miss rate is disqualifying for any diagnostic application.
