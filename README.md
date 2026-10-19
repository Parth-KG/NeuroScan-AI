# 🧠 NeuroScan AI — Brain Tumor Detection

> An MRI brain-scan classifier built on EfficientNet-B0, evaluated on the 1,600-scan held-out test set of the deduplicated Kaggle Brain Tumor MRI dataset.
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

Evaluated on the held-out `Testing/` split of the dataset's deduplicated release: **1,600 scans, 400 per class.**

| Metric | Value |
|---|---|
| **Test accuracy** | **95.19%** (1,523 / 1,600) |
| Test loss | 0.4739 |
| Macro F1 | 0.9509 |
| ROC-AUC (macro) | 0.9950 |
| Cohen's κ | 0.9358 |
| Matthews correlation (MCC) | 0.9374 |
| Top-2 accuracy | 96.44% |

Cohen's κ and MCC both sit near 0.94. Both discount the agreement chance alone would produce, and both still land close to the accuracy figure.

> **Why not 99.39%?** An earlier version of this README reported 99.39% on the dataset's original split (5,712 training / 1,311 test images). That split has near-duplicate images on both sides of the train/test boundary, which inflates test scores. The deduplicated release removes repeated images, so 95.19% is the better number, but it is still not leakage-free (see Limitations). The confusion matrix in `assets/` is from that original-split run.

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

The final stage adds **+2.13 accuracy points and +0.021 macro F1** over the baseline. That gain combines heavier augmentation with 10 more epochs at a lower learning rate, so it can't be credited to augmentation alone. Validation doesn't pick the checkpoint either: each script saves its last epoch.

---

## ⚠️ Limitations

Stated plainly, because the aggregate numbers hide them.

**Glioma recall is the weak point — 0.830.** Roughly one in six glioma scans receives a different label. Glioma *precision* is 0.997, so when the model does say "glioma" it is almost always right; the problem is entirely one of missed cases, not over-calling. Anyone reusing this model should treat a negative glioma prediction as weak evidence and a positive one as strong.

**All 26 false negatives are tumors called healthy.** The zero-false-positive result is genuinely good, but it should not be read in isolation — a model biased toward "no tumor" will produce exactly this shape of result. The 2.17% miss rate is the number to weigh against it.

**Test loss of 0.4739 is high for 95% accuracy.** That gap suggests the model is assigning high confidence to at least some of its wrong predictions. The confidence scores shown in the Streamlit app should be read as relative rankings, not calibrated probabilities. No calibration (temperature scaling, Platt) has been applied.

**Top-2 accuracy is only 96.44%.** Just 1.25 points above top-1, meaning the correct label usually is not sitting in second place when the model gets it wrong — the errors are confident, not near-misses.

**The split is by image, not by patient.** The dataset's `Training/` and `Testing/` folders don't keep a patient's slices together, so slices from one patient can land on both sides. My follow-up audit, [neuroscan-ood](https://github.com/Parth-KG/neuroscan-ood), measures this on the source Figshare data: a patient-grouped split scores about 5.8 points lower than a slice-level one. Read 95.19% as an upper bound.

**One dataset, no external validation.** All training and test scans come from one Kaggle dataset, itself a merge of the Figshare, SARTAJ and Br35H collections, and neuroscan-ood finds that most of SARTAJ's tumour images duplicate Figshare's. Performance on scans from different scanners, acquisition protocols, or patient populations is untested and should not be assumed.

---

## 🗂️ Project Structure

```
NeuroScan-AI/
├── src/
│   ├── train.py               # Stage 1: baseline from pretrained EfficientNet-B0
│   ├── finetune.py            # Stage 2: intermediate experiment (not used in final)
│   ├── augmentedFinetune.py   # Stage 3: final model — heavy aug + clean val split
│   ├── evaluate.py            # Test accuracy, classification report, confusion matrix
│   ├── predict.py             # Random test image → prediction + Grad-CAM
│   └── app.py                 # Streamlit web application
├── assets/                    # Screenshots and evaluation plots
├── requirements.txt
└── LICENSE
```

Created on your machine, not committed:

```
data/
├── Training/                  # Training images, one folder per class
└── Testing/                   # Test images, one folder per class
models/
├── brain_tumor_efficientnet.pth            # Stage 1 (baseline)
├── brain_tumor_efficientnet_finetuned.pth  # Stage 2
└── brain_tumor_efficientnet_augmented.pth  # Stage 3 (final, used by app and predict)
```

Run every command from the repo root. The scripts read `./data/` and write `./models/` relative to where you run them.

---

## ⚙️ Training Pipeline

### Stage 1 — `train.py` *(baseline)*
- Loads pretrained **EfficientNet-B0** from `timm` (ImageNet weights)
- Replaces the classification head with a 4-class output layer
- Trains for **5 epochs**, AdamW, LR 0.001, random 80/20 train/val split
- Light augmentation (`RandomHorizontalFlip`, `RandomRotation(10)`), applied to both splits
- Saves to `models/brain_tumor_efficientnet.pth`
- **Result: 93.06% test accuracy, macro F1 0.9296**

### Stage 2 — `finetune.py` *(intermediate, not used in final)*
- Continues from Stage 1 weights with the same light augmentation
- 10 epochs at LR 0.0001, `weight_decay=1e-4`
- Normalises with mean/std 0.5 rather than the ImageNet statistics the other stages use
- Kept in the repo for reference; superseded by Stage 3 and **not re-evaluated** under the current test protocol, so no comparable figure is reported

### Stage 3 — `augmentedFinetune.py` *(final model)*
- Loads Stage 1 weights and applies **heavy augmentation** to training data:
  `RandomHorizontalFlip`, `RandomRotation(15)`, `RandomAffine`, `ColorJitter`
- 10 epochs at LR 0.0001, `weight_decay=1e-4`
- Validation kept **clean** (no augmentation): an index-based 80/20 split via `numpy` gives the augmented training subset and the clean validation subset no images in common
- The split is re-drawn on every run (no fixed seed) and independently of Stage 1's split, so most validation images were already trained on in Stage 1. Treat Stage 3 validation accuracy as optimistic; the test set is unaffected
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
| Devices | Apple MPS or CPU (the scripts don't select CUDA) |

Roughly 67 scans per second at that latency.

---

## 🔍 Grad-CAM Explainability

Both `predict.py` and `app.py` use **Gradient-weighted Class Activation Mapping** to explain predictions visually.

- Target layer: `model.conv_head` (final convolutional layer of EfficientNet-B0)
- Highlights the regions of the MRI that most influenced the classification
- Rendered as a color overlay on the original scan

![Grad-CAM heatmap from the Streamlit app](assets/HeatMap.png)

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
streamlit run src/app.py
```

The app loads `models/brain_tumor_efficientnet_augmented.pth`. Trained weights aren't committed, so run the training steps below first, or put your own checkpoint at that path.

> Confidence scores are uncalibrated — see Limitations.

---

## 📊 Evaluation — `evaluate.py`

```bash
python src/evaluate.py
```

Runs the final augmented model against the held-out `Testing/` directory and reports overall accuracy, per-class `classification_report` (precision / recall / F1), and a confusion matrix heatmap. The extra metrics in Results (test loss, ROC-AUC, κ, MCC, top-2, latency) aren't computed by the committed `evaluate.py` yet.

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

**Dataset:**
Download the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle and place it in `./data/` with `Training/` and `Testing/` subdirectories. Make sure you have the deduplicated version: 5,600 training and 1,600 test images, 400 per class in test. The original version (5,712 / 1,311) has near-duplicates across the split.

**Train from scratch:**
```bash
python src/train.py
python src/augmentedFinetune.py
python src/evaluate.py
```

---

## 🧪 Quick Prediction Test

```bash
python src/predict.py
```

Picks a random image from the test set, runs the final model, and displays the original scan alongside the Grad-CAM heatmap — color-coded green (correct) or red (incorrect).

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Model | EfficientNet-B0 (`timm`) |
| Framework | PyTorch |
| Explainability | Grad-CAM (`grad-cam` package) |
| UI | Streamlit |
| Data | `torchvision.datasets.ImageFolder` |
| Hardware | Apple MPS / CPU |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

> ⚠️ **Disclaimer:** NeuroScan AI is a research and educational project. It is not intended for clinical use and does not replace professional medical diagnosis. The limitations above are not boilerplate — a 17% glioma miss rate is disqualifying for any diagnostic application.
