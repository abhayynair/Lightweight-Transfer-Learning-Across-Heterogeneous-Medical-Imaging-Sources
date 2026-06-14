# Implicit Domain Adaptation Through Pretrained Feature Reuse

### An Empirical Study of Lightweight Transfer Learning Across Heterogeneous Medical Imaging Sources

**Minor Specialization Project — Computational Intelligence**
Department of Computer Science & Engineering, Manipal Institute of Technology, MAHE

**Team:** Abhay Sivakumar Nair (220905199) · Shruthi Andra (230905223)
**Guide:** Dr. Tyson B D Cunha
**Target venue:** IEEE Access (submission targeted September 2026)

---

## 1. Overview

This project investigates whether lightweight ImageNet-pretrained convolutional neural networks (EfficientNetB0, MobileNetV2, ResNet50) provide **implicit domain adaptation** across heterogeneous medical imaging sources, removing the need for computationally expensive explicit alignment techniques such as adversarial domain adaptation or GAN-based style transfer.

The hypothesis is motivated by prior empirical work in structural crack detection, where a custom CNN trained from scratch achieved 99.77% accuracy on single-source data but degraded to 88.96% on pooled multi-source data with unstable convergence, while EfficientNetB0 under identical conditions achieved 98.14% with stable convergence. This project systematically tests whether that pattern holds for medical imaging, where cross-hospital, cross-scanner, and cross-population generalization failures are well documented and clinically significant.

**Planned scope:** three diagnostic tasks — skin lesion classification, chest X-ray pathology detection, and diabetic retinopathy grading — each evaluated across multiple heterogeneous data sources.

**Status:** Skin lesion classification (Phase 1) is complete. Chest X-ray and diabetic retinopathy experiments are planned next.

---

## 2. Core Hypothesis

> Lightweight ImageNet-pretrained CNNs encode sufficiently domain-invariant feature representations to bridge scanner-induced and population-induced distribution shifts in medical images — without any explicit domain alignment mechanism — while custom CNNs trained from scratch on the same pooled data fail to generalize across sources.

---

## 3. Skin Lesion Experiment (Complete)

### 3.1 Datasets

| Dataset | Source / Origin | Images Used | Classes |
|---|---|---|---|
| [ISIC 2019](https://www.kaggle.com/datasets/andrewmvd/isic-2019) | Global, multi-institution | 25,331 | 6 shared |
| [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) | Vienna + Australia (multi-dermatoscope) | 9,688 | 6 shared |
| **Combined (pooled)** | Multi-source | **35,019** | **6** |

**Shared diagnostic classes:** melanoma, nevus, basal cell carcinoma (bcc), benign keratosis (bkl), dermatofibroma (df), vascular lesion (vasc). Classes present in only one dataset (`AK`, `SCC` from ISIC 2019; `akiec` from HAM10000) were dropped to retain a common label space across both sources.

### 3.2 Data Pipeline

- Each source's metadata was mapped to a unified label set and tagged with a `source` column (`isic2019` / `ham10000`) for domain-aware analysis.
- **Stratified 70/15/15 train/val/test split**, performed independently per source then concatenated, ensuring both sources and all six classes are proportionally represented in every split.
  - Train: 24,512 images
  - Validation: 5,253 images
  - Test: 5,254 images
- **Class weighting** (`sklearn.utils.class_weight.compute_class_weight`, balanced) applied during training to address severe class imbalance (nevus: 19,580 images vs. df: 354 images across the combined dataset).
- **Data augmentation** (training only): rotation, width/height shift, horizontal flip, zoom.
- **Preprocessing:** each architecture uses its own standard preprocessing function (`efficientnet_preprocess`, `mobilenet_preprocess`, `resnet_preprocess` for the pretrained models; rescale-to-[0,1] for the custom CNN), per each architecture's original specification.

### 3.3 Model Architectures

| Model | Parameters | Pretraining | Notes |
|---|---|---|---|
| **Custom CNN** | 322,470 | None (random init) | 3-block CNN: Conv→BN→Conv→BN→MaxPool→Dropout per block (32→64→128 filters), GlobalAvgPool, Dense(256), Dense(6, softmax) |
| **MobileNetV2** | 2,587,462 | ImageNet | `include_top=False` backbone + GlobalAvgPool, Dense(256), Dropout(0.3), Dense(6, softmax) |
| **EfficientNetB0** | 4,379,049 | ImageNet | Same head structure as MobileNetV2 |
| **ResNet50** | 24,113,798 | ImageNet | Same head structure as MobileNetV2 |

All three pretrained models follow a **two-phase fine-tuning protocol**:
- **Phase 1:** backbone frozen, train classification head only (Adam, lr=1e-3)
- **Phase 2:** full backbone unfrozen, fine-tune end-to-end (Adam, lr=1e-5), with `EarlyStopping` (patience=7, restore best weights) and `ModelCheckpoint` on `val_accuracy`

The custom CNN is trained from scratch for up to 30 epochs with the same early-stopping protocol.

### 3.4 Training Summary

| Model | Total Epochs (all phases) | Best Val Accuracy |
|---|---|---|
| Custom CNN | 27 (early stopped) | 57.66% |
| MobileNetV2 | 58 (Phase 1 + Phase 2 + continuation) | 74.81% |
| EfficientNetB0 | 100 (Phase 1 + Phase 2 + 3 continuations) | 82.58% |
| ResNet50 | 60 (Phase 1 + Phase 2 + continuation) | 88.03% |

Per-phase training curves (CSV logs) are in [`results/training_logs/`](results/training_logs/).

### 3.5 Final Test Set Results

Evaluated on the held-out test set (5,254 images, never seen during training):

| Model | Test Accuracy | Macro F1 | Macro AUC | Gap vs Custom CNN |
|---|---|---|---|---|
| Custom CNN | **52.84%** | 0.29 | 0.754 | — |
| MobileNetV2 | **76.63%** | 0.73 | 0.957 | +23.79 pts |
| EfficientNetB0 | **82.98%** | 0.81 | 0.972 | +30.14 pts |
| ResNet50 | **88.35%** | 0.85 | 0.981 | +35.51 pts |

**All three pretrained architectures outperform the custom CNN by 24–35.5 percentage points on identical pooled multi-source data**, supporting the implicit domain adaptation hypothesis. ResNet50 achieves the highest raw accuracy but shows the largest train–validation gap (97.5% train vs 88.0% val), indicating overfitting given its 24.1M parameters relative to the 24,512-image training set; EfficientNetB0 and MobileNetV2 show smaller train–val gaps.

### 3.6 Statistical Validation

**McNemar's test** (Custom CNN vs EfficientNetB0, on test set predictions):

| Both Correct | CNN Correct, Eff Wrong | CNN Wrong, Eff Correct | Both Wrong | Statistic | p-value |
|---|---|---|---|---|---|
| 2,401 | 375 | 1,959 | 519 | 1073.65 | **1.77 × 10⁻²³⁵** |

The accuracy difference is significant at an overwhelming level. EfficientNetB0 correctly classified 1,959 images that the custom CNN got wrong, while the custom CNN recovered only 375 images EfficientNetB0 missed — a ~5:1 asymmetry.

**Per-class AUC with DeLong 95% confidence intervals** (Custom CNN vs EfficientNetB0):

| Class | Custom CNN AUC (95% CI) | EfficientNetB0 AUC (95% CI) |
|---|---|---|
| bcc | 0.776 (0.752–0.799) | 0.986 (0.979–0.993) |
| bkl | 0.676 (0.650–0.701) | 0.968 (0.958–0.979) |
| df | 0.786 (0.713–0.860) | 0.983 (0.958–1.000) |
| melanoma | 0.645 (0.626–0.665) | 0.928 (0.917–0.939) |
| nevus | 0.734 (0.721–0.747) | 0.965 (0.961–0.970) |
| vasc | 0.907 (0.857–0.958) | 1.000 (0.999–1.000) |

Confidence intervals do not overlap between models for any class. The melanoma class — the most clinically critical category — shows the largest AUC improvement (0.645 → 0.928).

### 3.7 Per-Class Performance (Test Set)

<details>
<summary><b>Custom CNN — Macro F1 0.29</b></summary>

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bcc | 0.35 | 0.09 | 0.15 | 576 |
| bkl | 0.35 | 0.06 | 0.10 | 558 |
| df | 0.04 | 0.45 | 0.07 | 53 |
| melanoma | 0.33 | 0.24 | 0.28 | 1070 |
| nevus | 0.69 | 0.81 | 0.74 | 2937 |
| vasc | 0.38 | 0.42 | 0.40 | 60 |

</details>

<details>
<summary><b>MobileNetV2 — Macro F1 0.73</b></summary>

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bcc | 0.75 | 0.82 | 0.78 | 576 |
| bkl | 0.45 | 0.87 | 0.59 | 558 |
| df | 0.53 | 0.77 | 0.63 | 53 |
| melanoma | 0.69 | 0.62 | 0.65 | 1070 |
| nevus | 0.94 | 0.79 | 0.86 | 2937 |
| vasc | 0.89 | 0.90 | 0.89 | 60 |

</details>

<details>
<summary><b>EfficientNetB0 — Macro F1 0.81</b></summary>

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bcc | 0.80 | 0.85 | 0.83 | 576 |
| bkl | 0.66 | 0.84 | 0.74 | 558 |
| df | 0.79 | 0.77 | 0.78 | 53 |
| melanoma | 0.69 | 0.75 | 0.72 | 1070 |
| nevus | 0.94 | 0.85 | 0.89 | 2937 |
| vasc | 0.87 | 1.00 | 0.93 | 60 |

</details>

<details>
<summary><b>ResNet50 — Macro F1 0.85</b></summary>

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| bcc | 0.82 | 0.89 | 0.85 | 576 |
| bkl | 0.77 | 0.83 | 0.80 | 558 |
| df | 0.80 | 0.75 | 0.78 | 53 |
| melanoma | 0.80 | 0.81 | 0.80 | 1070 |
| nevus | 0.95 | 0.92 | 0.94 | 2937 |
| vasc | 0.91 | 0.97 | 0.94 | 60 |

</details>

The custom CNN's per-class results reveal it defaults heavily toward the majority class (nevus, 81% recall) while almost completely failing on minority classes (bcc 9% recall, bkl 6% recall) — clinically dangerous behavior. All three pretrained models distribute predictions across all six classes with substantially higher per-class recall, including on the rarest classes (df: 53 test images, vasc: 60 test images).

### 3.8 Interpretability — Grad-CAM

Grad-CAM attention maps were generated for all four architectures on representative samples:

| Figure | Description |
|---|---|
| [`results/figures/gradcam_melanoma.png`](results/figures/gradcam_melanoma.png) | **Primary figure.** EfficientNetB0 is the only model to correctly classify this melanoma sample, with attention concentrated on the pigmented lesion. Custom CNN misclassifies as nevus with attention on an isolated, non-lesion hotspot. |
| [`results/figures/gradcam_bcc.png`](results/figures/gradcam_bcc.png) | Three of four pretrained models (MobileNetV2, EfficientNetB0, ResNet50) correctly classify with lesion-centered attention; Custom CNN misclassifies with scattered attention. |
| [`results/figures/gradcam_df.png`](results/figures/gradcam_df.png) | **Honest limitation example.** All four models misclassify this rare-class (df, 53 test images) sample, consistent with df's lower per-class recall across all models. |

### 3.9 Confusion Matrices

Normalized (per-class recall) confusion matrices for all four models: [`results/figures/confusion_matrices_all4.png`](results/figures/confusion_matrices_all4.png)

The custom CNN's matrix shows mass concentrated toward the nevus and df columns regardless of true class. All three pretrained models show strong diagonal concentration across all six classes.

---

## 4. Repository Structure

```
.
├── README.md
├── notebooks/
│   └── skin_lesion_4model_comparison.ipynb   # Full training + evaluation pipeline
├── models/
│   ├── best_custom_cnn.keras
│   ├── best_mobilenet_v2.keras
│   ├── best_efficientnet_v4.keras
│   └── best_resnet50_v2.keras
├── results/
│   ├── training_logs/
│   │   ├── efficientnet_training_log.csv     # Phase 2 (30 epochs)
│   │   ├── efficientnet_v2_log.csv           # Continuation 1 (20 epochs)
│   │   ├── efficientnet_v3_log.csv           # Continuation 2 (20 epochs)
│   │   ├── efficientnet_v4_log.csv           # Continuation 3 (20 epochs)
│   │   ├── mobilenet_log.csv                 # Phase 2 (30 epochs)
│   │   ├── mobilenet_v2_log.csv              # Continuation (18 epochs)
│   │   ├── resnet50_log.csv                  # Phase 2 (30 epochs)
│   │   └── resnet50_v2_log.csv               # Continuation (20 epochs)
│   └── figures/
│       ├── confusion_matrices_all4.png
│       ├── gradcam_melanoma.png
│       ├── gradcam_bcc.png
│       └── gradcam_df.png
├── reports/
│   ├── Week1_Progress_Report.docx
│   ├── Week2_Progress_Report.docx
│   └── Week3_Progress_Report.docx
└── docs/
    ├── synopsis.pdf                          # Minor project synopsis
    └── technical_implementation_plan.pdf     # Full experimental design document
```

> Note: Phase 1 (frozen-backbone, head-only) training logs for EfficientNetB0, MobileNetV2, and ResNet50 were not separately saved as CSVs; their per-epoch validation accuracies are recorded in the Week 3 progress report.

---

## 5. Environment & Reproducibility

| | |
|---|---|
| **Framework** | TensorFlow / Keras 2.19.0 |
| **Hardware** | Kaggle Notebooks, Tesla T4 × 2 GPU |
| **Image size** | 224 × 224 |
| **Batch size** | 32 |
| **Optimizer** | Adam (lr=1e-3 for frozen-backbone phase, lr=1e-5 for fine-tuning phase) |
| **Loss** | Categorical cross-entropy with class weighting |
| **Early stopping** | `val_accuracy`, patience=7, restore best weights |

**Reproducibility note:** Skin lesion experiments (this phase) were conducted without a fixed random seed; run-to-run variance of 1–4 percentage points was observed for continuation runs due to unseeded data augmentation. The reported metrics in §3.5–3.7 are from re-verified evaluations of the final saved checkpoints on the fixed test set (re-running `model.evaluate()` on the saved weights reproduces these numbers exactly). Future experiments (cross-source baseline, chest X-ray, diabetic retinopathy) will use a fixed seed (42) set before any data/model operations for full reproducibility.

---

## 6. Roadmap

- [x] Skin lesion: 4-model pooled training, evaluation, statistical validation, Grad-CAM, confusion matrices
- [ ] Skin lesion: cross-source baseline (train ISIC-only → test HAM-only, and vice versa)
- [ ] Skin lesion: Maximum Mean Discrepancy (MMD) domain gap quantification
- [ ] Skin lesion: fine-tuning depth ablation (frozen / partial / full)
- [ ] Skin lesion: CORAL baseline comparison
- [ ] Skin lesion: 5-fold cross-validation (mean ± std reporting)
- [ ] Chest X-ray classification (CheXpert, VinDr-CXR, NIH ChestX-ray14)
- [ ] Diabetic retinopathy grading (EyePACS, APTOS 2019, Messidor-2)
- [ ] Manuscript preparation and IEEE Access submission

---

## 7. Acknowledgments

This project builds directly on prior work in CUDA-accelerated structural crack detection, which first surfaced the implicit-domain-adaptation pattern this study formalizes and tests.

Portions of the code in this repository (data pipeline construction, model training scripts, statistical analysis, and visualization code) were developed with the assistance of **Claude (Anthropic)**. All AI-assisted code was reviewed, executed, and validated by the authors, who take full responsibility for the correctness of all results reported here.

---

## 8. Citation

If you use this work, please cite (paper details to be updated upon publication):

```bibtex
@article{nair2026implicit,
  title={Implicit Domain Adaptation Through Pretrained Feature Reuse: An Empirical Study of Lightweight Transfer Learning Across Heterogeneous Medical Imaging Sources},
  author={Nair, Abhay Sivakumar and Andra, Shruthi},
  journal={IEEE Access (submitted)},
  year={2026}
}
```
