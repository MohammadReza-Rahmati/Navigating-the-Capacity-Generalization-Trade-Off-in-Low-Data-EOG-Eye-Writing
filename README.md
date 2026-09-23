# Navigating the Capacity-Generalization Trade-Off in Low-Data EOG Eye-Writing

**A Systematic Multi-Architecture Study**  
*Submitted to Biomedical Signal Processing and Control (BSPC), Elsevier*

[![Paper](https://img.shields.io/badge/paper-BSPC-blue.svg)](https://www.journals.elsevier.com/biomedical-signal-processing-and-control)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Official implementation of our systematic empirical study on the trade-off between **model capacity** and **cross-user generalization** in data-scarce electrooculography (EOG) eye-writing — an assistive communication paradigm for individuals with amyotrophic lateral sclerosis (ALS).

---

## 📖 Overview

Collecting large-scale EOG datasets is constrained by user fatigue and physiological variability. Instead of proposing yet another opaque architecture, this repository provides a **rigorous, reproducible benchmark** of a family of 1D-CNNs — from standard to ultra-lightweight — under three complementary evaluation protocols:

| Protocol | Scenario | Training data per fold |
|---|---|---|
| **Subject-Mixed (5-fold)** | Calibrated system (upper-bound reference) | ~579 samples |
| **User-Dependent (10-fold/subject)** | Personalized deployment | ~108 samples |
| **Leave-One-Subject-Out (6-fold)** | Plug-and-play, user-independent | ~604 samples |

### Key contributions
- **Systematic capacity–generalization analysis:** high capacity benefits personalized calibration, while reduced capacity acts as an *implicit regularizer* for user-independent (LOSO) generalization.
- **Difference channel (`x_h − x_v`):** a trivially derived stabilization signal that prevents catastrophic degradation in ultra-lightweight models (ablation: up to **+22.65 pp**).
- **Structure-constrained character decoder:** stroke-level confusion probabilities + Hamming-distance candidate selection over valid Katakana stroke sequences (1–4 strokes), bridging the stroke→character gap (up to **98.77 %** character accuracy).
- **Embedded-deployment evidence:** a 10,048-parameter model (39.2 kB, 6.3 MFLOPs) retaining **82.54 %** LOSO character accuracy *without any synthetic data augmentation*.

---

## 🏗️ Model Zoo

| Model | Params | Size (kB) | FLOPs (M) | Role |
|---|---:|---:|---:|---|
| `conv1d` (standard) | 157,024 | 613.4 | 69.8 | Best personalized (user-dependent) |
| `conv1d_fast` (lightweight) | 34,624 | 135.2 | 24.0 | Best user-independent (LOSO) |
| `conv1d_faster` (ultra-light) | 10,048 | 39.2 | 6.3 | Embedded / edge deployment |

Each model runs as **three parallel branches** (horizontal, vertical, difference channel) followed by a lightweight fusion classifier.

---

## 📊 Main Results (fused classifier)

| Model | Subject-Mixed (stroke / char) | User-Dep. (stroke / char) | LOSO (stroke / char) |
|---|---|---|---|
| `conv1d` | **92.54 / 94.75** | **90.33 / 92.78** | 76.24 / 81.92 |
| `conv1d_fast` | 85.36 / 93.64 | 72.10 / 79.48 | **79.14 / 88.23** |
| `conv1d_faster` | 81.91 / 88.66 | 64.09 / 72.58 | 71.13 / 82.54 |

**Comparison with prior work (same protocols):**
- User-Dependent stroke Macro-F1: **0.9040** vs. DTW 0.776 and GMM-HMM 0.865 (Fang & Shinozaki, 2018).
- LOSO stroke accuracy: **79.14 %** *without augmentation* vs. 75.64 % (direct train) and 80.36 % (9× diffusion-augmented) (Choi et al., 2025).

---

## 📁 Repository Structure

All experiments are organized **by evaluation protocol**, mirroring the paper's experimental design:

```
.
├── Subject_Mixed/
│   ├── stroke recognition/
│   │   ├── Conv1d_Mixed.ipynb
│   │   ├── Conv1d_fast_Mixed.ipynb
│   │   ├── Conv1d_faster_Mixed.ipynb
│   │   └── Ablation(Distance)/
│   │       ├── conv1d_DistanceAblation.ipynb
│   │       ├── conv1d_fast_DistanceAblation.ipynb
│   │       └── conv1d_faster_DistanceAblation.ipynb
│   ├── Character(Decoder)/
│   │   ├── Decoder_conv1d_Mixed.ipynb
│   │   ├── Decoder_conv1d_fast_Mixed.ipynb
│   │   └── Decoder_conv1d_faster_Mixed.ipynb
│   └── CI/
│       └── CI_Mixed.ipynb
├── User Dependent/
│   ├── stroke recognition/
│   │   ├── Conv1d_UD.ipynb
│   │   ├── Conv1d_fast_UD.ipynb
│   │   └── Conv1d_faster_UD.ipynb
│   ├── Character(Decoder)/
│   │   ├── Decoder_conv1d_UD.ipynb
│   │   ├── Decoder_conv1d_fast_UD.ipynb
│   │   └── Decoder_conv1d_faster_UD.ipynb
│   └── Wilcoxon/
│       └── Wilcoxon_UD.ipynb
├── LOSO/
│   ├── stroke recognition/
│   │   ├── Conv1d_LOSO.ipynb
│   │   ├── Conv1d_fast_LOSO.ipynb
│   │   └── Conv1d_faster_LOSO.ipynb
│   ├── Character(Decoder)/
│   │   ├── Decoder_conv1d_LOSO.ipynb
│   │   ├── Decoder_conv1d_fast_LOSO.ipynb
│   │   └── Decoder_conv1d_faster_LOSO.ipynb
│   └── Wilcoxon/
│       └── Wilcoxon_LOSO.ipynb
└── complexity_analysis.ipynb
```

### Notebook → paper mapping

| Notebook group | Reproduces |
|---|---|
| `*/stroke recognition/Conv1d*_*.ipynb` | Per-branch & fused stroke metrics, normalized confusion matrices, per-class Precision/Recall/F1 |
| `*/Character(Decoder)/Decoder_*.ipynb` | stroke-level conditional probabilities (Eq. 1), Character accuracy per stroke-length (1–4 strokes) and Overall; implements the structure-constrained decoder (Eqs. 2–4) |
| `Subject_Mixed/CI/CI_Mixed.ipynb` | 95 % confidence intervals across 5 folds |
| `Subject_Mixed/Ablation(Distance)/*.ipynb` | Difference-channel ablation (without `x_h − x_v`) |
| `User Dependent/Wilcoxon/Wilcoxon_UD.ipynb` | Wilcoxon signed-rank tests across 6 subjects |
| `LOSO/Wilcoxon/Wilcoxon_LOSO.ipynb` | Wilcoxon signed-rank tests across 6 LOSO folds |
| `complexity_analysis.ipynb` | Params / model size / FLOPs / inference time profiling |

---

## 🧠 Dataset

We use the **public EOG eye-writing benchmark** introduced by Fang & Shinozaki (2018), *PLOS ONE* ([doi:10.1371/journal.pone.0192684](https://doi.org/10.1371/journal.pone.0192684)):

- 6 healthy participants, 12 Japanese Katakana strokes, **724 isolated strokes**
- 2 channels (horizontal, vertical), 1.0 kHz, 1250 samples (1.25 s) per stroke
- Original ethics approval: Tokyo Institute of Technology, No. 2014083

> The dataset is **not redistributed** here. Download it from the original publication's Supporting Information.

---

## 🚀 Getting Started

### 1. Requirements
```bash
pip install jupyter tensorflow>=2.20 numpy pandas scipy scikit-learn matplotlib seaborn
```

### 2. Execution order (per protocol)
The notebooks have a strict dependency chain — run them in this order:

1. **Stroke recognition** — train the three branches + fusion classifier over all folds; exports confusion counts.
   ```bash
   jupyter nbconvert --to notebook --execute "LOSO/stroke recognition/Conv1d_fast_LOSO.ipynb" --output Conv1d_fast_LOSO_out.ipynb
   ```
2. **Character (Decoder)** — confusion probabilities + the valid Katakana stroke lookup table; computes character-level accuracy.
3. **Statistics** — `CI_Mixed.ipynb` (Subject-Mixed) or `Wilcoxon_UD/LOSO.ipynb` (per-subject / per-fold tests).
4. **Ablation** (Subject-Mixed only) — `Ablation(Distance)/` notebooks.
5. **Complexity** — `complexity_analysis.ipynb` can be run at any time.

### 3. Hyperparameters 
- Subject-Mixed / LOSO: 600 epochs, batch size 32 · User-Dependent: 450 epochs, batch size 16 (set in each notebook's config cell).
- No synthetic data augmentation is used anywhere; all results reflect the original, unaltered data distribution.

> 💡 Folder names contain spaces/parentheses for readability. If you prefer CLI-friendly paths, you may rename them.

---

## 📚 Citation

If you use this code or benchmark in your research, please cite:

```bibtex
@article{rahmati2026navigating,
  title   = {Navigating the Capacity-Generalization Trade-Off in Low-Data EOG Eye-Writing: A Systematic Multi-Architecture Study},
  author  = {Rahmati, Mohammad Reza and Zahabi, Seyed Jalal and Manshaei, Mohammad Hossein},
  journal = {Biomedical Signal Processing and Control},
  year    = {2026},
  note    = {Under review}
}
```
## 📄 License

This project is released under the [Apache License 2.0](LICENSE).
---

## 🙏 Acknowledgements

- We thank **F. Fang and T. Shinozaki** for publicly releasing the EOG eye-writing dataset that made this study possible.

---

## 📬 Contact

Mohammad Reza Rahmati — `m.rahmati@ec.iut.ac.ir`  
Department of Electrical and Computer Engineering, Isfahan University of Technology, Iran
