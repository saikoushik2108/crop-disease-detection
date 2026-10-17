# Multimodal Deep Learning for Paddy Disease Classification

[![Status](https://img.shields.io/badge/Status-Complete-green)](file:///e:/crop-detection/FINAL_PROJECT_STATUS.md)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Frontend-React-61DAFB)](https://react.dev/)
[![Paper](https://img.shields.io/badge/Paper-Research%20Draft-blue)](file:///e:/crop-detection/paper/research_paper.md)

An academic research and production-grade implementation for automated paddy (*Oryza sativa*) disease classification using deep multimodal fusion of in-situ RGB leaf images and agronomic metadata (cultivar variety + chronological crop age).

---

## 🏆 Key Experimental Results

Evaluated rigorously on the **Paddy Doctor** benchmark (10,407 real in-situ field images) using a zero-leakage stratified test partition ($N_{\text{test}} = 1,561$ samples, 10 classes):

| Experiment | Modality | Test Accuracy | Macro F1 | Macro Recall | Macro Precision | Weighted F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Exp B: Context-Only** | Variety + Crop Age (MLP) | 36.68% | 0.3240 | 0.3458 | 0.3741 | 0.3612 |
| **Exp A: Visual-Only** | ResNet-18 (RGB Leaf Image) | 73.11% | 0.7070 | 0.7355 | 0.7235 | 0.7385 |
| **Exp C: Multimodal Fusion** | Visual + Variety + Age | **85.47%** | **0.8399** | **0.8703** | **0.8259** | **0.8561** |
| **Net Multimodal Gain** | **Exp C vs Exp A Baseline** | **+12.36%** | **+0.1329** | **+0.1348** | **+0.1024** | **+0.1176** |

### Per-Class F1 Gains
Multimodal conditioning improved diagnostic performance across **all 10 disease categories**:
- `bacterial_panicle_blight`: **0.9074 F1** (+24.83% over visual baseline; recall: 98.0%)
- `bacterial_leaf_streak`: **0.8029 F1** (+20.86% over visual baseline)
- `downy_mildew`: **0.7387 F1** (+17.62% over visual baseline)
- `blast`: **0.8454 F1** (+14.20% over visual baseline; precision: 91.5%)
- `hispa`: **0.8535 F1** (+13.92% over visual baseline)
- `brown_spot`: **0.8062 F1** (+11.36% over visual baseline; precision: 92.0%)
- `normal` (Healthy): **0.9370 F1** (+9.31% over visual baseline; recall: 95.5%)
- `tungro`: **0.7649 F1** (+9.36% over visual baseline)
- `bacterial_leaf_blight`: **0.7952 F1** (+5.84% over visual baseline)
- `dead_heart`: **0.9474 F1** (+5.54% over visual baseline; precision: 98.5%)

---

## 🛠️ Architecture

The system implements intermediate feature fusion (`MultimodalCropNet`):
- **Visual Stream:** RGB Image ($224 \times 224$) $\rightarrow$ ResNet-18 feature backbone $\rightarrow$ 512-dimensional visual vector.
- **Agronomic Tabular Stream:** 
  - Variety index $\rightarrow$ Learnable Entity Embedding (16d).
  - Standardized Crop Age (days) $\rightarrow$ Dense Linear Projection (16d).
  - Concatenation (32d) $\rightarrow$ Tabular MLP with BatchNorm $\rightarrow$ 64-dimensional tabular vector.
- **Multimodal Fusion:** Concatenation (576d) $\rightarrow$ Dense Linear (128d) with BatchNorm, ReLU, Dropout(0.3) $\rightarrow$ 10 Disease Class Logits.
- **Optimization:** Warm-start from visual baseline, backbone freezing (epochs 1–5), and differential learning rate fine-tuning ($\eta_{\text{backbone}}=10^{-5}$, $\eta_{\text{fusion}}=3 \times 10^{-4}$).

---

## 🚀 Quickstart & Execution Guide

### 1. Backend Server (FastAPI)
```powershell
# In project root
.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- Interactive Swagger API: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Predict Endpoint: `POST http://localhost:8000/predict` (supports image, variety, age, lat, lon)

### 2. Frontend Web Dashboard (React + Vite + Tailwind CSS)
```powershell
cd frontend
npm install
npm run dev
```
- Access web interface at: `http://localhost:5173`

### 3. Evaluate Checkpoint on Test Set
```powershell
.venv\Scripts\python.exe src/evaluation/evaluate_saved_checkpoint.py
```

### 4. Regenerate Research Outputs (Ablation Tables & Grad-CAM)
```powershell
.venv\Scripts\python.exe src/evaluation/generate_real_research_outputs.py
```

---

## 📂 Repository Structure

```
crop-disease-detection/
├── api/                    # FastAPI REST backend & advisory endpoints
├── configs/                # Global YAML configuration
├── data/                   # Real Paddy Doctor dataset & train/val/test splits
│   └── processed/splits/   # Zero-leakage stratified splits (70/15/15)
├── frontend/               # React + Tailwind CSS web dashboard
├── models/                 # Model checkpoints (.pt) and preprocessing encoders
├── paper/                  # Complete academic conference research paper
│   └── research_paper.md
├── research/               # Ablation studies, audits, and scientific documentation
│   ├── REAL_ABLATION_STUDY.md
│   └── figures/
├── results/                # Evaluation metrics JSONs, CSV summaries, and Grad-CAM figures
│   ├── metrics/
│   └── figures/
└── src/                    # Source code modules
    ├── data/               # Multimodal PyTorch dataset loader
    ├── evaluation/         # Metrics, Grad-CAM, and ablation scripts
    ├── inference/          # Multi-context inference engine
    ├── models/             # PyTorch model definitions (ResNet18, MLP, Fusion)
    ├── preprocessing/      # Image transforms & tabular normalization
    ├── training/           # Phased trainer with warm-start & differential LR
    └── utils/              # Configuration, seeding, and device utilities
```

---

## 📜 Academic Research Paper
A conference research manuscript is provided in [`paper/research_paper.md`](file:///e:/crop-detection/paper/research_paper.md), ready for formatting into IEEE/Springer conference templates.
