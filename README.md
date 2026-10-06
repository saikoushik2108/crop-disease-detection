# Multimodal Multi-Context Aware Paddy Disease Classification

![Status](https://img.shields.io/badge/Status-Complete-green)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)
![React](https://img.shields.io/badge/Frontend-React-61DAFB)

A professional research implementation for identifying paddy diseases using a multimodal fusion of visual leaf features and agronomic context, augmented by real-time environmental risk analysis.

## 🚀 Overview
Traditional disease classification relies solely on images. This system improves reliability by combining:
1. **Visual Features**: ResNet-18 extracted from leaf images.
2. **Agronomic Context**: Paddy variety and plant age.
3. **Runtime Context**: Real-time weather and geographic data via external APIs.

## 🛠️ Architecture
The system employs a **Late Fusion** architecture:
- **Visual Stream**: Image $\rightarrow$ ResNet-18 $\rightarrow$ 512-d vector.
- **Tabular Stream**: Variety + Age $\rightarrow$ MLP $\rightarrow$ 64-d vector.
- **Fusion**: Concatenation $\rightarrow$ Dense Layers $\rightarrow$ 10 Disease Classes.
- **Context Layer**: Runtime API data is used for interpretation, not as trained features.

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/saikoushik2108/crop-disease-detection.git
cd crop-disease-detection
```

### 2. Backend Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\\Scripts\\activate     # Windows
pip install -r requirements.txt
```
Configure your `.env` file based on `.env.example`.

### 3. Dataset & Models
- Place the Paddy Doctor dataset in `data/raw/` (See `data/README.md`).
- Place trained weights in `models/` (See `models/README.md`).

### 4. Running the System
```bash
# Start Backend
uvicorn api.main:app --reload

# Start Frontend
cd frontend
npm install
npm run dev
```

## 🔬 Research Features
- **Multi-Context Awareness**: Combines predictive ML with real-time environmental risk indicators.
- **Explainability**: Integrated Grad-CAM heatmaps to visualize disease-driving regions of the leaf.
- **Scientific Honesty**: Strict separation between trained modalities and runtime context to prevent data fabrication.

## 📂 Project Structure
- `src/`: Core ML pipeline (Data $\rightarrow$ Models $\rightarrow$ Training $\rightarrow$ Inference).
- `api/`: FastAPI backend implementation.
- `frontend/`: Professional React dashboard.
- `docs/`: Architecture, Deployment, and API guides.
- `research/`: Data integrity reports and ablation study results.
- `models/`: Model configuration and weights management.
