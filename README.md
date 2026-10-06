# Multimodal Deep Learning for Crop Disease Detection

A research prototype investigating whether combining **visual features** (crop leaf images) with **environmental/agronomic context** (variety, crop age, regional climate baselines) improves crop disease classification accuracy over image-only baselines.

## Research Question

> *Can context-aware multimodal fusion (image + tabular environmental features) outperform unimodal image classification for paddy disease detection, and does Grad-CAM analysis reveal more pathologically meaningful attention patterns in the multimodal model?*

## Architecture

```
┌─────────────────┐    ┌──────────────────────┐
│  Image Stream   │    │   Tabular Stream     │
│  (ResNet-18)    │    │   (MLP Encoder)      │
│  → 512-d feat   │    │   → 64-d feat        │
└────────┬────────┘    └──────────┬───────────┘
         │                        │
         └──────┬─────────────────┘
                │  Concatenation
         ┌──────▼──────┐
         │ Fusion Head │
         │ 576 → 256   │
         │ → N classes │
         └─────────────┘
```

## Ablation Study

| Experiment | Input | Purpose |
|---|---|---|
| **A** – Image Only | Leaf images | Unimodal visual baseline |
| **B** – Tabular Only | Variety + Age + Environment | Context-only ceiling |
| **D** – Multimodal | Image + Tabular (fusion) | Primary hypothesis test |

## Quick Start

### Prerequisites
- Python ≥ 3.10 (3.12+ recommended)
- [uv](https://docs.astral.sh/uv/) package manager

### Setup
```bash
# Clone and enter project
git clone <repo-url> && cd crop-detection

# Create virtual environment and install dependencies
uv venv .venv
.venv/Scripts/activate   # Windows
uv pip install -r requirements.txt
```

### Download Data
Download the [Paddy Doctor](https://www.kaggle.com/competitions/paddy-disease-classification) dataset and extract:
- `train_images/` → `data/raw/train_images/`
- `train.csv` → `data/raw/train.csv`

### Run Pipeline
```bash
# Step 1: Prepare data splits
python src/data/prepare_data.py

# Step 2: Train all experiments (debug mode: 3 epochs)
python src/training/train_all.py --debug

# Step 2 alt: Full training (25 epochs)
python src/training/train_all.py

# Step 3: Launch API
uvicorn api.main:app --reload
```

### Run Tests
```bash
python -m pytest tests/ -v
```

## Project Structure

```
crop-detection/
├── configs/config.yaml          # Centralized configuration
├── src/
│   ├── data/                    # Dataset class, splitter, preparation
│   ├── models/                  # ResNet18, MLP, Multimodal fusion
│   ├── preprocessing/           # Image transforms, tabular pipeline
│   ├── training/                # Trainer engine, ablation orchestrator
│   ├── evaluation/              # Metrics, Grad-CAM XAI
│   └── utils/                   # Config, seeding, hardware detection
├── api/main.py                  # FastAPI inference endpoint
├── tests/                       # Pytest suite
├── docs/                        # Research documentation
│   ├── architecture.md
│   ├── data_pipeline.md
│   ├── training.md
│   ├── evaluation.md
│   ├── limitations.md           # ⚠ MANDATORY READING
│   └── api.md
├── data/raw/                    # User-provided dataset
├── models/checkpoints/          # Saved model weights
└── results/                     # Metrics, plots, error analysis
```

## Key Design Decisions

1. **Data leakage prevention**: `TabularPreprocessor` is fitted exclusively on the training split; val/test use frozen transforms.
2. **Reproducibility**: All random seeds are fixed globally (NumPy, PyTorch, Python `random`).
3. **Stratified splitting**: Class proportions are maintained across train/val/test.
4. **No temporal fabrication**: Environmental variables are agro-ecological baselines, not fabricated per-image sensor data. See [limitations.md](docs/limitations.md).

## ⚠ Limitations

This project has important scientific constraints documented in [`docs/limitations.md`](docs/limitations.md). Key points:
- No true longitudinal early prediction is possible with current public datasets
- Environmental variables are regional baselines, not per-leaf sensor readings
- Results on the development benchmark dataset are for pipeline verification only

## License

This project is for academic/research purposes.
