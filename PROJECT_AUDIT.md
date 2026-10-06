# PROJECT_AUDIT.md

## 1. Current Architecture
- **Type**: Multimodal Deep Learning System for Paddy Disease Classification.
- **Core Model**: Image (ResNet-18) + Tabular (MLP for Variety and Age) $\rightarrow$ Concatenation $\rightarrow$ Fusion Head $\rightarrow$ Classification.
- **Backend**: FastAPI (`api/main.py`) - currently appears basic.
- **Frontend**: Single HTML file (`frontend/index.html`) - not yet a full React application.
- **Data Pipeline**: `src/data/`, `src/preprocessing/` for handling Paddy Doctor dataset.
- **Training**: Ablation-based approach (Image-only, Tabular-only, Multimodal).

## 2. Current Functionality
- **ML Pipeline**: Basic training scripts exist (`src/training/train_all.py`).
- **Model Storage**: Checkpoints saved in `models/checkpoints/`.
- **Evaluation**: Grad-CAM and metrics implementation exists in `src/evaluation/`.
- **API**: Basic FastAPI wrapper for inference.
- **Frontend**: Basic HTML prototype.

## 3. Incomplete Functionality
- **Frontend**: Missing a professional, responsive React application.
- **Backend**: Needs a more robust structure for multi-context (Location/Weather) handling.
- **Multi-Context Engine**: Runtime context (Weather/Location APIs) is not yet integrated into the inference flow.
- **Explainability**: Grad-CAM exists but needs integration into the final UI.
- **Documentation**: Basic README exists, but comprehensive GitHub-ready docs are missing.
- **Testing**: Basic tests exist, but end-to-end validation of the multi-context flow is missing.

## 4. Git Problems
- **Large Files Tracked**: Several `.pt` model checkpoints (>10MB) are tracked in `models/` and `models/checkpoints/`.
- **Virtual Environment**: `.venv` directory is present and potentially tracked (need to verify `git ls-files`).
- **Data Tracking**: `data/` directory contains raw/processed data which must be excluded.
- **Repository Pollution**: `results/` and `.pytest_cache` may be tracked.

## 5. Large Files & Directories
- **`.venv/`**: Extremely large (standard for Python envs).
- **`models/checkpoints/`**: Contains multiple `.pt` files (experiment results).
- **`models/`**: Contains production-ready `.pt` files.
- **`data/`**: Likely contains the Paddy Doctor images and CSV (huge).
- **`results/`**: Contains figures and potentially large logs.

## 6. Files/Folders to Ignore
- `.venv/`
- `__pycache__/`
- `data/raw/`
- `data/processed/`
- `models/checkpoints/` (unless specific production models are small)
- `.pytest_cache/`
- `*.log`
- `.env`

## 7. Files to Remain Tracked
- `src/` (All source code)
- `api/` (Backend code)
- `frontend/` (Once migrated to React)
- `configs/` (Configuration files)
- `tests/` (Test suite)
- `docs/` (Documentation)
- `requirements.txt`
- `.gitignore`
- `README.md`
- `models/model_config.json` and `variety_mapping.json` (small config files)

## 8. Files to be Regenerated Locally
- Virtual environment (`.venv`) via `requirements.txt`.
- Dataset (`data/raw/`) via Kaggle/Manual download.
- Model checkpoints (`models/checkpoints/`) via `src/training/train_all.py`.
- Processed data (`data/processed/`) via `src/data/prepare_data.py`.
- Results/Figures (`results/`) via evaluation scripts.
