# Model Management

This directory contains the trained weights and configuration files for the Paddy Disease Classification system.

## Model Files
- `multimodal_best.pt`: The primary trained model utilizing visual and tabular fusion.
- `image_only_best.pt`: The unimodal ResNet-18 baseline.
- `metadata_only_best.pt`: The tabular MLP baseline.
- `tabular_preprocessor.json`: Scaling and mapping parameters for variety and age.
- `class_mapping.json`: Mapping of numerical indices to disease labels.

## How to Obtain Models
Since model weights are large, they are excluded from the Git repository via `.gitignore`.

1. **Download**: Weights can be downloaded from the project's designated storage (e.g., Zenodo, Google Drive, or HuggingFace).
2. **Placement**: Place the `.pt` and `.json` files directly in the `models/` folder.
3. **Verification**: Ensure `multimodal_best.pt` is present for the API to start.

## Retraining the Model
To retrain the models from scratch:
1. Ensure the dataset is present in `data/raw/`.
2. Run the training orchestrator:
   ```bash
   python src/training/train_all.py
   ```
3. The best checkpoints will be automatically saved to `models/` and `models/checkpoints/`.
