# Data Pipeline & Leakage Prevention Strategy

## 1. Pipeline Stages

1. **Metadata Ingestion & Integrity Check:**
   - Validates existence of `image_id`, `label`, `variety`, and `age`.
   - Checks image file readability on disk.
2. **Agro-Environmental Context Generation:**
   - Enriches sample records with microclimate indicators based on Tirunelveli paddy basin meteorological ranges:
     - `temperature` (°C)
     - `humidity` (%)
     - `vpd` (Vapor Pressure Deficit in kPa calculated via Tetens equation)
     - `rain_risk` (0.0 to 1.0)
3. **Stratified Splitting:**
   - Strict 70% Train, 15% Validation, 15% Test partition.
   - Stratified by disease class to preserve exact distribution priors.
   - Disjoint image ID validation assertion prevents cross-split contamination.
4. **Preprocessor State Isolation:**
   - `TabularPreprocessor` fits categorical mappings (`variety_to_idx`) and continuous feature statistics (mean, std, median) strictly on the training partition.
   - Transformation of validation and test sets uses the frozen training statistics.
5. **Image Transforms:**
   - Training: Resize to $224 \times 224$, Random Horizontal Flip ($p=0.5$), Random Rotation ($\pm 15^\circ$), Color Jitter ($15\%$), and ImageNet Normalization.
   - Validation/Test/Inference: Deterministic Resize ($224 \times 224$) and ImageNet Normalization only.
