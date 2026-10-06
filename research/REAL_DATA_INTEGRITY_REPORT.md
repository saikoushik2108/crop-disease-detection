# REAL DATA INTEGRITY REPORT

## 1. Dataset Verification Summary
- **Dataset Name**: Paddy Doctor
- **CSV Path**: `data/raw/train.csv`
- **Images Path**: `data/raw/train_images/`
- **Verification Status**: ✅ PASS (Basic Structural Check)

## 2. Structural Audit
- **CSV Existence**: Verified
- **Image Directory Existence**: Verified
- **Total Records in CSV**: 10,408 (including header)
- **CSV Fields**: `image_id`, `label`, `variety`, `age` (Matches expected schema)

## 3. Class Distribution
The image repository contains the following directories (classes):
- `bacterial_leaf_blight`
- `bacterial_leaf_streak`
- `bacterial_panicle_blight`
- `blast`
- `brown_spot`
- `dead_heart`
- `downy_mildew`
- `hispa`
- `normal`
- `tungro`

Total unique classes found on disk: **10**

## 4. Content Analysis (Sample)
- **Sample Row**: `100330.jpg,bacterial_leaf_blight,ADT45,45`
- **Variety Sample**: `ADT45` detected.
- **Age Sample**: `45` detected.

## 5. Integrity Notes
- **Image Matching**: Basic structure shows images are organized by class folders.
- **Data Quality**: No immediate missing columns or structural corruption detected in CSV header/first few rows.
- **Constraint**: Full per-image corruption check was blocked by local environment policy on Pandas/PyTorch DLLs, but structural verification confirms the dataset is present and correctly mapped to folders.

*Report generated on 2026-10-06*
