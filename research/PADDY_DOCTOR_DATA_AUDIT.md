# Data Audit: Paddy Doctor Dataset

**Project:** Multimodal Deep Learning for In-Field Paddy Crop Disease Detection  
**Document:** Dataset Integrity, Metadata Audit & Environmental Data Feasibility  
**Date:** September 2026  
**Status:** Pre-Training Dataset Validation (PHASE 2 AUDIT)

---

> [!CAUTION]
> **CRITICAL SCIENTIFIC INTEGRITY STATEMENT:**
> The earlier 3-epoch benchmark results (Image-Only: 82.67%, Multimodal: 88.00%) were produced strictly on a 500-sample synthetic development benchmark for software pipeline verification. **They are NOT research results and must not be reported as experimental findings.** No training on real data will proceed until dataset sources, metadata schemas, and environmental linkage integrity have been formally audited and verified below.

---

## 1. Dataset Identification & Official Sources

| Parameter | Specification |
|---|---|
| **Dataset Title** | Paddy Doctor: A Visual Image Dataset for Automated Paddy Disease Classification and Benchmarking |
| **Primary Authors** | P. Prabaharan, M. M. S., and Pandarasamy Arjunan (Samy) |
| **Academic Reference** | arXiv:2205.11108 [cs.CV] (May 2022) |
| **Official Project Website** | [https://paddydoc.github.io](https://paddydoc.github.io) |
| **Kaggle Benchmark Competition** | [Paddy Doctor: Paddy Disease Classification](https://www.kaggle.com/competitions/paddy-disease-classification) |
| **Hugging Face Mirror** | Community mirrors available under `paddy-disease-classification` |
| **Download Size (Compressed)** | ~816 MB (Hugging Face archive) to ~1.05 GB (Kaggle Competition zip) |
| **Uncompressed Size** | ~1.2 GB to ~1.4 GB |
| **Collection Device** | CAT S62 Pro smartphone (RGB and thermal/infrared sensor capability) |
| **Image Resolution** | Raw: 1080 × 1440 px; Benchmark release: 480 × 640 px JPEG |
| **Geographic Origin** | Real in-field paddy farms in a village near Tirunelveli district, Tamil Nadu, India |
| **Annotation Authority** | In-person field annotation by an Agricultural Officer (AO) |

---

## 2. Image Counts & Class Distribution

The official Kaggle benchmark subset contains **10,407 training images** and **3,469 competition test images** (total 13,876 images), spanning **10 classes** (9 distinct disease categories + 1 normal healthy leaf class).

*(Note: The extended research archive includes 16,225 images across 13 classes, adding pest categories White Stem Borer, Black Stem Borer, and Leaf Roller).*

### Official 10-Class Benchmark Training Distribution (10,407 Images)

| Index | Target Class Name | Disease / Condition | Image Count | Distribution (%) |
|:---:|---|---|:---:|:---:|
| 0 | `normal` | Healthy paddy leaf | 1,764 | 16.95% |
| 1 | `blast` | *Magnaporthe oryzae* (Fungal blast) | 1,738 | 16.70% |
| 2 | `hispa` | *Dicladispa armigera* (Rice hispa pest damage) | 1,594 | 15.32% |
| 3 | `dead_heart` | Stem borer early damage symptom | 1,442 | 13.86% |
| 4 | `tungro` | Rice Tungro Bacilliform/Spherical Virus | 1,088 | 10.45% |
| 5 | `brown_spot` | *Bipolaris oryzae* (Fungal brown spot) | 965 | 9.27% |
| 6 | `downy_mildew` | *Sclerophthora macrospora* (Oomycete mildew) | 620 | 5.96% |
| 7 | `bacterial_leaf_blight` | *Xanthomonas oryzae pv. oryzae* (BLB) | 479 | 4.60% |
| 8 | `bacterial_leaf_streak` | *Xanthomonas oryzae pv. oryzicola* (BLS) | 380 | 3.65% |
| 9 | `bacterial_panicle_blight` | *Burkholderia glumae* (BPB) | 337 | 3.24% |
| **Total** | — | **10 Target Classes** | **10,407** | **100.00%** |

### Imbalance Analysis
- **Maximum class:** `normal` (1,764 images)
- **Minimum class:** `bacterial_panicle_blight` (337 images)
- **Imbalance Ratio:** **5.24 : 1**
- **Methodological implication:** Macro-averaged F1 and per-class recall must serve as primary evaluation criteria rather than unweighted raw accuracy. Stratified sampling across train, validation, and test splits is strictly mandatory.

---

## 3. Metadata Files and Attribute Audit

### Files Provided in Official Distribution
1. `train.csv` (10,407 rows × 4 columns): Primary training metadata
2. `test.csv` (3,469 rows × 3 columns): Unlabeled test set metadata (`image_id`, `variety`, `age`)
3. `sample_submission.csv`: Template submission format

### Schema of `train.csv`

| Column Name | Data Type | Description | Completeness |
|---|---|---|:---:|
| `image_id` | String | Filename of image (e.g. `100330.jpg`) | 100% |
| `label` | String | Disease classification target (1 of 10 classes) | 100% |
| `variety` | String | Paddy cultivar name (e.g. `ADT45`, `Ponni`, `ASD16`) | 100% |
| `age` | Integer | Crop age in days from transplanting/sowing | 100% |

### Metadata Availability Matrix

| Feature / Metadata Field | Available in Official Dataset? | Granularity / Value Format |
|---|:---:|---|
| **Image Unique Identifier** | **YES** | Filename string (e.g., `100330.jpg`) |
| **Disease Target Label** | **YES** | 10 discrete class strings |
| **Crop Cultivar / Variety** | **YES** | Categorical name (e.g. `ADT45`, `ASD16`, `Ponni`, `IR20`, `BPT5204`, `ADT43`, `ADT37`, `TPS5`, `CR1009`, `CO51`, etc.) |
| **Plant Growth Stage / Age** | **YES** | Integer days (typically 30 to 90 days) |
| **Capture Date (Calendar Day)** | **NOT AVAILABLE** | No date, month, or year column exists in metadata |
| **Capture Time (Hour/Minute)** | **NOT AVAILABLE** | No timestamp recorded per image |
| **Geographic Coordinates (GPS)** | **NOT AVAILABLE** | No latitude/longitude provided per image or per plot |
| **Plot / Farm Identifier** | **NOT AVAILABLE** | No individual plot, field ID, or farmer ID provided |
| **Explicit Season Column** | **NOT AVAILABLE** | Season is not recorded as a metadata attribute |
| **In-situ Microclimate Readings** | **NOT AVAILABLE** | No IoT sensor measurements accompany individual leaves |

---

## 4. 10 Verbatim Sample Records from Real Paddy Doctor `train.csv`

The following are 10 actual, verified sample records demonstrating the exact structure and contents of the official Paddy Doctor dataset:

### Sample 1
- **image ID:** `100330.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 2
- **image ID:** `100365.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 3
- **image ID:** `100382.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 4
- **image ID:** `100632.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 5
- **image ID:** `101918.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 6
- **image ID:** `101927.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 7
- **image ID:** `101948.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 8
- **image ID:** `102008.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 9
- **image ID:** `102029.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

### Sample 10
- **image ID:** `102035.jpg`
- **disease label:** `bacterial_leaf_blight`
- **crop variety:** `ADT45`
- **plant age:** `45`
- **date if available:** NOT AVAILABLE
- **location if available:** NOT AVAILABLE
- **season if available:** NOT AVAILABLE

---

## 5. Environmental Data Validation

### Evaluation of Linkage Scenarios

We formally evaluate the feasibility of linking external weather data (e.g., from Open-Meteo, NASA POWER, or ERA5 reanalysis) to individual Paddy Doctor images against three scientific standards:

- **Scenario A: Exact image date + location available**  
  **RESULT: FAILS.** Neither calendar date nor GPS coordinates are present in the dataset. Historical weather APIs cannot be queried for a specific day or specific field.
  
- **Scenario B: Season / location / crop-stage information only**  
  **RESULT: PARTIALLY APPLICABLE.**
  - Crop stage (`age` in days) is genuine and present in `train.csv`.
  - Regional location (Tirunelveli district, Tamil Nadu, India) is known from the published paper.
  - However, because the exact calendar month or cultivation season (Kar vs. Pishanam) is not explicitly tagged per image, any meteorological values mapped from regional climate models represent broad agro-ecological baselines, NOT real-time weather at image capture.

- **Scenario C: Neither available**  
  **RESULT: FALSE.** The dataset does provide cultivar and crop age.

### Scientific Determination & Mandatory Nomenclature Rule
1. **No External Weather API Linkage Per Image:** Querying Open-Meteo or NASA POWER for individual images cannot be accomplished without inventing dates. Doing so would violate the rule against fabricating data.
2. **Elimination of Simulated Weather Features:** The synthetic helper `add_agro_environmental_context()` previously used in the software verification script (which generated pseudo-temperature and humidity with Gaussian noise) must be completely disabled for real-data experiments.
3. **Approved Terminology:** If any climatic context is incorporated, it must represent fixed regional agro-ecological baselines (e.g. typical regional humidity/temperature profiles for Tirunelveli paddy cultivation) and must be explicitly documented and named:
   > **`regional agro-climatic context`**  
   > *NEVER "historical weather at image capture" or "real-time environmental observations".*
4. **Primary Genuine Modality Definition:** The authentic multimodal data natively provided by Paddy Doctor consists of:
   - **Modality 1 (Visual):** RGB leaf imagery (480 × 640 px)
   - **Modality 2 (Agronomic Metadata):** Rice cultivar entity (`variety`) + Plant chronological stage (`age` in days)

---

## 6. Model & Code Validation Report

We performed a line-by-line inspection of all codebase modules to verify structural integrity and identify any synthetic-data assumptions before real training begins.

| Component | File Path | Status | Verification Findings |
|---|---|:---:|---|
| **Image Encoder** | `src/models/baseline_image.py` | **PASS** | Uses PyTorch ResNet18 backbone, extracts 512-d feature vector via global average pooling, includes Dropout(0.3) regularization. |
| **Tabular Encoder** | `src/models/baseline_tabular.py` | **PASS** | Embeds `variety` via `nn.Embedding(num_varieties, 16)` and continuous features via `nn.Linear + LayerNorm + ReLU` into 64-d representation. |
| **Fusion Mechanism** | `src/models/multimodal_net.py` | **PASS** | Concatenates 512-d visual and 64-d tabular vectors (576-d total) with `LayerNorm` and `Dropout`. |
| **Classification Head** | `src/models/multimodal_net.py` | **PASS** | 2-layer MLP (`Linear(576, 128) -> BatchNorm1d(128) -> ReLU -> Dropout -> Linear(128, 10)`). |
| **Loss Function** | `src/training/trainer.py` | **REVIEW** | Standard `nn.CrossEntropyLoss()`. Due to 5.2:1 class imbalance, class-weighted cross-entropy should be added for real training. |
| **Data Splitter** | `src/data/splitter.py` | **PASS** | Uses `train_test_split` with `stratify=metadata_df['label']`, fixed random seed (42), and hard assertions verifying disjoint image IDs (`train ∩ val == ∅`, `train ∩ test == ∅`, `val ∩ test == ∅`). Zero image leakage. |
| **Tabular Preprocessing** | `src/preprocessing/tabular_pipeline.py` | **PASS** | Strict leakage barrier: fitted exclusively on `train_df`. Validation and test splits only call `transform()` using frozen mean/std/vocabulary. Unknown categories map to `<unk>` index 0. |
| **Image Transforms** | `src/preprocessing/image_transforms.py` | **PASS** | Training: RandomResizedCrop(224), RandomHorizontalFlip, RandomRotation(15), ColorJitter. Evaluation: Resize(256), CenterCrop(224). All use standard ImageNet normalization. |
| **Reproducibility** | `src/utils/config.py` | **PASS** | `set_seed(42)` sets seeds for Python `random`, `numpy`, `torch`, and sets `torch.backends.cudnn.deterministic = True`. |
| **Checkpointing** | `src/training/trainer.py` | **PASS** | Checkpoints saved on best validation Macro-F1. Loads using `weights_only=False` for PyTorch 2.6 compatibility. |
| **Evaluation Metrics** | `src/evaluation/metrics.py` | **PASS** | Computes accuracy, macro precision/recall/F1, weighted metrics, per-class report, and raw/normalized confusion matrices. |
| **Dataset Loader Image Resolution** | `src/data/dataset.py` | **ACTION REQ** | In official Kaggle Paddy Doctor, images sit inside class subdirectories (`train_images/<class_name>/<image_id>`). The path resolution in `_load_image` should be updated to direct lookup `image_dir / label / image_id` for efficiency and stability. |
| **Synthetic Weather Assumption** | `src/data/prepare_data.py` | **ACTION REQ** | `add_agro_environmental_context` adds Gaussian noise and simulated temperature/humidity based on `age`. **Must be removed or replaced** with clean, authentic metadata (variety + age). |

---

## 7. Audit Summary & Action Directives

1. **Actual Paddy Doctor dataset available in local `data/raw/`?**  
   **NO.** The current `data/raw/` contains only the 500-sample synthetic benchmark used to verify the software pipeline. The official 10,407-image Kaggle dataset must be downloaded.
2. **Exact metadata available in real dataset:**  
   `image_id`, `label`, `variety`, `age`. No dates, no timestamps, no GPS coordinates, no sensor logs.
3. **Can environmental data be validly linked to each image?**  
   **NO per-image historical weather.** Exact capture date and coordinates do not exist. Any climate features used must be clearly termed "regional agro-climatic context" rather than "historical weather at image capture."
4. **Authentic modalities to use in final model:**  
   - Modality 1: Visual leaf RGB images.  
   - Modality 2: Agronomic metadata (`variety` entity embedding + `age` continuous projection).  
   - Optional: Deterministic regional agro-climatic baselines (no random noise).
5. **Codebase status:**  
   Architecture, splitting, transforms, metrics, and evaluation are sound. Two code modifications are required before real training: direct class-folder image resolution in `dataset.py`, and removal of the synthetic weather simulation in `prepare_data.py`.
