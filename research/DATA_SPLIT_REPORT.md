# DATA SPLIT REPORT: Stratified Splitting & Leakage Prevention

**Project Title:** Multimodal Deep Learning for Paddy Disease Classification Using Leaf Images and Agronomic Metadata  
**Date of Audit:** September 2026  
**Auditor:** Lead ML Engineer & Research Implementation Agent  
**Partition Scheme:** 70% Training / 15% Validation / 15% Test  
**Random Seed:** `42` (Strict deterministic reproducibility)  
**Status:** **100% VERIFIED — ZERO DATA LEAKAGE**

---

## 1. Split Dimensions and Proportions

Splitting was executed directly on the authentic 10,407-sample metadata from `data/raw/train.csv` using stratified sampling on target disease labels (`stratify=df['label']`).

| Split | Number of Samples | Target Ratio | Actual Proportion | File Path |
|---|:---:|:---:|:---:|---|
| **Training Set** | 7,284 | 70.0% | **69.99%** | `data/processed/splits/train.csv` |
| **Validation Set** | 1,561 | 15.0% | **15.00%** | `data/processed/splits/val.csv` |
| **Test Set** | 1,562 | 15.0% | **15.01%** | `data/processed/splits/test.csv` |
| **Total** | **10,407** | **100.0%** | **100.00%** | — |

---

## 2. Data Leakage Verification (Disjoint Set Assertions)

Zero sample overlap exists across partitions. Programmatic set intersections on unique `image_id` strings yielded exactly zero common elements:

$$\text{ImageIDs}_{\text{Train}} \cap \text{ImageIDs}_{\text{Val}} = \emptyset \quad (\text{Count: } 0)$$
$$\text{ImageIDs}_{\text{Train}} \cap \text{ImageIDs}_{\text{Test}} = \emptyset \quad (\text{Count: } 0)$$
$$\text{ImageIDs}_{\text{Val}} \cap \text{ImageIDs}_{\text{Test}} = \emptyset \quad (\text{Count: } 0)$$

* Hard assertion checks in `src/data/splitter.py` passed with zero errors.

---

## 3. Stratification Fidelity Across Splits

Class proportions are preserved across all three splits, preventing bias while retaining natural prior probabilities:

| Disease Target Label | Total Count | Train Count (70%) | Train % | Val Count (15%) | Val % | Test Count (15%) | Test % |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `normal` | 1,764 | 1,235 | 16.95% | 264 | 16.91% | 265 | 16.97% |
| `blast` | 1,738 | 1,216 | 16.69% | 261 | 16.72% | 261 | 16.71% |
| `hispa` | 1,594 | 1,116 | 15.32% | 239 | 15.31% | 239 | 15.30% |
| `dead_heart` | 1,442 | 1,009 | 13.85% | 216 | 13.84% | 217 | 13.89% |
| `tungro` | 1,088 | 762 | 10.46% | 163 | 10.44% | 163 | 10.44% |
| `brown_spot` | 965 | 675 | 9.27% | 145 | 9.29% | 145 | 9.28% |
| `downy_mildew` | 620 | 434 | 5.96% | 93 | 5.96% | 93 | 5.95% |
| `bacterial_leaf_blight` | 479 | 335 | 4.60% | 72 | 4.61% | 72 | 4.61% |
| `bacterial_leaf_streak` | 380 | 266 | 3.65% | 57 | 3.65% | 57 | 3.65% |
| `bacterial_panicle_blight` | 337 | 236 | 3.24% | 51 | 3.27% | 50 | 3.20% |
| **Total** | **10,407** | **7,284** | **100.00%** | **1,561** | **100.00%** | **1,562** | **100.00%** |

---

## 4. Metadata Distribution Across Splits

### 4.1 Rice Variety Representation
All 10 rice cultivars are present in all three splits without category starvation:

| Variety Name | Train Count | Val Count | Test Count | Total |
|---|:---:|:---:|:---:|:---:|
| `ADT45` | 4,937 | 1,021 | 1,034 | 6,992 |
| `KarnatakaPonni` | 689 | 161 | 138 | 988 |
| `Ponni` | 449 | 99 | 109 | 657 |
| `AtchayaPonni` | 308 | 77 | 76 | 461 |
| `Zonal` | 270 | 54 | 75 | 399 |
| `AndraPonni` | 271 | 51 | 55 | 377 |
| `Onthanel` | 239 | 61 | 51 | 351 |
| `IR20` | 72 | 29 | 13 | 114 |
| `RR` | 26 | 3 | 7 | 36 |
| `Surya` | 23 | 5 | 4 | 32 |

### 4.2 Plant Age Statistics Across Splits
The statistical moments for plant age match across all splits:

| Split | Mean Age (days) | Standard Deviation | Min Age | Max Age |
|---|:---:|:---:|:---:|:---:|
| **Training Set** | **64.02** | 8.97 | 45 | 82 |
| **Validation Set** | **64.25** | 8.92 | 45 | 80 |
| **Test Set** | **63.95** | 8.93 | 45 | 80 |

* **Zero Leakage Preprocessing:** Mean ($\mu = 64.0205$) and standard deviation ($\sigma = 8.9734$) computed strictly on the training partition are stored in `data/processed/tabular_preprocessor.json`. Validation and test age features are normalized using these frozen parameters.

---

## 5. Class Imbalance Weighting (Training Partition Only)

To counter the 5.234 : 1 imbalance between majority class (`normal`) and minority class (`bacterial_panicle_blight`), class weights are derived using the standard inverse frequency formula strictly on the 7,284 training samples:

$$W_c = \frac{N_{\text{train}}}{K \times N_c}, \quad \tilde{W}_c = W_c \times \frac{K}{\sum_{k} W_k}$$

Where $N_{\text{train}} = 7,284$, $K = 10$ classes, and $N_c$ is the sample count for class $c$.

| Index | Target Class | Train Samples ($N_c$) | Raw Inverse Weight ($W_c$) | Normalized Weight ($\tilde{W}_c$) |
|:---:|---|:---:|:---:|:---:|
| 0 | `bacterial_leaf_blight` | 335 | 2.1743 | **1.5231** |
| 1 | `bacterial_leaf_streak` | 266 | 2.7383 | **1.9182** |
| 2 | `bacterial_panicle_blight` | 236 | 3.0864 | **2.1620** |
| 3 | `blast` | 1,216 | 0.5990 | **0.4196** |
| 4 | `brown_spot` | 675 | 1.0791 | **0.7559** |
| 5 | `dead_heart` | 1,009 | 0.7219 | **0.5057** |
| 6 | `downy_mildew` | 434 | 1.6783 | **1.1756** |
| 7 | `hispa` | 1,116 | 0.6527 | **0.4572** |
| 8 | `normal` | 1,235 | 0.5898 | **0.4131** |
| 9 | `tungro` | 762 | 0.9559 | **0.6696** |
| **Sum** | — | **7,284** | — | **10.0000** |

These weights are passed to `nn.CrossEntropyLoss(weight=class_weights)` during training to penalize errors on minority bacterial classes without artificial oversampling.
