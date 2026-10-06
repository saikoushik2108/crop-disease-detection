# RESEARCH PLAN: Multimodal Crop Disease Detection & Context-Aware Diagnosis

**Project Title:** Multimodal Deep Learning for In-Field Paddy Crop Disease Detection Integrating Visual and Agro-Environmental Context  
**Author:** AI Research Engineer & ML Developer  
**Student Profile:** B.E. (CSE - AIML) Final Year Capstone Project  
**Timeline:** Approximately 2 Months (8 Weeks)  
**Status:** Post-Software Verification & Data Audit Phase (Phase 2 Pre-Training Gate)  
**Date:** September 2026  

---

> [!CAUTION]
> **RESEARCH INTEGRITY & BENCHMARK CLARIFICATION:**
> The earlier 3-epoch benchmark metrics (Exp A Image-Only: 82.67%, Exp B Tabular-Only: 12.00%, Exp D Multimodal: 88.00%) were produced strictly on a 500-sample synthetic development benchmark for software pipeline verification. **They are NOT research findings and must not be reported as experimental results.**
> Real experimental results will only be produced on the authentic 10,407-sample Paddy Doctor Kaggle dataset following the Data Audit in `research/PADDY_DOCTOR_DATA_AUDIT.md`.

---

## 1. Final Research Question

> **Primary Research Question:**  
> *"Can multimodal intermediate feature fusion of in-field leaf imagery and genuine crop agronomic metadata (cultivar variety and developmental plant age) significantly improve disease classification accuracy and reduce confusion among visually ambiguous foliar pathologies compared with unimodal image-only deep learning models?"*

> **Sub-Question (Ablation & Context Utility):**  
> *"Does adding fixed regional agro-climatic context (calibrated for the Tirunelveli paddy basin) provide further diagnostic benefit, or is the performance gain driven primarily by the interaction between visual symptoms, cultivar resistance (variety), and phenological susceptibility (plant age)?"*

---

## 2. Formal Hypotheses

* **Null Hypothesis ($H_0$):**  
  The addition of agronomic metadata (crop variety, plant age) to deep convolutional visual features provides no statistically significant improvement in classification performance (measured by Macro F1-score and Class-Specific Recall on minority and ambiguous classes) compared to a standard unimodal image-only ResNet18 model ($p \ge 0.05$).
* **Alternative Hypothesis ($H_1$):**  
  Multimodal fusion of visual features with genuine agronomic metadata yields a statistically significant increase in Macro F1-score and significantly reduces misclassifications between symptomatically ambiguous sibling disease classes (specifically *Bacterial Leaf Blight* vs. *Bacterial Leaf Streak*, and *Blast* vs. *Brown Spot*).

---

## 3. Research Gap & Motivation

1. **The Laboratory Bias Gap:** Most published agricultural ML papers rely on PlantVillage, where leaves are photographed against uniform synthetic backgrounds. These models collapse when evaluated in real-world field conditions with soil, occlusions, sun glare, and pest damage.
2. **Visual Ambiguity at Critical Windows:** Plant pathogens belonging to the same genus or producing early chlorotic flecks appear visually indistinguishable in single-leaf RGB images. However, their occurrence is conditioned by cultivar genetics (`variety`) and developmental stage (`age`). Unimodal vision models cannot exploit this domain prior.
3. **Absence of Pragmatic Multimodal Benchmarks:** Existing multimodal studies either rely on private drone/IoT networks (inaccessible to ordinary researchers) or arbitrarily synthesize fake weather numbers on PlantVillage. There is a pressing need for a reproducible, scientifically defensible multimodal pipeline grounded in real field imagery and verified metadata.

---

## 4. Proposed Scientific Contribution

1. **Defensible Multimodal Benchmark:** A fully reproducible multimodal pipeline combining real field smartphone images from **Paddy Doctor** with verified crop variety and plant age.
2. **Systematic Ablation Study:** A structured empirical comparison across four distinct configurations:
   - *Experiment A:* Image-Only Baseline (ResNet18)
   - *Experiment B:* Agronomic Context-Only Baseline (Variety + Age MLP)
   - *Experiment C:* Multimodal Fusion (Image + Variety + Age)
   - *Experiment D:* Multimodal Fusion + Regional Agro-Climatic Context (Image + Variety + Age + Regional Context)
3. **Clinical Confusion Analysis:** Detailed error analysis targeting the resolution of clinically confusing disease pairs (*Blast* vs. *Brown Spot*; *Bacterial Leaf Blight* vs. *Bacterial Leaf Streak*).
4. **Interpretable Dual-Modality XAI:** Visual feature attribution via **Grad-CAM** coupled with tabular feature importance (Permutation analysis) to explain model decisions.
5. **Lightweight, Deployable Prototype:** Complete end-to-end system including FastAPI backend and a responsive user interface capable of inference on standard CPU / consumer laptop hardware.

---

## 5. Experimental Setup & Dataset Specification

### 5.1 Selected Crop
* **Crop:** Paddy / Rice (*Oryza sativa*)
* **Significance:** Crucial global food security crop, deeply impacted by foliar pathogens that cause up to 30–50% yield loss if untreated.

### 5.2 Selected Dataset
* **Dataset:** **Paddy Doctor: Paddy Disease Classification Dataset** (arXiv:2205.11108)
* **Image Count:** 10,407 labeled training images across 10 classes
* **Image Resolution:** Native $480 \times 640$ pixels; normalized and resized to $224 \times 224$ for standard CNN backbones.
* **Target Classes (10 classes):**
  1. `bacterial_leaf_blight` (Bacterial)
  2. `bacterial_leaf_streak` (Bacterial)
  3. `bacterial_panicle_blight` (Bacterial)
  4. `blast` (Fungal)
  5. `brown_spot` (Fungal)
  6. `dead_heart` (Stem borer pest damage)
  7. `downy_mildew` (Oomycete/Fungal)
  8. `hispa` (Beetle pest damage)
  9. `normal` (Healthy reference leaf)
  10. `tungro` (Viral disease transmitted by leafhopper)

### 5.3 Modality Specifications & Anti-Fabrication Constraints

1. **Modality 1: Visual Imagery (Field RGB):**
   - High-resolution smartphone images captured in real paddy fields under ambient natural lighting.
2. **Modality 2: Agronomic Metadata (Verified In-situ):**
   - **Crop Variety (Categorical):** Cultivar entity embedding (`ADT45`, `ASD16`, `Ponni`, `IR20`, `BPT5204`, etc.).
   - **Plant Age (Continuous):** Chronological age of the paddy crop in days (ranging from 30 to 90+ days), standardized via z-score normalization based strictly on training split statistics.
3. **Modality 3 (Ablation D only): Regional Agro-Climatic Context:**
   - **MANDATORY NOMENCLATURE:** Must be named `regional agro-climatic context`, NEVER "historical weather at image capture."
   - **NO RANDOM NOISE:** No artificial Gaussian perturbations or pseudo-sensor values.
   - Represents the broad agro-ecological climatic baseline for Tirunelveli district, Tamil Nadu.

### 5.4 Data Splitting & Leakage Prevention Strategy
* **Split Ratios:** 70% Training, 15% Validation, 15% Test.
* **Stratification:** Stratified by disease class (`stratify=df['label']`) to preserve exact class priors.
* **Data Leakage Safeguards:**
  - Disjoint sample IDs enforced with hard assertions: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
  - All scalers fit *exclusively* on the training set.
  - Image augmentations applied *only* during training.
  - Validation and test evaluations conducted on clean, unaugmented inputs.

---

## 6. Model Architectures & Fusion Pipeline

### 6.1 Unimodal Image Baseline (Experiment A)
* **Backbone:** ResNet18 pretrained on ImageNet.
* **Feature Extractor:** Global Average Pooling producing embedding vector $f_{img} \in \mathbb{R}^{512}$.
* **Classifier:** Dropout ($p=0.3$) $\rightarrow$ Linear($512 \rightarrow 10$) $\rightarrow$ Logits.

### 6.2 Unimodal Metadata Baseline (Experiment B)
* **Inputs:** Variety (categorical index) + Age (standardized continuous scalar).
* **Architecture:**
  - Variety index $\rightarrow$ Embedding Layer ($d_{embed} = 16$).
  - Age $\rightarrow$ Dense Layer ($d_{cont} = 16$) $\rightarrow$ LayerNorm $\rightarrow$ ReLU.
  - Concatenation ($16 + 16 = 32$) $\rightarrow$ Dense Layer ($32 \rightarrow 64$) $\rightarrow$ ReLU $\rightarrow$ Dropout ($0.2$) $\rightarrow$ Linear($64 \rightarrow 10$).

### 6.3 Multimodal Intermediate Feature Fusion (Experiment C & D)
```
                    [INPUT LEAF IMAGE (224x224x3)]
                                  |
                                  v
                    ResNet18 Backbone (Pretrained)
                                  |
                           Global Avg Pool
                                  |
                                  v
                    Visual Embedding: f_img (512-d)
                                  |
                                  +------------------------------------+
                                                                       |
  [CROP VARIETY] -> Embedding (16-d) --+                               |
                                       |---> Dense + LayerNorm (64-d)  |
  [PLANT AGE]    -> Dense (16-d) ------+             |                 |
                                                     v                 |
                                        Tabular Embedding: f_tab (64-d)|
                                                     |                 |
                                                     +-----------------+
                                                               |
                                                               v
                                                  CONCATENATION [512 + 64 = 576-d]
                                                               |
                                                               v
                                                  LayerNorm + Dropout (0.3)
                                                               |
                                                               v
                                                     Fusion Dense Layer (128-d)
                                                               |
                                                               v
                                                          ReLU + Dropout
                                                               |
                                                               v
                                                  Classification Head (10 Classes)
```

---

## 7. Evaluation Protocol & Metrics

To capture overall and per-class performance under class imbalance (5.2:1 ratio):
1. **Accuracy (Overall):** Correct predictions over total test samples.
2. **Macro-Averaged Precision, Recall, and F1-Score:** Treating every disease class with equal weight to prevent majority-class bias.
3. **Weighted F1-Score:** Accounting for sample distribution across classes.
4. **Normalized Confusion Matrix:** Quantitative analysis of cross-class confusion.
5. **Class-Specific Recall & Ambiguity Resolution:** Tracking whether multimodal context resolves the confusion between *Bacterial Leaf Blight* and *Bacterial Leaf Streak*.

---

## 8. Explainability & Interpretability Plan

1. **Visual Explainability (Grad-CAM):**
   - Extract activation gradients from the final convolutional layer of ResNet18 (`layer4`).
   - Generate Grad-CAM heatmaps overlaid onto the original RGB leaf image.
   - Verify that the model focuses on actual pathological lesions rather than background soil, water drops, or leaf tips.
2. **Tabular Feature Attribution:**
   - Permutation Feature Importance: Measure drop in Macro F1 when crop variety and plant age are shuffled.
   - Evaluate whether plant age acts as an epidemiological filter for stage-dependent diseases.

---

## 9. Statement on Temporal Sequence & Early Prediction

* **Scientific Truth:** There are no longitudinal, repeat-visit image sequences of the same plant available in public benchmarks.
* **Firm Position:** We do NOT implement artificial 7-day future prediction, as it would require fabricating temporal transitions without ground truth.
* **Ethical Standard:** Cross-sectional images can only be used for point-in-time diagnosis and stage-informed risk assessment.

---

## 10. Deliverables & Phased Milestone Plan

| Phase | Duration | Status | Key Milestones & Outputs |
| :--- | :--- | :---: | :--- |
| **Phase 1: Research & Feasibility** | Week 1 | **DONE** | Completed audit, literature review, dataset comparison, research plan, project repository structure. |
| **Phase 2: Data Audit & Pipeline Verification** | Week 2 | **DONE** | Software pipeline verified via debug benchmark; full data audit completed in `PADDY_DOCTOR_DATA_AUDIT.md`. |
| **Phase 3: Real Dataset Acquisition & Prep** | Week 3 | **NEXT** | Download real Paddy Doctor dataset (10,407 images); run clean splitting (70/15/15); inspect real class distributions. |
| **Phase 4: Unimodal Baselines** | Week 4 | PENDING | Train ResNet18 image-only baseline; train tabular metadata-only baseline on real data. |
| **Phase 5: Multimodal Architecture & Ablation** | Week 5 | PENDING | Train multimodal fusion on real data; complete Experiments A, B, C, D; generate real model comparison tables. |
| **Phase 6: Explainability (XAI)** | Week 6 | PENDING | Implement Grad-CAM heatmap generator on real validation samples; compute tabular feature importance. |
| **Phase 7: API & Web Application** | Week 7 | PENDING | Update FastAPI backend with real model checkpoint; test inference and error handling. |
| **Phase 8: Research Paper & Viva Support** | Week 8 | PENDING | Complete research paper draft (`paper/`), final presentation slides (`presentation/`), and viva voce defense guide. |
