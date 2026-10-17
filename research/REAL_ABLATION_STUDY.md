# Empirical Research Findings: Authentic Ablation Study

**Project Title:** Multimodal Deep Learning for Paddy Disease Classification Using Leaf Images and Agronomic Metadata  
**Dataset:** Authentic In-Situ Paddy Doctor Dataset (10,407 Images, Tirunelveli District, Tamil Nadu, India)  
**Evaluation Scheme:** Stratified Holdout Test Partition (1,561 unseen samples, 15.00% of dataset, zero data leakage)  
**Random Seed:** 42 (Deterministic Reproducibility)  
**Status:** Completed Full Evaluation on Authentic Ground Truth  

---

## 1. Research Question & Empirical Hypothesis

> **Primary Research Question:**  
> Does conditioning visual disease representations on authentic agronomic context (cultivar variety + chronological plant age) via intermediate multimodal feature fusion improve classification accuracy, minority class recall, and diagnostic reliability compared to visual-only and metadata-only baselines?

**Empirical Hypothesis:**  
In-field agricultural pathology diagnoses frequently suffer from visual ambiguity (e.g., overlapping foliar lesion morphology between blast and brown spot, or streak vs blight). Conditioning image representations on cultivar variety (which governs inherent genetic disease resistance) and chronological crop age (which governs phenological susceptibility windows) will resolve morphological ambiguities and yield superior diagnostic accuracy compared to unimodal visual baselines.

---

## 2. Experimental Ablation Comparison Table

Evaluation metrics computed on the disjoint, zero-leakage test set ($N_{\text{test}} = 1,561$ samples):

| Experiment | Architecture & Inputs | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Exp B: Context-Only Baseline** | TabularContextMLP (Cultivar Embedding + Standardized Plant Age) | 36.68% | 0.3741 | 0.3458 | 0.3240 | 0.3612 |
| **Exp A: Visual-Only Baseline** | ResNet18 Transfer Learning (RGB Leaf Imagery) | 73.11% | 0.7235 | 0.7355 | 0.7070 | 0.7385 |
| **Exp C: Multimodal Fusion (Proposed)** | MultimodalCropNet (Visual Encoder + Variety Embedding + Age Projection) | **85.47%** | **0.8259** | **0.8703** | **0.8399** | **0.8561** |
| **Net Multimodal Gain ($\Delta$)** | **Multimodal vs Visual Baseline** | **+12.36%** | **+0.1024** | **+0.1348** | **+0.1329** | **+0.1176** |

---

## 3. Class-by-Class Comparative Breakdown (Test Set, $N=1,561$)

| Class Name | Test Support | Metadata F1 | Image F1 | Multimodal F1 | $\Delta$ F1 (Multi vs Image) | Pathology & Phenology Rationale |
|---|:---:|:---:|:---:|:---:|:---:|---|
| `bacterial_panicle_blight` | 50 | 0.2308 | 0.6591 | **0.9074** | **+0.2483 (+24.83%)** | *Burkholderia glumae* attacks emerging/flowering panicles (age 65–85d). Context acts as a decisive stage-gated discriminator. |
| `bacterial_leaf_streak` | 57 | 0.4684 | 0.5943 | **0.8029** | **+0.2086 (+20.86%)** | *X. oryzicola* interveinal streaks visually overlap with early blight; cultivar susceptibility priors resolve visual ambiguity. |
| `downy_mildew` | 93 | 0.3741 | 0.5625 | **0.7387** | **+0.1762 (+17.62%)** | Foliar downy fungal growth is subtle in early stages; seedling stage correlation strongly aids detection. |
| `blast` | 261 | 0.4877 | 0.7034 | **0.8454** | **+0.1420 (+14.20%)** | Spindle-shaped lesions (*Magnaporthe oryzae*) often confused with brown spot; age + variety context boosts precision to 91.5%. |
| `hispa` | 239 | 0.5027 | 0.7143 | **0.8535** | **+0.1392 (+13.92%)** | Epidermal scraping streaks caused by insect feeding; vegetative stage correlation improves F1 significantly. |
| `brown_spot` | 145 | 0.0278 | 0.6926 | **0.8062** | **+0.1136 (+11.36%)** | Nutrient deficiency associated necrotic spots; precision elevated from 65.2% to 92.0% with multimodal fusion. |
| `tungro` | 163 | 0.0495 | 0.6713 | **0.7649** | **+0.0936 (+9.36%)** | Viral leaf yellowing/stunting transmitted by leafhoppers; variety resistance genetics guide classification. |
| `normal` (Healthy) | 265 | 0.5477 | 0.8439 | **0.9370** | **+0.0931 (+9.31%)** | Healthy leaves achieve 95.5% recall and 92.0% precision; visual + metadata eliminates false positive disease alarms. |
| `bacterial_leaf_blight` | 72 | 0.2283 | 0.7368 | **0.7952** | **+0.0584 (+5.84%)** | Marginal necrosis (*Xanthomonas oryzae*); recall elevated from 68.1% to 91.7%. |
| `dead_heart` | 217 | 0.3227 | 0.8920 | **0.9474** | **+0.0554 (+5.54%)** | Stem borer (*Scirpophaga incertulas*) central whorl death; precision reaches 98.5%. |

**Key Result:** In all 10 target classes, multimodal fusion strictly outperforms the image-only baseline, proving that agronomic context provides complementary discriminative signal across the full pathology spectrum.

---

## 4. Key Scientific Insights & Discussion

### 4.1 Agronomic Context Disambiguates Stage-Specific Pathologies
The most dramatic performance elevation occurred on **`bacterial_panicle_blight` (BPB)**:
- BPB is the rarest minority class in the benchmark dataset (only 3.24% of training samples, 50 test samples).
- Visual-only ResNet18 achieved **0.6591 F1** (Recall: 58.00%, Precision: 76.32%).
- Multimodal Fusion elevated BPB to **0.9074 F1** (Recall: **98.00%**, Precision: **84.50%**), an absolute improvement of **+24.83 percentage points in F1** and **+40.00 percentage points in recall**.
- **Agronomic explanation:** Unlike foliar blast or brown spot which can strike at multiple growth stages, panicle blight pathogen *Burkholderia glumae* specifically colonizes panicles during heading and grain filling ($65 \le \text{age} \le 85$ days). Conditioning the visual representation on plant age effectively eliminates false-positive misclassifications from vegetative growth stages.

### 4.2 Cultivar Priors Resolve Visual Confusion Between Blast and Brown Spot
In visual-only classification, *Magnaporthe oryzae* (blast) and *Bipolaris oryzae* (brown spot) frequently cross-contaminate predictions due to similar elliptical brown necrotic lesions.
- In the multimodal model, precision on `brown_spot` surged from **65.24% to 92.00%**, and `blast` precision rose from **87.93% to 91.50%**.
- This demonstrates that cultivar susceptibility profiles (e.g., ADT45 vs CR1009 resistance differentials) act as strong prior constraints that steer feature representations away from confusing visual distractors.

### 4.3 Training Dynamics: The Warm-Start & Differential Learning Rate Mechanism
Early attempts at joint end-to-end multimodal training from scratch produced sub-optimal results (65.94% accuracy) because random gradients from randomly initialized tabular projection heads disrupted pretrained ResNet convolutional filters.
The proposed training strategy solved this via:
1. **Visual Backbone Warm-Start:** Initializing the visual encoder with weights pretrained on the domain-specific visual task (`image_only_best.pt`).
2. **Backbone Freezing:** Freezing ResNet18 convolutional layers for the first 5 epochs, allowing the tabular embedding and multimodal fusion MLP to adapt without perturbing visual representations.
3. **Differential Learning Rates:** When unfreezing at epoch 6, applying $\eta_{\text{backbone}} = 10^{-5}$ alongside $\eta_{\text{fusion}} = 3 \times 10^{-4}$, enabling subtle domain fine-tuning while preventing catastrophic forgetting.
4. **Convergence:** The model achieved peak validation Macro F1 of **0.8451** and validation accuracy of **86.23%**, translating to **85.47% test accuracy** and **0.8399 Macro F1** on unseen test data.

---

## 5. Interpretability: Grad-CAM Saliency Analysis

Using Grad-CAM (*Selvaraju et al., 2017*) on the final convolutional layer (`layer4`) of the ResNet18 visual encoder:
1. **Target Attention:** Saliency maps confirm that gradient activations concentrate tightly on necrotic lesions, interveinal discoloration, and foliar damage boundaries rather than background clutter (soil, water, field implements, or farmer fingers).
2. **Context-Visual Alignment:** When conditioned on variety and crop age, the network maintains sharp, focused visual attention while tabular embeddings adjust classification logits to align with epidemiological likelihood.

---

## 6. Reproducibility & Artifact Verification

All empirical checkpoints and serialized artifacts are fully reproducible:
- **Test Set Split:** `data/processed/splits/test.csv` (1,561 samples)
- **Multimodal Checkpoint:** `models/multimodal_best.pt` & `models/checkpoints/multimodal_best.pt`
- **Evaluation Script:** `src/evaluation/evaluate_saved_checkpoint.py`
- **Metrics Summary:** `results/metrics/real_ablation_comparison.csv`
- **Per-Class Metrics:** `results/metrics/real_per_class_ablation.csv`
- **Figures:** `results/figures/real_per_class_f1_comparison.png`, `results/figures/real_gradcam_explanations.png`
