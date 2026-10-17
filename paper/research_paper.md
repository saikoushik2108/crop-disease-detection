# Multimodal Deep Learning for Paddy Disease Classification Using Leaf Imagery and Agronomic Metadata

**Author:** Undergraduate Researcher, Department of Computer Science & Engineering (AIML)  
**Affiliation:** Academic Research Project  
**Target Venue:** IEEE / Springer Conference on Artificial Intelligence and Smart Agriculture  
**Code & Checkpoints:** Fully Reproducible In-Situ System  

---

## Abstract

Paddy (*Oryza sativa*) is a primary staple food supporting over half of the global population, yet foliar diseases cause catastrophic annual yield losses exceeding 20–40% in vulnerable agrarian regions. Deep convolutional neural networks (CNNs) have shown promise for automated foliar lesion diagnosis; however, existing vision-only approaches frequently falter in real-world agricultural conditions due to visual mimicry between pathologies, lighting variations, and overlapping symptomatic lesions. Crucially, plant pathologists in the field do not diagnose disease from visual symptoms alone—they leverage agronomic context, including host cultivar resistance genetics and chronological plant age. In this work, we present a **Multi-Context Aware Deep Multimodal Learning Architecture** for paddy disease classification that fuses in-situ RGB leaf imagery with agronomic metadata (cultivar variety and plant age in days). Using the authentic **Paddy Doctor** benchmark dataset comprising 10,407 field-captured images across 10 disease categories collected in Tirunelveli, Tamil Nadu, India, we establish a leak-free, stratified experimental protocol (70% train, 15% validation, 15% test). We propose a dual-stream architecture combining a ResNet-18 visual feature extractor with an agronomic tabular embedding network via intermediate feature fusion. To overcome gradient interference between disparate modalities, we devise a **warm-start differential learning rate training strategy** with gradual backbone unfreezing. Extensive empirical ablation demonstrates that our multimodal model achieves a **Test Accuracy of 85.47%** and a **Macro F1-score of 0.8399** on 1,561 unseen test samples, significantly outperforming both the metadata-only baseline (36.68% accuracy, 0.3240 Macro F1) and the visual-only ResNet-18 baseline (73.11% accuracy, 0.7070 Macro F1). Notably, multimodal conditioning yields a **+24.83% F1 gain** on the critical minority class *Bacterial Panicle Blight* by exploiting stage-gated phenological constraints. Finally, we provide visual interpretability via Grad-CAM and deploy the validated model into an asynchronous FastAPI and React web system for field diagnostics.

**Keywords:** Multimodal Deep Learning, Agricultural Computer Vision, Paddy Disease Classification, Intermediate Feature Fusion, Agronomic Metadata, Explainable AI (Grad-CAM), Paddy Doctor.

---

## 1. Introduction

Paddy rice is central to global food security. In developing agrarian economies, smallholder farmers lose millions of tons of rice annually to fungal, bacterial, viral, and insect-vectored pathologies. Timely and precise diagnosis is essential to prescribe targeted agricultural interventions (e.g., targeted bactericides, balanced nutrient amendments, biological biocontrols) and prevent indiscriminate agrochemical overapplication that damages soil ecosystems.

While automated plant disease detection using deep learning has expanded rapidly, the vast majority of literature relies exclusively on computer vision applied to laboratory-curated datasets (e.g., PlantVillage). These models perform poorly when deployed in authentic agricultural environments due to several structural challenges:
1. **Visual Mimicry (Equifinality):** Different pathogens manifest overlapping foliar necrosis. For instance, *Bipolaris oryzae* (brown spot) and early-stage *Magnaporthe oryzae* (blast) produce visually similar elliptical necrotic lesions.
2. **Phenological Dependency:** Certain devastating pathologies are strictly restricted to specific growth phases. *Burkholderia glumae* (bacterial panicle blight) infects only emerging panicles during heading and grain filling ($65 \le \text{age} \le 85$ days), whereas vegetative seedlings are physiologically immune. Vision-only systems lack knowledge of crop phenology and misclassify early vegetative leaf marks as panicle blight.
3. **Genetic Host Susceptibility:** Different paddy varieties (e.g., ADT45, IR20, CR1009) have distinct genetic resistance profiles. An experienced agricultural extension officer conditions visual observations on cultivar identity.

### Contributions
To bridge the gap between pure computer vision and real-world agronomic diagnosis, this paper delivers:
- **A Multi-Context Intermediate Fusion Architecture (`MultimodalCropNet`):** Integrates 512-dimensional CNN visual representations with 64-dimensional tabular embeddings (cultivar entity embedding + normalized crop age projection).
- **A Leakage-Free Empirical Protocol on Real Data:** Evaluated rigorously on the real Paddy Doctor dataset (10,407 samples, 10 classes) under strict stratified holdout partitioning, avoiding data contamination and synthetic shortcuts.
- **Warm-Start Differential Optimization:** A phased training scheme that prevents catastrophic forgetting of visual representations while training tabular projection layers, unlocking a +12.36% accuracy and +13.29% Macro F1 improvement over strong visual baselines.
- **Empirical & Phenological Analysis:** Detailed ablation proving that agronomic context acts as a Bayesian prior that resolves visual equifinality across all 10 disease categories.
- **Explainability & Field Deployment:** Grad-CAM saliency verification confirming lesion localization, wrapped in a production-ready asynchronous REST API.

---

## 2. Related Work

### 2.1 Computer Vision in Rice Pathology
Convolutional Neural Networks—notably VGG, ResNet, MobileNet, and Vision Transformers—have been applied extensively to rice disease detection (Chen et al., 2020; Rahman et al., 2020). However, most benchmarks were trained on clean laboratory images with uniform black or white backgrounds. When applied to in-situ field conditions featuring complex backgrounds, direct sunlight, shadows, and multiple co-occurring stresses, visual accuracy drops precipitously.

### 2.2 The Paddy Doctor Benchmark
The Paddy Doctor dataset (P. et al., 2023) is an open-access, in-field collection of 10,407 paddy leaf images annotated with real ground truth collected from farmers' fields across Tirunelveli District, Tamil Nadu, India. Uniquely, Paddy Doctor records two critical agronomic attributes alongside each image: **rice variety** (cultivar) and **crop age (days)**. Previous Kaggle competition approaches largely discarded this metadata or trained isolated gradient-boosted trees on tabular features without deep intermediate representation fusion.

### 2.3 Multimodal Deep Learning in Agriculture
Multimodal learning combines heterogeneous data modalities (e.g., text, audio, images, sensors) into unified embedding spaces (Baltrušaitis et al., 2018). While multimodal fusion is standard in medical imaging (fusing radiology scans with clinical electronic health records), its application to crop pathology remains largely nascent. Prior agricultural attempts have been limited by small synthetic datasets, data leakage across image splits, or naive late-fusion averaging that fails to capture complex cross-modal interactions.

---

## 3. Methodology

```
+---------------------------------------------------------------------------------------+
|                                    INPUT MODALITIES                                   |
|   Leaf Image [3 x 224 x 224]       Variety Index [1]         Crop Age (Days) [1]      |
+---------------------+---------------------+---------------------------+---------------+
                      |                     |                           |
                      v                     v                           v
             +-----------------+   +------------------+        +-----------------+
             | ResNet-18 CNN   |   | Variety Embed    |        | Age Linear Proj |
             | (ImageNet Init) |   | (Dim: 11 -> 16)  |        | (Dim: 1 -> 16)  |
             +--------+--------+   +--------+---------+        +--------+--------+
                      |                     |                           |
                      |                     +-------------+-------------+
                      |                                   |
                      |                                   v
                      |                         +-------------------+
                      |                         | Tabular MLP Head  |
                      |                         | Linear(32 -> 64)  |
                      |                         | BatchNorm, ReLU   |
                      |                         +---------+---------+
                      |                                   |
                      v                                   v
             Visual Vector [512d]               Tabular Vector [64d]
                      \                                   /
                       \                                 /
                        v                               v
                     +-------------------------------------+
                     | Concatenation Layer (Fusion: 576d)  |
                     +------------------+------------------+
                                        |
                                        v
                     +-------------------------------------+
                     | Multimodal Fusion MLP               |
                     | Linear(576 -> 128) + BN + ReLU      |
                     | Dropout(p=0.3)                      |
                     | Linear(128 -> 10)                   |
                     +------------------+------------------+
                                        |
                                        v
                                Disease Logits (10)
```

### 3.1 Data Preparation and Zero-Leakage Splitting
The Paddy Doctor dataset comprises $N = 10,407$ samples distributed across 10 classes:
- Fungal: *Blast*, *Brown Spot*, *Downy Mildew*
- Bacterial: *Bacterial Leaf Blight*, *Bacterial Leaf Streak*, *Bacterial Panicle Blight*
- Insect damage: *Dead Heart* (stem borer), *Hispa* (leaf scraping beetle)
- Viral: *Tungro* (transmitted by *Nephotettix virescens*)
- Healthy: *Normal*

To prevent data contamination, we partition the dataset strictly into:
- **Training Set (70%):** 7,284 samples
- **Validation Set (15%):** 1,562 samples
- **Holdout Test Set (15%):** 1,561 samples

Stratification preserves class proportions across all three disjoint partitions. All continuous features (plant age) are standardized using parameters $\mu_{\text{train}} = 64.02$ and $\sigma_{\text{train}} = 8.97$ derived strictly from the training partition.

### 3.2 Visual Encoder Stream
The visual stream processes input images $\mathbf{x}_{\text{img}} \in \mathbb{R}^{3 \times 224 \times 224}$. We employ a ResNet-18 architecture pretrained on ImageNet:
$$\mathbf{z}_{\text{img}} = f_{\text{visual}}(\mathbf{x}_{\text{img}}; \theta_{\text{visual}}) \in \mathbb{R}^{512}$$
Adaptive average pooling collapses spatial dimensions to a 512-dimensional visual descriptor.

### 3.3 Agronomic Tabular Stream
The tabular stream processes:
1. **Cultivar Variety:** Discrete index $v \in \{0, \dots, V-1\}$ where $V=11$ varieties. Variety is projected via a learnable entity embedding matrix:
   $$\mathbf{e}_{\text{variety}} = \mathbf{W}_{\text{embed}}[v] \in \mathbb{R}^{16}$$
2. **Plant Age:** Standardized continuous value $a = \frac{\text{age} - \mu_{\text{age}}}{\sigma_{\text{age}}}$, projected through a linear dense layer:
   $$\mathbf{e}_{\text{age}} = \text{ReLU}(\mathbf{W}_{\text{age}} a + \mathbf{b}_{\text{age}}) \in \mathbb{R}^{16}$$

The tabular features are concatenated and transformed through a projection MLP:
$$\mathbf{z}_{\text{tab}} = \text{ReLU}(\text{BatchNorm}(\mathbf{W}_{\text{tab}} [\mathbf{e}_{\text{variety}} \,\|\, \mathbf{e}_{\text{age}}] + \mathbf{b}_{\text{tab}})) \in \mathbb{R}^{64}$$

### 3.4 Intermediate Multimodal Feature Fusion
Rather than late score averaging, intermediate representations are fused via concatenated feature projection:
$$\mathbf{z}_{\text{fusion}} = [\mathbf{z}_{\text{img}} \,\|\, \mathbf{z}_{\text{tab}}] \in \mathbb{R}^{576}$$
The fused vector passes through a non-linear fusion head with regularization:
$$\mathbf{h}_1 = \text{ReLU}(\text{BatchNorm}(\mathbf{W}_{f1} \mathbf{z}_{\text{fusion}} + \mathbf{b}_{f1})) \in \mathbb{R}^{128}$$
$$\hat{\mathbf{y}} = \mathbf{W}_{f2} (\text{Dropout}(\mathbf{h}_1, p=0.3)) + \mathbf{b}_{f2} \in \mathbb{R}^{10}$$

### 3.5 Warm-Start Differential Optimization
Simultaneous end-to-end training of CNN backbones and randomly initialized fusion layers often destabilizes visual representations. To resolve this, we implement a **two-phase optimization protocol**:
1. **Phase 1 (Epochs 1–5, Warm-Start & Backbone Freeze):** The visual encoder weights are initialized from the pretrained image-only baseline (`image_only_best.pt`). The ResNet backbone parameters $\theta_{\text{visual}}$ are frozen ($\nabla_{\theta_{\text{visual}}} = 0$). Only tabular layers, embedding weights, and fusion layers are trained with learning rate $\eta_{\text{fusion}} = 3 \times 10^{-4}$ using the AdamW optimizer ($weight\_decay = 10^{-4}$).
2. **Phase 2 (Epochs 6+, Gradual Unfreezing & Differential LR):** The entire visual backbone is unfrozen, but with a differential learning rate:
   $$\eta_{\text{backbone}} = 10^{-5}, \quad \eta_{\text{fusion}} = 3 \times 10^{-4}$$
   This allows fine-tuning of convolutional filters to multimodal feature representations without catastrophic forgetting.
3. **Loss Function:** We train using class-weighted Cross-Entropy loss to penalize minority class errors:
   $$\mathcal{L} = - \sum_{c=1}^{C} w_c y_c \log \frac{\exp(\hat{y}_c)}{\sum_{j=1}^{C} \exp(\hat{y}_j)}$$
   where $w_c = \frac{N_{\text{total}}}{C \cdot N_c}$ normalized to $\sum w_c = C$.

---

## 4. Experimental Results and Discussion

### 4.1 Global Ablation Comparison
We benchmark three controlled systems on the exact same 1,561 test samples:
- **Experiment B (Metadata-Only):** Tabular MLP on variety + plant age.
- **Experiment A (Visual-Only):** ResNet-18 on RGB leaf images.
- **Experiment C (Multimodal Fusion):** Proposed `MultimodalCropNet`.

```
========================================================================================
TABLE 1: Global Empirical Performance on Test Partition (N = 1,561 samples)
========================================================================================
Experiment             Modality             Accuracy    Macro P   Macro R   Macro F1  Weighted F1
----------------------------------------------------------------------------------------
Exp B: Context-Only    Variety + Age (MLP)   36.68%     0.3741    0.3458    0.3240    0.3612
Exp A: Visual-Only     Leaf Image (ResNet18) 73.11%     0.7235    0.7355    0.7070    0.7385
Exp C: Multimodal      Visual + Variety + Age 85.47%    0.8259    0.8703    0.8399    0.8561
----------------------------------------------------------------------------------------
Net Multimodal Gain    Exp C vs Exp A       +12.36%    +0.1024   +0.1348   +0.1329   +0.1176
========================================================================================
```

**Key Findings:**
1. **Metadata alone is insufficient for diagnosis (36.68% accuracy):** While significantly higher than random guess (10%), metadata provides prior epidemiological context rather than diagnostic certainty.
2. **Vision provides the primary diagnostic foundation (73.11% accuracy).**
3. **Multimodal fusion produces a substantial synergistic leap:** Combining vision with agronomic context increases test accuracy by **+12.36%** (to 85.47%) and Macro F1 by **+0.1329** (to 0.8399). Macro recall reaches **87.03%**, demonstrating robust sensitivity across all pathologies.

---

### 4.2 Class-by-Class Comparative Breakdown

```
========================================================================================
TABLE 2: Per-Class F1-Score Breakdown on Unseen Test Set (N = 1,561 samples)
========================================================================================
Disease Class            Support   Metadata F1   Image F1   Multimodal F1   Delta (C vs A)
----------------------------------------------------------------------------------------
bacterial_panicle_blight    50       0.2308       0.6591       0.9074        +0.2483 (+24.8%)
bacterial_leaf_streak       57       0.4684       0.5943       0.8029        +0.2086 (+20.9%)
downy_mildew                93       0.3741       0.5625       0.7387        +0.1762 (+17.6%)
blast                      261       0.4877       0.7034       0.8454        +0.1420 (+14.2%)
hispa                      239       0.5027       0.7143       0.8535        +0.1392 (+13.9%)
brown_spot                 145       0.0278       0.6926       0.8062        +0.1136 (+11.4%)
normal (Healthy)           265       0.5477       0.8439       0.9370        +0.0931 (+9.3%)
tungro                     163       0.0495       0.6713       0.7649        +0.0936 (+9.4%)
bacterial_leaf_blight       72       0.2283       0.7368       0.7952        +0.0584 (+5.8%)
dead_heart                 217       0.3227       0.8920       0.9474        +0.0554 (+5.5%)
========================================================================================
```

**Crucial Result:** Multimodal fusion outperforms the visual baseline across **every single class** (10 out of 10 categories).

---

### 4.3 Deep Diagnostic Case Studies

#### Case 1: Bacterial Panicle Blight (+24.83% F1 Gain)
*Burkholderia glumae* attacks panicles during the heading and grain maturation stages ($65 \le \text{age} \le 85$ days). In the visual-only model, early necrotic lesions on leaves caused false positives from vegetative stages, yielding only 58.00% recall. Multimodal conditioning on crop age acts as a stage-gated filter, elevating test recall to **98.00%** and precision to **84.50%** (F1 = 0.9074).

#### Case 2: Resolving Blast vs. Brown Spot Equifinality
Both fungal pathogens present brown, oval necrotic lesions. In the visual baseline, precision on `brown_spot` was limited to 65.24% due to false alarms triggered by atypical blast lesions. In the multimodal model, brown spot precision surged to **92.00%**, and blast F1 rose from 0.7034 to **0.8454**, demonstrating that cultivar genetics (variety-specific susceptibility profiles) effectively disambiguate morphologically similar foliar lesions.

---

## 5. Explainability Analysis via Grad-CAM

To ensure the multimodal network relies on genuine pathological indicators rather than image acquisition artifacts, we generated Class Activation Maps using Grad-CAM on `layer4` of the ResNet-18 visual encoder.

```
Visual Inspection of Heatmap Activations:
1. Dead Heart: Activations concentrate precisely along the dried, severed central spindle leaf where yellow stem borer larvae bore into the tiller base.
2. Bacterial Leaf Blight: Peak gradients align along wavy, water-soaked margins extending downward from the leaf tip.
3. Brown Spot & Blast: Focused activation islands track individual circular and diamond-shaped necrotic lesions rather than healthy leaf lamina or background soil.
4. Non-Artifact Attribution: The model maintains near-zero gradient activation across farmer fingers, background field water, and specular highlights.
```

---

## 6. System Architecture & Edge Deployment

The model is deployed as a production-grade automated advisory system:
- **FastAPI Asynchronous Engine:** Provides sub-100ms inference endpoints (`/predict`) accepting multipart form uploads (image file, variety name, crop age, optional GPS coordinates).
- **Dual Output Schema:** Delivers disease classification logits, top-3 differential confidence scores, disease severity assessment, and curated agronomic chemical/cultural treatment recommendations.
- **Embedded Grad-CAM Overlay:** Automatically returns base64-encoded visual heatmaps for real-time inspection by agricultural extension officers.
- **Interactive UI Dashboard:** A modern, mobile-responsive web application for field validation.

---

## 7. Threats to Validity & Limitations

1. **Regional Homogeneity:** The Paddy Doctor dataset was collected within Tirunelveli District, Tamil Nadu. While it captures authentic smallholder field conditions, performance across distinct agro-ecological zones (e.g., East Asian temperate rice ecosystems or sub-Saharan upland rice) requires cross-region domain adaptation validation.
2. **Missing Sensor Telemetry:** Current metadata relies on farmer-reported cultivar and age. Integrating ambient microclimate telemetry (temperature, relative humidity, leaf wetness duration) represents a promising next step to model disease epidemiology dynamically.

---

## 8. Conclusion

This study addresses the fundamental limitation of vision-only plant pathology systems by integrating authentic in-situ leaf imagery with agronomic metadata. By implementing a dual-stream architecture with intermediate feature fusion and a warm-start differential learning rate training schedule, our proposed `MultimodalCropNet` achieves **85.47% test accuracy** and **0.8399 Macro F1** on the 10-class Paddy Doctor benchmark, representing an absolute **+12.36% accuracy and +13.29% Macro F1 improvement** over the visual baseline. The findings validate the hypothesis that conditioning visual disease recognition on cultivar genetics and chronological crop age resolves visual equifinality and provides superior diagnostic reliability for smart agriculture.

---

## References

1. P., M. et al. (2023). "Paddy Doctor: A Visual Dataset for Paddy Disease Detection and Classification." *Scientific Data* / Kaggle Research Dataset.
2. Baltrušaitis, T., Ahuja, C., & Morency, L. P. (2018). "Multimodal Machine Learning: A Survey and Taxonomy." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 41(2), 423-443.
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770-778.
4. Selvaraju, R. R. et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization." *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618-626.
5. Rahman, C. R. et al. (2020). "A Two-Stage Deep Learning Approach for Rice Disease Detection Using Visual and Contextual Data." *Smart Agricultural Technology*, 100018.
6. Chen, J., Chen, J., Zhang, D., Sun, Y., & Nanehkaran, Y. A. (2020). "Using Deep Transfer Learning for Image-Based Plant Disease Identification." *Computers and Electronics in Agriculture*, 173, 105393.
