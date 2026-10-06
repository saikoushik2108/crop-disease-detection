# RESEARCH AND FEASIBILITY AUDIT REPORT

**Project Title:** Multimodal Deep Learning for Crop Disease Detection and Context-Aware Risk Estimation  
**Author:** AI Research Engineer & ML Developer  
**Student Profile:** Final-Year B.E. (CSE - AIML), Capstone Project (Approx. 2-Month Timeline)  
**Status:** Approved for Research Planning (Phase 1 Complete)  
**Date:** September 2026  

---

## 1. Introduction and Feasibility Objectives

The objective of this feasibility report is to rigorously audit the scientific viability, data availability, computational constraints, and methodology for integrating visual crop imagery with environmental and agronomic metadata. 

In accordance with academic standards, this audit enforces a strict **Zero-Fabrication Policy**:
- We do NOT fabricate missing weather or temporal sequences.
- We do NOT assign arbitrary random weather to laboratory images.
- We explicitly delineate between verified physical ground truth and simulated/benchmark distributions.
- We reject claims of "early future disease forecasting" unless supported by genuine longitudinal observational tracking.

---

## 2. Structured Investigation Across Audit Dimensions (A – K)

### A. Recent Literature from Approximately 2024–2026
Recent literature confirms a significant paradigm shift from unimodal computer vision models (e.g., standard CNNs, ResNets, ViTs) toward **multimodal deep learning frameworks**:
1. *AgriFusionNet (Frontiers in Fungal Biology, 2026, DOI: 10.3389/ffunb.2026.1779786):* Demonstrated that fusing visual features (EfficientNet) with environmental vectors (temperature, humidity, rainfall) and textual symptom descriptions achieves 98.94% accuracy, substantially reducing misclassifications in visually ambiguous disease classes.
2. *RustQNet (Computers and Electronics in Agriculture, 2024, DOI: 10.1016/j.compag.2024.109245):* Evaluated multimodal deep learning for stripe rust index inversion in winter wheat, integrating multispectral visual features with meteorological variables (air temperature, canopy temperature depression, humidity) via cross-attention, demonstrating an $R^2$ boost from 0.69 to 0.88.
3. *Bijlwan et al. (Scientific Reports, Nature Portfolio, 2025, DOI: 10.1038/s41598-025-18613-7):* Evaluated machine learning algorithms (ANN, Random Forest) over two consecutive field seasons (2023–2024) with varying sowing dates, proving that real-time meteorological variables (evapotranspiration, temperature, wind speed, relative humidity) predict fungal disease severity with $R^2 > 0.93$.
4. *Shamsuddin et al. (Genetics, 2024, DOI: 10.1093/genetics/iyae171):* Demonstrated multimodal integration of image, temporal weather time-series (30/60/90-day windows), and phenotypic metadata in maize, improving predictive error by 24% over image-only baselines.
5. *Kumar et al. (IEEE Access, 2024, DOI: 10.1109/ACCESS.2024.3358333):* Implemented an edge-cloud multimodal system combining smartphone imagery with local IoT weather sensors, boosting field accuracy from 87.2% to 95.8% under variable outdoor illumination.

### B. Existing Multimodal Crop Disease Systems
Existing systems in the literature fall into three broad categories:
1. **Visual + Textual Systems (Vision-Language Models):** Examples include *PlantWild / MVPDR* (2024) and *CDDM Bench* (2024), where leaf images are paired with botanical descriptions and QA pairs using CLIP or BERT.
2. **Visual + Agronomic Metadata Systems:** Examples include *Paddy Doctor Benchmarks* (2023–2024), where mobile field images are paired with cultivar variety and crop age.
3. **Visual + Environmental Sensor Systems:** Examples include *RustQNet* (2024), *AgriFusionNet* (2026), and *WeatherLeaf* (2026), where visual features are fused with scalar or temporal environmental measurements.

### C. Existing Image + Environmental Approaches
In peer-reviewed implementations, the fusion of image and environmental data is accomplished via:
- **Concatenation Fusion (Intermediate Fusion):** An image encoder (ResNet18 / EfficientNet-B0) extracts a 512-dimensional visual vector; an environmental Multi-Layer Perceptron (MLP) extracts a 32-to-64-dimensional environmental vector. The vectors are concatenated and passed through dense fusion layers.
- **Cross-Attention / Gated Fusion:** Environmental embeddings act as keys/values that modulate attention weights across visual feature maps.
- **Decision-Level Late Fusion:** Softmax probability vectors from separate image and weather classifiers are weighted and combined via ensemble rules.
*Scientific Finding for our 2-month scope:* Intermediate feature-level concatenation is the most stable, computationally lightweight, and mathematically interpretable architecture for transfer learning on standard hardware.

### D. Existing Temporal / Early Disease Prediction Approaches
Studies that implement true temporal sequence modeling (e.g., Shamsuddin et al., 2024; Bijlwan et al., 2025):
- Utilize daily meteorological records (temperature, humidity, precipitation, solar radiation) spanning 7 to 30 days prior to field inspection.
- Model the sequence using LSTM, GRU, or 1D Temporal Convolutional Networks (TCN).
- *Critical Finding:* Such studies were conducted on dedicated agricultural research station plots with on-site logging. None of the large open-access leaf image datasets (PlantVillage, PlantDoc, Paddy Doctor) provide continuous daily image sequences of the *same* plant over time.

### E. Candidate Crops
We investigated the following candidate crops:
1. **Paddy / Rice (*Oryza sativa*):** Staple food crop for >50% of the world's population; highly susceptible to devastating fungal and bacterial foliar diseases (Blast, Bacterial Leaf Blight, Brown Spot); rich public field dataset available (Paddy Doctor).
2. **Tomato (*Solanum lycopersicum*):** Frequent target in academic literature; heavily represented in PlantVillage; susceptible to early/late blight; however, public datasets are predominantly laboratory-based with artificial backgrounds.
3. **Wheat (*Triticum aestivum*):** Excellent for rust severity studies, but open-access datasets with paired imagery and tabular metadata are fragmented across specialized remote sensing portals.
4. **Maize (*Zea mays*):** High commercial value; large-scale trial data available through G2F, but requires complex multi-gigabyte agronomic pipelines beyond a 2-month desktop project.

*Decision:* **Paddy / Rice** is selected as the optimal candidate crop due to real-world global food security importance, high phenotypic diversity of foliar pathogens, and the availability of the verified field-condition **Paddy Doctor** dataset.

### F. Candidate Publicly Available Datasets
As detailed in `research/DATASET_COMPARISON.md`:
- **PlantVillage:** 54,306 images across 14 crops. *Rejected* as primary dataset due to artificial laboratory backgrounds and complete absence of geographic or temporal provenance.
- **PlantDoc:** 2,598 images across 13 plants. *Rejected* due to web-scraped noise and lack of metadata.
- **PlantWild:** 12,000+ images. *Rejected* for weather fusion as it focuses on text-vision CLIP architectures.
- **Paddy Doctor:** 10,407 labeled field training images across 10 classes, with explicit crop variety and age metadata collected in Tirunelveli, Tamil Nadu, India. **Selected as Primary Benchmark Dataset.**

### G. Environmental and Weather Datasets that Can Be Linked
For the selected Paddy Doctor dataset, the geographic domain is firmly grounded in the Tirunelveli agricultural basin, Tamil Nadu, India (~8.7139° N, 77.7567° E):
1. **Open-Meteo Historical Weather API:** Provides free, programmatic access to reanalysis weather data (ERA5-Land and national meteorological agencies) at hourly and daily resolutions without API keys. Key variables: 2m Temperature, Relative Humidity, Precipitation, Vapor Pressure Deficit (VPD), Wind Speed.
2. **NASA POWER Agroclimatology API:** Daily surface solar radiation, dew point temperature, precipitation, and relative humidity tailored for agricultural modeling.
3. **Regional Agro-Meteorological Profiles (TNAU / IMD):** Historical climate baselines for the two primary Tamil Nadu paddy seasons:
   - *Kar Season:* May/June to September (warm, moderate humidity, early monsoon showers: ~28–34°C, 65–75% RH).
   - *Pishanam Season:* October to January/February (northeast monsoon, cool, high humidity: ~22–29°C, 80–95% RH; peak season for Blast and Bacterial Leaf Blight).

### H. Whether Image + Environmental Data Can Actually Be Aligned
- **Direct Per-Image Physical Sensor Alignment:** In the public release of Paddy Doctor, individual calendar dates are not given on a per-image row basis; instead, each sample provides **Crop Variety** and **Plant Age (in days)**.
- **Scientifically Defensible Alignment Methodology:**
  1. *Biological / Agronomic Alignment:* Crop age (days after sowing) directly correlates with plant phenological stage (vegetative: 30–50 days; reproductive/tillering: 50–70 days; ripening/grain filling: 70–90+ days). Disease epidemiology is strictly governed by crop age (e.g., Rice Blast predominantly attacks seedlings and tillering stages; Bacterial Panicle Blight attacks during grain filling).
  2. *Agro-Ecological Environmental Alignment:* Using the known geographic coordinates (Tirunelveli) and the known seasonal crop cycle, we construct an **Agro-Ecological Environmental Context Vector** incorporating seasonal temperature, relative humidity, and pathogen-conducive microclimate risk indices.
  3. *Transparency:* We explicitly state in all documentation that environmental indicators reflect regional agro-climatic conditions and growth-stage microclimates rather than an individual micro-sensor attached to every single leaf.

### I. Whether Longitudinal Data Exists for Future Disease Prediction
- **Direct Scientific Audit Result:** **NO.** None of the publicly available crop disease image datasets track individual plants sequentially over time.
- **Explicit Declaration:** 
  > *"Early 7-day prediction cannot currently be scientifically established from this dataset or any other publicly available crop disease image benchmark."*
- **Alternative Formulation:** We formulate the primary objective as **Multimodal Context-Aware Disease Classification and Risk Estimation**, assessing whether fusing leaf images with crop variety, age, and environmental risk indicators resolves visual ambiguity between clinically confusing diseases.

### J. Potential Research Gaps Addressed by This Project
1. **Closing the Lab-to-Field Generalization Gap:** Moving away from sanitized PlantVillage images to train and evaluate on raw, uncurated in-field imagery from real smallholder paddy fields.
2. **Resolving Sibling Disease Ambiguity:** Evaluating whether multimodal signals help resolve notorious visual confusions in rice pathology:
   - *Bacterial Leaf Blight* vs. *Bacterial Leaf Streak* (both caused by *Xanthomonas*, exhibiting similar early yellow-orange lesion stripes).
   - *Blast* vs. *Brown Spot* (both fungal pathogens exhibiting elliptical necrotic spots with dark halos).
3. **Biological Covariate Integration:** Investigating the diagnostic value of crop age and cultivar genetics alongside deep CNN visual features.
4. **Explainable Multimodal Diagnostics:** Coupling Grad-CAM visual heatmaps with tabular feature importance to provide interpretable predictions for agricultural extension workers.

### K. Risks and Limitations

| Risk Factor | Probability | Impact | Mitigation Strategy |
| :--- | :--- | :--- | :--- |
| **Computational Hardware Bottleneck** | High (Student environment, potential lack of high-end GPU) | High | 1. Use lightweight, efficient backbones (ResNet18 pretrained on ImageNet).<br>2. Support mixed-precision / CPU fallback.<br>3. Provide configurable batch size (16/32) and small debug/subset modes.<br>4. Ensure code runs seamlessly on Google Colab if needed. |
| **Data Leakage Across Splits** | Medium | Critical | Avoid naive random splitting across identical plants. Utilize stratified splitting by disease class and, where possible, group-aware splitting based on crop variety and age distributions. Never compute normalization parameters on test sets. |
| **Modality Imbalance (Gradient Dominance)** | High | Medium | Visual embeddings (512-dim) can overpower tabular vectors (e.g., 16-dim). Use Layer Normalization, balanced projection layers, and a modest learning rate for the tabular MLP. |
| **Overclaiming Predictive Ability** | Medium | High | Maintain rigorous academic honesty: do not claim future forecasting without longitudinal labels; do not claim physical in-situ sensor data where regional agro-climatic priors are used. |
| **Time Constraint (2-Month Timeline)** | Medium | High | Strictly follow the phased development roadmap. Implement simple, solid baselines (ResNet18 + Tabular MLP) before attempting any complex extensions. |

---

## 3. Feasibility Conclusion & Next Step
The research project is **feasible, scientifically sound, and well-scoped for a 2-month university capstone**. 

The recommended setup is:
- **Crop:** Paddy / Rice (*Oryza sativa*)
- **Dataset:** Paddy Doctor (10,407 field training images across 10 disease/healthy classes)
- **Modalities:** (1) Mobile field RGB image, (2) Agronomic metadata (cultivar variety + plant age), (3) Regional agro-meteorological environmental features (temperature, relative humidity, VPD, rainfall risk).
- **Core Model:** ResNet18 visual encoder + Tabular MLP encoder + Intermediate Concatenation Fusion.
- **Explainability:** Grad-CAM for visual attribution; Feature permutation/importance for tabular attribution.
