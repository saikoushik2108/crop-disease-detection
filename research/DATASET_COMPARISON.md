# Systematic Dataset Comparison & Audit for Multimodal Crop Disease Research

**Author:** AI Research Engineer & ML Developer  
**Project:** Multimodal Deep Learning for Crop Disease Detection and Context-Aware Risk Estimation  
**Academic Level:** University B.E. (CSE - AIML) Capstone Research  
**Target Evaluation:** Realism, Scientific Validity, Multimodal Feasibility, Reproducibility  
**Date:** September 2026  

---

## 1. Candidate Datasets Overview

To build a scientifically defensible multimodal crop disease research system, we evaluated candidate datasets across computer vision quality, agronomic realism, metadata richness, microclimatic feasibility, and risk of data leakage.

The candidate datasets investigated are:
1. **PlantVillage** (Hughes & Salathé, 2015)
2. **PlantDoc** (Singh et al., CoDS-COMAD 2020)
3. **Paddy Doctor** (Arjunan et al., CoDS-COMAD 2023 / IEEE DataPort)
4. **PlantWild** (Wei et al., 2024 / Hugging Face & GitHub)
5. **NUST Wheat Rust Disease (NWRD)** (Zenodo / GitHub 2024)
6. **Smart Farming & Environmental Repositories** (Dryad / Zenodo / NASA POWER / Open-Meteo)

---

## 2. Comprehensive Multi-Criteria Comparison Matrix

| Attribute | PlantVillage | PlantDoc | Paddy Doctor | PlantWild | Wheat Rust (NWRD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Primary Crop(s)** | 14 species (Tomato, Potato, Apple, Corn, Grape, etc.) | 13 species (Apple, Bell Pepper, Potato, Tomato, etc.) | **Paddy / Rice (*Oryza sativa*)** | 100+ plant & crop species | Winter Wheat (*Triticum aestivum*) |
| **Number of Images** | 54,306 images | 2,598 images | **10,407 train + 3,469 test images** | ~12,000+ images | 3,200+ images |
| **Disease Classes** | 26 disease classes | 17 disease classes | **10 classes (9 diseases + normal)** | 100+ disease classes | Stripe rust severity stages |
| **Healthy Control Class** | Yes (12 healthy classes) | Yes (10 healthy classes) | **Yes (`normal` class)** | Yes (healthy leaf images) | Yes (healthy canopy) |
| **Image Setting** | **Laboratory (Detached leaves on grey/black paper)** | **Real Field (Web-scraped)** | **Real Field (In-situ smartphone captures in paddy fields)** | **Real Field (In-the-wild natural lighting & clutter)** | **Real Field (Handheld & near-canopy field plots)** |
| **Visual Realism & Complexity** | Very Low (Uniform background, no shadows, artificial lighting) | High (Multiple leaves, cluttered background, varied scale) | **High (Field illumination, occlusions, complex leaf orientations)** | Very High (Natural field clutter, diverse angles) | High (Field canopy, sun glare) |
| **Location Availability** | **None** (Collected in university greenhouses/labs) | **None** (Scraped from web search engines without origin) | **Known Region (Tirunelveli district, Tamil Nadu, India: 8.71° N, 77.75° E)** | **None / Sparse** (Web & heterogeneous sources) | Known (Experimental field stations in Shaanxi/Gansu) |
| **Capture Date Availability** | **None** | **None** | **Known Season/Year (2021–2022 Kharif/Rabi cycle; crop age in days)** | **None** | Known observation dates |
| **Crop Metadata Available** | None (only crop name and disease label) | Bounding box coordinates | **Yes: Cultivar/Variety (ADT37, ADT43, ADT45, ASD16, etc.) AND Plant Age (days: 30–90+)** | Text descriptions | Growth stage (jointing, heading, filling) |
| **Environmental Sensor Data in Dataset** | **None** | **None** | **None explicitly in CSV (Regional agro-weather linkable via Tirunelveli coords)** | **None** (textual only) | Available in custom station logs |
| **Can Real Weather Be Directly Linked?** | **NO** (No dates, no locations. Linking weather requires arbitrary synthetic simulation) | **NO** (Web-scraped, completely unknown timestamps & locations) | **PARTIAL / REGIONALLY VALID** (Precise region is known; crop growth stage in days is known; seasonal climate envelope can be calibrated) | **NO** (Heterogeneous origins) | **YES, for specific research plots** |
| **True Longitudinal Sequence for Same Plant?** | **NO** (Single snapshot per detached leaf) | **NO** (Single snapshot per scraped image) | **NO** (Single snapshot per field inspection; cross-sectional across ages) | **NO** (Single snapshots) | Cross-sectional across plots |
| **Licensing & Accessibility** | Open Access (Public Domain / CC0) | Open Access (Academic use) | **Open Access (Kaggle & IEEE DataPort, CC BY 4.0)** | Open Access (Academic use) | Open Access (Zenodo) |
| **Class Balance** | Severe imbalance (e.g. 630 to 5,507 images/class) | Extreme imbalance (some classes have <30 images) | **Moderate imbalance (well-distributed across common diseases: 300 to 1,500/class)** | Variable | Imbalanced across severity stages |
| **Reproducibility for 2-Month Capstone** | High (easy download, but scientifically questionable for multimodal) | Medium (low image count leads to high variance) | **High (Standardized benchmarks, official test splits, realistic metadata)** | Medium (large download, text-focused) | Medium (specialized agronomic focus) |

---

## 3. Deep Scientific Audit of Candidate Datasets

### 3.1 PlantVillage: The Synthetic Multimodal Pitfall
* **Why it is tempting:** Over 54,000 clean images; widely used in undergraduate projects; yields >99% validation accuracy with simple CNNs.
* **Why it is scientifically flawed for multimodal crop disease research:**
  1. *Laboratory Artifacts:* Leaves were clipped and photographed on uniform grey or black poster boards. Standard CNNs inadvertently learn background color, specular reflection, and paper texture rather than pathological lesions (Mohanty et al., 2016; Barbedo, 2018).
  2. *Absence of Spatiotemporal Metadata:* There are zero timestamps, zero GPS coordinates, and zero field identifiers.
  3. *The Synthetic Weather Trap:* Several recent papers (e.g., *AgriFusionNet*, Frontiers 2026; *WeatherLeaf*, 2026) attempt multimodal fusion on PlantVillage by artificially generating temperature and humidity numbers (e.g., sampling from Gaussian distributions centered around typical pathogen incubation temperatures). 
  4. *Scientific Verdict:* Synthesizing weather based on the disease label causes severe **data leakage** and circularity: the model learns the synthetic distribution parameter rather than learning real-world multi-modal synergy. For a university capstone project aiming for peer-reviewed defensibility, claiming real multimodal fusion on PlantVillage is **unacceptable**.

### 3.2 PlantDoc: Real-world Vision, Zero Context
* **Strengths:** 2,598 images scraped from real-world search results; presents challenging real backgrounds, lighting variations, and multiple leaves.
* **Fatal Flaw for Multimodal Learning:** Because images were scraped from web queries (Flickr, Google Images) without EXIF or geographical provenance, it is fundamentally impossible to associate real atmospheric, weather, or soil conditions with any individual sample.

### 3.3 PlantWild: Vision-Language Focus
* **Strengths:** Excellent dataset for multimodal vision-language models (e.g., CLIP, MVPDR) using disease textual descriptions.
* **Limitation for Environmental Fusion:** Designed for textual prompts and visual features, not physical agro-meteorological sensor streams.

### 3.4 Paddy Doctor: The Recommended Primary Dataset
* **Strengths:**
  1. *True Field Conditions:* 10,407 labeled high-resolution training images captured directly in real agricultural fields using mobile devices under natural illumination, shadows, and natural backgrounds.
  2. *Genuine Agronomic Metadata:* Every sample includes two critical ground-truth biological variables:
     - **Paddy Variety (Cultivar):** 10+ distinct commercial varieties (e.g., ADT37, ADT43, ADT45, ASD16, Ponni, IR20), each possessing known genetic disease susceptibility profiles.
     - **Plant Age (Days):** The developmental stage of the crop (ranging from 30 days to 90+ days after planting).
  3. *Known Geographic Origin:* Captured across field plots in Tirunelveli district, Tamil Nadu, India (Coordinates: ~8.7139° N, 77.7567° E), allowing genuine agro-ecological characterization from verified agro-meteorological sources (such as Tamil Nadu Agricultural University / IMD gridded climate reanalysis / NASA POWER).
  4. *Clinically Meaningful Disease Classes (10 classes):*
     - Fungal diseases: *Blast*, *Brown Spot*
     - Bacterial diseases: *Bacterial Leaf Blight*, *Bacterial Leaf Streak*, *Bacterial Panicle Blight*
     - Viral/Vector diseases: *Tungro*
     - Pests & Physiological: *Hispa*, *Dead Heart*, *Downy Mildew*, and *Normal (Healthy)*
  5. *Benchmark Feasibility:* Official competition benchmarks exist, enabling direct scientific comparison with published papers (ACM CoDS-COMAD 2023).

---

## 4. Can True Early Prediction (e.g., 7-Day Ahead) Be Scientifically Established?

### Explicit Scientific Statement:
> **"Early 7-day disease prediction cannot currently be scientifically established from this dataset or any other publicly available crop disease image benchmark."**

### Detailed Justification:
1. **Cross-Sectional vs. Longitudinal:** All public crop disease image datasets (including Paddy Doctor, PlantVillage, PlantDoc, and PlantWild) are **cross-sectional snapshots**. An individual image represents a single inspection event of a plant at time $t$.
2. **Absence of Future Outcome Labels:** No dataset provides repeat-visit tracking where an asymptomatic leaf photographed on Day 0 has a verified ground-truth label of whether that exact leaf developed lesions by Day 7.
3. **Severe Research Violation of Simulating Early Prediction:** Creating synthetic "Day 7" labels by arbitrarily shifting dates or randomly assigning future disease states violates the core scientific integrity rule.
4. **Valid, Defensible Alternative:**
   Instead of fabricating future prediction, we formulate two scientifically robust and impactful tasks:
   - **Task A (Multimodal Disease Diagnosis):** Fusing visual leaf representations with agronomic metadata (cultivar, crop age) and microclimatic vulnerability scores to resolve visual ambiguities between clinically similar diseases (e.g., Bacterial Leaf Blight vs. Bacterial Leaf Streak; Blast vs. Brown Spot).
   - **Task B (Context-Aware Disease Severity & Vulnerability Scoring):** Estimating disease risk and classification confidence conditioned on the plant's physiological age and cultivar resistance.

---

## 5. Dataset Audit Summary & Final Recommendation

| Recommendation Item | Decision |
| :--- | :--- |
| **Recommended Primary Dataset** | **Paddy Doctor: Paddy Disease Classification Dataset** |
| **Crop Selected** | **Paddy / Rice (*Oryza sativa*)** |
| **Image Count** | 10,407 training images across 10 classes |
| **Modality 1 (Visual)** | RGB smartphone field images ($480 \times 640$ / resized to $224 \times 224$) |
| **Modality 2 (Agronomic Metadata)** | Cultivar / Variety (Categorical embedding) + Crop Age in Days (Normalized continuous) |
| **Modality 3 (Environmental Context)** | Seasonal agro-ecological microclimate parameters calibrated for the Tirunelveli paddy basin |
| **Primary Task** | Multimodal Context-Aware Paddy Disease Classification |
| **Baseline Benchmark** | Image-Only ResNet18 Transfer Learning |
| **Comparative Baselines** | Metadata-Only MLP; Image-Only ResNet18; Multimodal Fusion Net (ResNet18 + Tabular MLP) |
