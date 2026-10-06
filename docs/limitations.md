# Known Limitations & Scientific Caveats

This document **must** be read alongside any experimental results or claims made by this project. Scientific honesty demands that limitations are stated explicitly and prominently.

---

## 1. No True Longitudinal Early Prediction

> **"Early 7-day disease prediction cannot currently be scientifically established from this dataset or any other publicly available crop disease image benchmark."**

The Paddy Doctor dataset — and every other public plant disease image dataset evaluated (PlantVillage, PlantDoc, PlantWild) — is **cross-sectional**: each image captures a single point-in-time snapshot of one leaf. No dataset tracks the same physical leaf sequentially from Day 0 (asymptomatic) through Day 7 (symptomatic outcome). Any "prediction" of future disease occurrence would require fabrication of temporal labels, which violates fundamental research ethics.

## 2. Environmental Variables Are Agro-Ecologically Calibrated, Not Per-Leaf Sensor Measurements

The Paddy Doctor dataset provides **variety** and **crop age** per image, but does not embed physical IoT weather sensor readings on a per-image basis. The environmental context variables (temperature, humidity, VPD, rain risk) used in this project are derived from the known geographic domain (Tirunelveli, Tamil Nadu) and seasonal phenological models. They represent valid regional agro-ecological baselines, but are not direct micro-sensor measurements attached to each individual leaf.

## 3. Development Benchmark Dataset

When the official Paddy Doctor Kaggle images are not present in `data/raw/`, the system generates a synthetic development benchmark dataset with procedurally drawn leaf images. Results obtained on this benchmark are **only valid for pipeline verification** and must not be reported as real experimental findings. All published results must be produced on the actual Paddy Doctor competition data.

## 4. Single Geographic Domain

All training and evaluation images originate from one agro-ecological zone (Tirunelveli district, Tamil Nadu, India). The model may not generalize to rice varieties, diseases, or climatic conditions from other geographies (e.g., Southeast Asia, Sub-Saharan Africa) without domain adaptation or fine-tuning.

## 5. Class Imbalance

Some disease classes in the Paddy Doctor dataset have significantly fewer samples than others. While stratified splitting preserves proportional representation, minority-class recall may be lower. No oversampling or class-weighted loss was applied in the baseline configuration.

## 6. Computational Constraints

The project is designed for consumer hardware (CPU or modest GPU). ResNet18 was selected for efficiency, not maximal accuracy. Larger backbones (ResNet50, EfficientNet-B4, Swin Transformer) might improve performance but are beyond the 2-month timeline and hardware budget.

## 7. Grad-CAM Is Correlational, Not Causal

Grad-CAM heatmaps show which spatial regions of the image the model *attends to* when making a prediction. High activation over a lesion does **not** prove the model has learned true pathological features; it may also be influenced by co-occurring artifacts (e.g., leaf veins near lesions, water droplets). Grad-CAM is an explanatory tool, not a causal proof.
