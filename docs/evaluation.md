# Model Evaluation & Explainability Protocols

## 1. Quantitative Evaluation Metrics

1. **Overall Accuracy:** Percentage of correct disease diagnoses across total test samples.
2. **Macro-Averaged Precision, Recall, and F1-Score:** Unweighted arithmetic mean across all 10 classes, ensuring rare classes have equal diagnostic weight.
3. **Weighted F1-Score:** F1-score weighted by class prevalence.
4. **Normalized Confusion Matrix:** Analysis of false positive and false negative transitions between visually confusing disease pairs.

## 2. Qualitative Explainability via Grad-CAM

- **Target Layer:** ResNet18 `layer4` (the final convolutional block before global average pooling).
- **Visualization:** Class activation gradients are backpropagated to compute spatial feature importance heatmaps, rendered via OpenCV `COLORMAP_JET` and overlaid at 45% alpha over original leaves.
- **Verification Rule:** Confirm that high-activation regions align with physical lesion centers rather than background soil, water droplets, or leaf margins.
