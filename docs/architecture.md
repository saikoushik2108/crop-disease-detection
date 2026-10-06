# System Architecture: Multimodal Crop Disease Detection

## 1. Architectural Overview

The system is designed to overcome the limitations of unimodal computer vision models by fusing three distinct information streams:
1. **Visual Stream (RGB Leaf Imagery):** Captures physical macroscopic lesion morphology (chlorosis, necrosis, spot geometry, margins).
2. **Agronomic Metadata Stream (Cultivar Variety & Crop Age):** Captures genetic resistance priors and developmental phenological susceptibility.
3. **Agro-Environmental Context Stream (Microclimate Indicators):** Captures pathogen-conducive environmental drivers (temperature, relative humidity, vapor pressure deficit, monsoonal rainfall risk).

```
+-----------------------------+
|   RGB Leaf Image (224x224)  |
+-----------------------------+
               |
               v
  Pretrained ResNet18 Backbone
  (conv1 -> layer4 -> avgpool)
               |
               v
    Visual Feature Vector
       f_img in R^512
               |
               +----------------------------------+
                                                  |
+------------------------+                        |
| Variety (Categorical)  |                        |
+------------------------+                        |
            |                                     |
            v                                     |
  nn.Embedding(N, 16)                             |
            |                                     |
            v                                     |
+------------------------+                        |
| Age & Agro-climate     |                        |
| (Continuous 5-d)       |                        |
+------------------------+                        |
            |                                     |
            v                                     |
  Linear(5, 16) + LayerNorm                       |
            |                                     |
            +------------> Concatenation (32-d)   |
                                   |              |
                                   v              |
                         Linear(32, 64) + LN      |
                                   |              |
                                   v              |
                         Context Feature Vector   |
                             f_tab in R^64        |
                                   |              |
                                   +--------------+
                                          |
                                          v
                              Intermediate Fusion Vector
                                 [f_img || f_tab] in R^576
                                          |
                                          v
                              LayerNorm + Dropout(0.3)
                                          |
                                          v
                              Linear(576, 128) + BatchNorm + ReLU
                                          |
                                          v
                              Dropout(0.225) -> Linear(128, 10)
                                          |
                                          v
                              10-Class Disease Softmax Logits
```

## 2. Design Rationale

### Why Intermediate Feature Concatenation?
- **Stability:** Avoids the gradient instability of early raw-pixel concatenation.
- **Independence:** Allows both visual feature maps and tabular embeddings to be extracted and analyzed independently.
- **Efficiency:** Adding only 64 context dimensions to a 512-dimensional visual vector incurs negligible computational overhead (<1% extra FLOPs) while providing vital epidemiological priors.
