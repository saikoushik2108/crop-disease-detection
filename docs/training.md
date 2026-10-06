# Training Strategy & Ablation Methodology

## 1. Experimental Setup

- **Optimizer:** AdamW with initial learning rate $\eta = 3 \times 10^{-4}$ and weight decay $\lambda = 10^{-4}$.
- **Loss Function:** Multi-Class Cross-Entropy Loss with Softmax.
- **Learning Rate Schedule:** `ReduceLROnPlateau` monitoring validation Macro F1 with patience of 2 epochs and reduction factor $\gamma = 0.5$.
- **Early Stopping:** Monitored on validation Macro F1 with patience of 4 epochs.
- **Checkpointing:** State dictionary saved whenever validation Macro F1 achieves a new maximum.

## 2. Ablation Study Design

| Experiment | Modalities Included | Primary Hypothesis Tested |
| :--- | :--- | :--- |
| **Exp A: Image-Only Baseline** | Leaf RGB pixels only | Measures raw visual diagnostic capacity using transfer learning. |
| **Exp B: Tabular-Only Baseline** | Variety + Age + Agro-climate | Establishes the upper bound of non-visual agronomic context alone. |
| **Exp C: Image + Metadata** | Image + Variety + Age | Evaluates the gain from purely biological host characteristics. |
| **Exp D: Full Multimodal Fusion** | Image + Metadata + Agro-climate | Evaluates the full tripartite plant disease triangle hypothesis. |
