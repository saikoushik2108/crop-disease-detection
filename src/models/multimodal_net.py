import torch
import torch.nn as nn
from typing import Tuple
from src.models.baseline_image import ImageBaselineResNet
from src.models.baseline_tabular import TabularContextMLP


class MultimodalCropNet(nn.Module):
    """
    Intermediate Feature Concatenation Multimodal Neural Network.
    
    Architecture:
    1. Vision Stream: ResNet18 backbone -> 512-d visual embedding (f_img)
    2. Context Stream: Tabular MLP -> 64-d context embedding (f_tab)
    3. Intermediate Fusion: Concatenation [f_img || f_tab] -> 576-d vector
    4. Fusion Head: LayerNorm -> Dropout -> Dense(128) -> ReLU -> Linear(10)
    """

    def __init__(
        self,
        num_classes: int = 10,
        num_varieties: int = 20,
        num_continuous: int = 5,
        pretrained: bool = True,
        variety_embedding_dim: int = 16,
        continuous_proj_dim: int = 16,
        tabular_embedding_dim: int = 64,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        ablation_mode: str = "full"
    ):
        super().__init__()
        self.ablation_mode = ablation_mode
        self.variety_embedding_dim = variety_embedding_dim
        self.continuous_proj_dim = continuous_proj_dim
        
        # 1. Vision Stream
        self.image_encoder = ImageBaselineResNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        self.image_feature_dim = self.image_encoder.feature_dim  # 512

        # 2. Context Stream
        self.tabular_encoder = TabularContextMLP(
            num_varieties=num_varieties,
            num_continuous=num_continuous,
            num_classes=num_classes,
            variety_embedding_dim=variety_embedding_dim,
            continuous_proj_dim=continuous_proj_dim,
            tabular_embedding_dim=tabular_embedding_dim,
            dropout=dropout * 0.75
        )
        self.tabular_feature_dim = tabular_embedding_dim  # 64

        # 3. Intermediate Multimodal Fusion Head
        total_fusion_dim = self.image_feature_dim + self.tabular_feature_dim  # 512 + 64 = 576
        
        self.fusion_norm = nn.LayerNorm(total_fusion_dim)
        self.fusion_dropout = nn.Dropout(p=dropout)
        
        self.fusion_head = nn.Sequential(
            nn.Linear(total_fusion_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout * 0.75),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def extract_multimodal_features(
        self,
        image: torch.Tensor,
        variety_idx: torch.Tensor,
        cont_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract individual modality vectors and fused representation with strict ablation masking."""
        f_img = self.image_encoder.extract_features(image)
        
        if self.ablation_mode == "image_variety":
            # Exclude age from forward pass
            masked_cont = torch.zeros_like(cont_features)
            f_tab = self.tabular_encoder.extract_features(variety_idx, masked_cont)
        elif self.ablation_mode == "image_age":
            # Exclude variety from forward pass
            masked_var = torch.zeros_like(variety_idx)
            f_tab = self.tabular_encoder.extract_features(masked_var, cont_features)
        else:
            f_tab = self.tabular_encoder.extract_features(variety_idx, cont_features)

        f_fused = torch.cat([f_img, f_tab], dim=1)
        return f_img, f_tab, f_fused

    def forward(
        self,
        image: torch.Tensor,
        variety_idx: torch.Tensor,
        cont_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through multimodal pipeline returning class logits."""
        _, _, f_fused = self.extract_multimodal_features(image, variety_idx, cont_features)
        normed = self.fusion_norm(f_fused)
        dropped = self.fusion_dropout(normed)
        logits = self.fusion_head(dropped)
        return logits
