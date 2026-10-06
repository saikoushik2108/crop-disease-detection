import torch
import torch.nn as nn


class TabularContextMLP(nn.Module):
    """
    Tabular Context-Only Baseline.
    
    Processes:
    1. Variety (categorical entity embedding)
    2. Plant Age & Agro-environmental indicators (continuous dense projection)
    
    Can function as an independent classifier or as the context encoder
    branch in the multimodal network.
    """

    def __init__(
        self,
        num_varieties: int,
        num_continuous: int,
        num_classes: int = 10,
        variety_embedding_dim: int = 16,
        continuous_proj_dim: int = 16,
        tabular_embedding_dim: int = 64,
        dropout: float = 0.2
    ):
        super().__init__()
        self.variety_embedding = nn.Embedding(
            num_embeddings=num_varieties,
            embedding_dim=variety_embedding_dim
        )
        self.continuous_layer = nn.Sequential(
            nn.Linear(num_continuous, continuous_proj_dim),
            nn.LayerNorm(continuous_proj_dim),
            nn.ReLU()
        )

        fused_in_dim = variety_embedding_dim + continuous_proj_dim

        self.context_encoder = nn.Sequential(
            nn.Linear(fused_in_dim, tabular_embedding_dim),
            nn.LayerNorm(tabular_embedding_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.classifier = nn.Sequential(
            nn.Linear(tabular_embedding_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(32, num_classes)
        )
        self.tabular_embedding_dim = tabular_embedding_dim

    def extract_features(
        self,
        variety_idx: torch.Tensor,
        cont_features: torch.Tensor
    ) -> torch.Tensor:
        """Extract compact tabular context embedding vector."""
        var_emb = self.variety_embedding(variety_idx)
        cont_emb = self.continuous_layer(cont_features)
        combined = torch.cat([var_emb, cont_emb], dim=1)
        context_vec = self.context_encoder(combined)
        return context_vec

    def forward(
        self,
        variety_idx: torch.Tensor,
        cont_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through tabular network returning class logits."""
        feat = self.extract_features(variety_idx, cont_features)
        logits = self.classifier(feat)
        return logits
