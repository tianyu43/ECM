from __future__ import annotations

from torch import Tensor, nn

from model.dual_branch_transformer import DualBranchS1S2Transformer


def _make_mlp_head(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ECMBaselineModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.backbone = DualBranchS1S2Transformer(
            s1_dim=2,
            s2_dim=10,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.classifier = _make_mlp_head(d_model, d_model//2, num_classes, dropout)

    def encode(self, x: Tensor, obs_mask: Tensor, doy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.backbone(x, obs_mask=obs_mask, doy=doy)
        return out.features, out.tokens, out.token_mask

    def forward(self, x: Tensor, obs_mask: Tensor, doy: Tensor) -> Tensor:
        features, _, _ = self.encode(x=x, obs_mask=obs_mask, doy=doy)
        return self.classifier(features)
