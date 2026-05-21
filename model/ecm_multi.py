from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.dual_branch_transformer import DualBranchS1S2Transformer
from model.ecm_baseline import _make_mlp_head


class ECMMultiModel(nn.Module):
    """Early-view classifier with future-view auxiliary projection heads."""

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
        self.future_projector = _make_mlp_head(d_model, d_model*2, d_model, dropout)
        self.future_predictor = _make_mlp_head(d_model, d_model*2, d_model, dropout)

    def encode(self, x: Tensor, obs_mask: Tensor, doy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.backbone(x, obs_mask=obs_mask, doy=doy)
        return out.features, out.tokens, out.token_mask

    def forward(self, x: Tensor, obs_mask: Tensor, doy: Tensor) -> Tensor:
        features, _, _ = self.encode(x=x, obs_mask=obs_mask, doy=doy)
        return self.classifier(features)

    def predict_future_features(self, early_features: Tensor) -> Tensor:
        early_projected = self.future_projector(early_features)
        return self.future_predictor(early_projected)

    def project_future_target(self, future_features: Tensor) -> Tensor:
        return self.future_projector(future_features)


def make_cutoff_view(
    x: Tensor,
    obs_mask: Tensor,
    doy: Tensor,
    cutoffs: Tensor,
) -> tuple[Tensor, Tensor]:
    keep = doy <= cutoffs.unsqueeze(1)
    view_mask = obs_mask & keep.unsqueeze(-1)
    view_x = torch.where(view_mask, x, torch.zeros_like(x))
    return view_x, view_mask


def future_kd_loss(early_logits: Tensor, future_logits: Tensor, temperature: float) -> Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    early_log_probs = F.log_softmax(early_logits.float() / temperature, dim=1)
    future_probs = F.softmax(future_logits.detach().float() / temperature, dim=1)
    return F.kl_div(early_log_probs, future_probs, reduction="batchmean") * (temperature * temperature)


def future_proto_ce_loss(
    early_features: Tensor,
    future_features: Tensor,
    y: Tensor,
    temperature: float,
) -> Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    early_norm = F.normalize(early_features.float(), dim=1)
    future_norm = F.normalize(future_features.detach().float(), dim=1)
    classes = y.unique(sorted=True)
    if classes.numel() < 2:
        return early_features.new_zeros(())
    prototypes = []
    target = torch.empty_like(y)
    for proto_idx, class_id in enumerate(classes):
        class_mask = y == class_id
        prototype = F.normalize(future_norm[class_mask].mean(dim=0), dim=0)
        prototypes.append(prototype)
        target[class_mask] = proto_idx
    prototype_matrix = torch.stack(prototypes, dim=0)
    logits = early_norm @ prototype_matrix.t() / temperature
    return F.cross_entropy(logits, target)


def future_prediction_loss(
    model: ECMMultiModel,
    early_features: Tensor,
    future_features: Tensor,
) -> Tensor:
    predicted_future = model.predict_future_features(early_features)
    with torch.no_grad():
        target_future = model.project_future_target(future_features.detach())
    predicted_future = F.normalize(predicted_future.float(), dim=1)
    target_future = F.normalize(target_future.float(), dim=1)
    return (1.0 - F.cosine_similarity(predicted_future, target_future, dim=1)).mean()


def auxiliary_scale(epoch: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= warmup_epochs:
        return 0.0
    return 1.0
