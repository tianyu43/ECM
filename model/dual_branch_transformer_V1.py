import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass
class DualBranchOutput:
    logits: Tensor
    features: Tensor
    tokens: Tensor
    token_mask: Tensor


class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, dim: int, max_period: int = 10000) -> None:
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, time_steps: Tensor) -> Tensor:
        half_dim = self.dim // 2
        device = time_steps.device
        freq = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )
        angles = time_steps.unsqueeze(-1).float() * freq
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb


class MaskedMeanPool(nn.Module):
    def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
        weights = token_mask.float().unsqueeze(-1)
        summed = (tokens * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return summed / denom


class SensorBranch(nn.Module):
    """Encode one sensor stream as time tokens with value, mask, DOY, and delta-time cues."""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_time_steps: int,
    ) -> None:
        super().__init__()
        self.max_time_steps = max_time_steps
        self.value_proj = nn.Sequential(
            nn.Linear(input_dim * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.doy_encoder = SinusoidalTimeEncoding(d_model)
        self.delta_encoder = SinusoidalTimeEncoding(d_model)
        self.position_embed = nn.Embedding(max_time_steps, d_model)
        self.pre_norm = nn.LayerNorm(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: Tensor, obs_mask: Tensor, doy: Tensor) -> Tensor:
        x_filled = torch.where(obs_mask, x, torch.zeros_like(x))
        token_mask = obs_mask.any(dim=-1)
        fused = torch.cat([x_filled, obs_mask.float()], dim=-1)
        tokens = self.value_proj(fused)

        delta = self._compute_time_delta(doy, token_mask)
        pos_ids = torch.arange(x.shape[1], device=x.device).clamp_max(self.max_time_steps - 1)
        pos_ids = pos_ids.unsqueeze(0).expand(x.shape[0], -1)
        tokens = tokens + self.doy_encoder(doy) + self.delta_encoder(delta) + self.position_embed(pos_ids)
        tokens = self.pre_norm(tokens)
        safe_tokens = tokens
        safe_key_mask = ~token_mask
        no_obs = ~token_mask.any(dim=1)
        if no_obs.any():
            safe_tokens = safe_tokens.clone()
            safe_key_mask = safe_key_mask.clone()
            safe_tokens[no_obs, 0] = 0.0
            safe_key_mask[no_obs, 0] = False

        encoded = self.encoder(safe_tokens, src_key_padding_mask=safe_key_mask)
        return encoded * token_mask.unsqueeze(-1).float()

    @staticmethod
    def _compute_time_delta(doy: Tensor, token_mask: Tensor) -> Tensor:
        doy = doy.float()
        _, steps = doy.shape
        delta = torch.zeros_like(doy)
        prev = doy[:, :1]
        for idx in range(steps):
            current = doy[:, idx : idx + 1]
            step_delta = (current - prev).clamp_min(0.0)
            has_obs = token_mask[:, idx : idx + 1]
            delta[:, idx : idx + 1] = torch.where(has_obs, step_delta, torch.zeros_like(step_delta))
            prev = torch.where(has_obs, current, prev)
        return delta


class SensorFusion(nn.Module):
    """Fuse S1 and S2 token sequences with bidirectional cross-attention."""

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.s1_to_s2 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.s2_to_s1 = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        self.s1_norm = nn.LayerNorm(d_model)
        self.s2_norm = nn.LayerNorm(d_model)
        self.merge_norm = nn.LayerNorm(d_model * 2)
        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, s1_tokens: Tensor, s2_tokens: Tensor, s1_token_mask: Tensor, s2_token_mask: Tensor) -> Tensor:
        s1_query_mask = s1_token_mask.unsqueeze(-1).float()
        s2_query_mask = s2_token_mask.unsqueeze(-1).float()

        safe_s1_tokens = s1_tokens
        safe_s2_tokens = s2_tokens
        safe_s1_key_mask = ~s1_token_mask
        safe_s2_key_mask = ~s2_token_mask

        no_s1 = ~s1_token_mask.any(dim=1)
        no_s2 = ~s2_token_mask.any(dim=1)
        if no_s1.any():
            safe_s1_key_mask = safe_s1_key_mask.clone()
            safe_s1_key_mask[no_s1, 0] = False
            safe_s1_tokens = safe_s1_tokens.clone()
            safe_s1_tokens[no_s1, 0] = 0.0
        if no_s2.any():
            safe_s2_key_mask = safe_s2_key_mask.clone()
            safe_s2_key_mask[no_s2, 0] = False
            safe_s2_tokens = safe_s2_tokens.clone()
            safe_s2_tokens[no_s2, 0] = 0.0

        s1_cross, _ = self.s1_to_s2(
            query=s1_tokens,
            key=safe_s2_tokens,
            value=safe_s2_tokens,
            key_padding_mask=safe_s2_key_mask,
        )
        s2_cross, _ = self.s2_to_s1(
            query=s2_tokens,
            key=safe_s1_tokens,
            value=safe_s1_tokens,
            key_padding_mask=safe_s1_key_mask,
        )

        s1_updated = self.s1_norm(s1_tokens + s1_cross) * s1_query_mask
        s2_updated = self.s2_norm(s2_tokens + s2_cross) * s2_query_mask

        fused = torch.cat([s1_updated, s2_updated], dim=-1)
        fused = self.merge_norm(fused)
        return self.out(fused)


class DualBranchS1S2Transformer(nn.Module):
    """
    Dual-branch S1/S2 transformer.

    Expected input layout:
    - channels [:s1_dim] are Sentinel-1
    - channels [s1_dim:s1_dim+s2_dim] are Sentinel-2
    """

    def __init__(
        self,
        s1_dim: int,
        s2_dim: int,
        num_classes: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_time_steps: int = 366,
    ) -> None:
        super().__init__()
        self.s1_dim = s1_dim
        self.s2_dim = s2_dim

        self.s1_branch = SensorBranch(
            input_dim=s1_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_time_steps=max_time_steps,
        )
        self.s2_branch = SensorBranch(
            input_dim=s2_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_time_steps=max_time_steps,
        )
        self.fusion = SensorFusion(d_model=d_model, dropout=dropout)
        self.pool = MaskedMeanPool()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: Tensor, obs_mask: Optional[Tensor] = None, doy: Optional[Tensor] = None) -> DualBranchOutput:
        if x.ndim != 3:
            raise ValueError(f"x must be [B, T, C], got shape {tuple(x.shape)}")
        if obs_mask is None:
            obs_mask = torch.isfinite(x)
        obs_mask = obs_mask.bool()
        if doy is None:
            doy = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        expected_dim = self.s1_dim + self.s2_dim
        if x.shape[-1] == expected_dim:
            s1_x = x[:, :, : self.s1_dim]
            s2_x = x[:, :, self.s1_dim : self.s1_dim + self.s2_dim]
            s1_mask = obs_mask[:, :, : self.s1_dim]
            s2_mask = obs_mask[:, :, self.s1_dim : self.s1_dim + self.s2_dim]
        elif x.shape[-1] == self.s2_dim:
            # Allow S2-only inputs by synthesizing an empty S1 branch.
            s1_x = torch.zeros(x.shape[0], x.shape[1], self.s1_dim, device=x.device, dtype=x.dtype)
            s2_x = x
            s1_mask = torch.zeros(x.shape[0], x.shape[1], self.s1_dim, device=x.device, dtype=torch.bool)
            s2_mask = obs_mask
        else:
            raise ValueError(
                f"Expected channel dim {expected_dim} (S1+S2) or {self.s2_dim} (S2-only), got {x.shape[-1]}"
            )

        s1_token_mask = s1_mask.any(dim=-1)
        s2_token_mask = s2_mask.any(dim=-1)
        token_mask = s1_token_mask | s2_token_mask

        doy = doy.to(x.device)
        s1_tokens = self.s1_branch(s1_x, s1_mask, doy)
        s2_tokens = self.s2_branch(s2_x, s2_mask, doy)
        fused_tokens = self.fusion(s1_tokens, s2_tokens, s1_token_mask, s2_token_mask)
        pooled = self.pool(fused_tokens, token_mask)
        logits = self.head(pooled)
        return DualBranchOutput(logits=logits, features=pooled, tokens=fused_tokens, token_mask=token_mask)
