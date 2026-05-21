from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from model.dual_branch_transformer import CyclicDoyEncoding, DualBranchS1S2Transformer
from model.ecm_baseline import _make_mlp_head


@dataclass
class ECMProtoV2Output:
    logits: Tensor
    proto_logits: Tensor
    class_proto_logits: Tensor
    early_features: Tensor
    proto_context: Tensor
    fused_features: Tensor
    prototype_attention: Tensor


class ECMProtoV2Model(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
        prototypes_per_class: int,
        fusion_mode: str = "gated_residual",
        future_query_tokens: int = 8,
        future_query_step_days: int = 5,
        future_query_max_doy: int = 270,
    ) -> None:
        super().__init__()
        if fusion_mode not in {"gated_residual", "proto_only"}:
            raise ValueError(f"unsupported fusion_mode={fusion_mode}")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive")
        if future_query_tokens < 0:
            raise ValueError("future_query_tokens must be non-negative")
        if future_query_step_days <= 0:
            raise ValueError("future_query_step_days must be positive")
        if future_query_max_doy <= 0:
            raise ValueError("future_query_max_doy must be positive")
        self.num_classes = num_classes
        self.prototypes_per_class = prototypes_per_class
        self.fusion_mode = fusion_mode
        self.future_query_tokens = future_query_tokens
        self.future_query_step_days = future_query_step_days
        self.future_query_max_doy = future_query_max_doy
        if future_query_tokens > 0:
            self.future_queries = nn.Parameter(torch.randn(future_query_tokens, d_model) * 0.02)
            self.future_doy_encoder = CyclicDoyEncoding(d_model)
        else:
            self.register_parameter("future_queries", None)
            self.future_doy_encoder = None

        self.backbone = DualBranchS1S2Transformer(
            s1_dim=2,
            s2_dim=10,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.time_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.prototype_attention_layer = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion_gate = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.context_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.classifier = _make_mlp_head(d_model, d_model//2, num_classes, dropout)

    def encode(self, x: Tensor, obs_mask: Tensor, doy: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.backbone(x, obs_mask=obs_mask, doy=doy)
        return out.features, out.tokens, out.token_mask

    def forward(
        self,
        early_x: Tensor,
        early_obs_mask: Tensor,
        early_doy: Tensor,
        prototype_tokens: Tensor,
        prototype_token_mask: Tensor,
        proto_temperature: float,
        query_cutoff_doy: Tensor | int | None = None,
    ) -> ECMProtoV2Output:
        if proto_temperature <= 0:
            raise ValueError("proto_temperature must be positive")
        early_features, early_tokens, early_token_mask = self.encode(early_x, early_obs_mask, early_doy)
        query = self.query_norm(early_tokens)
        query_mask = early_token_mask
        if self.future_queries is not None:
            cutoff_doy = self._future_query_start_doy(early_doy, early_token_mask, query_cutoff_doy)
            offsets = torch.arange(
                1,
                self.future_query_tokens + 1,
                device=early_doy.device,
                dtype=early_doy.dtype,
            ).unsqueeze(0) * self.future_query_step_days
            future_doy = (cutoff_doy.unsqueeze(1) + offsets).clamp_max(self.future_query_max_doy)
            future_queries = self.future_queries.unsqueeze(0).expand(query.shape[0], -1, -1)
            future_queries = future_queries + self.future_doy_encoder(future_doy)
            query = torch.cat([query, self.query_norm(future_queries)], dim=1)
            future_mask = torch.ones(query.shape[0], self.future_query_tokens, device=query_mask.device, dtype=torch.bool)
            query_mask = torch.cat([early_token_mask, future_mask], dim=1)

        attended, prototype_attention = self._hierarchical_prototype_attention(
            query=query,
            query_mask=query_mask,
            prototype_tokens=prototype_tokens,
            prototype_token_mask=prototype_token_mask,
        )
        proto_context = self.backbone.pool(attended, query_mask)

        if self.fusion_mode == "proto_only":
            fused_features = proto_context
            logits = self.classifier(fused_features)
        else:
            proto_delta = self.context_proj(proto_context)
            gate = self.fusion_gate(torch.cat([early_features, proto_context], dim=-1))
            fused_features = early_features + gate * proto_delta
            logits = self.classifier(fused_features)

        proto_vectors = self._prototype_vectors(prototype_tokens, prototype_token_mask)
        proto_logits = F.normalize(fused_features.float(), dim=1) @ F.normalize(proto_vectors.float(), dim=1).t()
        proto_logits = proto_logits / proto_temperature
        class_proto_logits = proto_logits.reshape(
            query.shape[0], self.num_classes, self.prototypes_per_class
        ).logsumexp(dim=2)

        return ECMProtoV2Output(
            logits=logits,
            proto_logits=proto_logits,
            class_proto_logits=class_proto_logits,
            early_features=early_features,
            proto_context=proto_context,
            fused_features=fused_features,
            prototype_attention=prototype_attention,
        )

    def _hierarchical_prototype_attention(
        self,
        query: Tensor,
        query_mask: Tensor,
        prototype_tokens: Tensor,
        prototype_token_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, query_len, d_model = query.shape
        num_proto, proto_len, _ = prototype_tokens.shape
        memory = self.memory_norm(prototype_tokens)

        safe_query = query
        no_query = ~query_mask.any(dim=1)
        if no_query.any():
            safe_query = safe_query.clone()
            safe_query[no_query, 0] = 0.0

        time_query = safe_query.unsqueeze(1).expand(batch_size, num_proto, query_len, d_model)
        time_key_value = memory.unsqueeze(0).expand(batch_size, num_proto, proto_len, d_model)
        time_key_mask = ~prototype_token_mask.unsqueeze(0).expand(batch_size, num_proto, proto_len)

        time_context, _ = self.time_attention(
            query=time_query.reshape(batch_size * num_proto, query_len, d_model),
            key=time_key_value.reshape(batch_size * num_proto, proto_len, d_model),
            value=time_key_value.reshape(batch_size * num_proto, proto_len, d_model),
            key_padding_mask=time_key_mask.reshape(batch_size * num_proto, proto_len),
            need_weights=False,
        )
        time_context = time_context.reshape(batch_size, num_proto, query_len, d_model).transpose(1, 2)

        proto_query = safe_query.reshape(batch_size * query_len, 1, d_model)
        proto_key_value = time_context.reshape(batch_size * query_len, num_proto, d_model)
        proto_valid = prototype_token_mask.any(dim=1)
        proto_key_mask = ~proto_valid.unsqueeze(0).expand(batch_size * query_len, num_proto)

        attended, proto_weights = self.prototype_attention_layer(
            query=proto_query,
            key=proto_key_value,
            value=proto_key_value,
            key_padding_mask=proto_key_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        attended = attended.reshape(batch_size, query_len, d_model)
        attended = torch.where(query_mask.unsqueeze(-1), attended, torch.zeros_like(attended))

        prototype_attention = proto_weights.reshape(batch_size, query_len, num_proto)
        prototype_attention = prototype_attention * query_mask.unsqueeze(-1).to(prototype_attention.dtype)
        prototype_attention = prototype_attention / prototype_attention.sum(dim=2, keepdim=True).clamp_min(1e-6)
        prototype_attention = prototype_attention.sum(dim=1)
        prototype_attention = prototype_attention / prototype_attention.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return attended, prototype_attention

    def _prototype_vectors(self, prototype_tokens: Tensor, prototype_token_mask: Tensor) -> Tensor:
        return self.backbone.pool(prototype_tokens, prototype_token_mask)

    @staticmethod
    def _future_query_start_doy(early_doy: Tensor, early_token_mask: Tensor, query_cutoff_doy: Tensor | int | None) -> Tensor:
        if query_cutoff_doy is not None:
            if torch.is_tensor(query_cutoff_doy):
                cutoff = query_cutoff_doy.to(device=early_doy.device, dtype=early_doy.dtype)
                if cutoff.ndim == 0:
                    cutoff = cutoff.expand(early_doy.shape[0])
                return cutoff
            return torch.full((early_doy.shape[0],), query_cutoff_doy, device=early_doy.device, dtype=early_doy.dtype)

        visible = early_token_mask.any(dim=1)
        observed_cutoff = early_doy.masked_fill(~early_token_mask, 0).amax(dim=1)
        fallback_cutoff = early_doy.amin(dim=1)
        return torch.where(visible, observed_cutoff, fallback_cutoff)
