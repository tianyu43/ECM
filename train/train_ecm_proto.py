from __future__ import annotations

try:
    from train._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from dataclasses import asdict, replace

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from config import ECMProtoTrainConfig, MODEL_SCALE_PRESETS, apply_model_scale
from model.ecm_proto import ECMProtoModel
try:
    from train.train_ecm_baseline import cutoff_sample_weights, resolve_experiment_site_dir, weighted_ce
except ModuleNotFoundError:
    from train_ecm_baseline import cutoff_sample_weights, resolve_experiment_site_dir, weighted_ce
from utils.data_pipeline import prepare_train_val_test_data
from utils.helper import apply_doy_cutoff, summarize_cutoff_metrics
from utils.training_common import (
    Logger,
    apply_time_series_augmentation,
    class_distribution,
    compute_classification_metrics,
    load_checkpoint,
    make_loader,
    move_batch_to_device,
    resolve_device,
    sample_random_early_mask,
    seed_everything,
)


def parse_args() -> ECMProtoTrainConfig:
    defaults = ECMProtoTrainConfig()
    parser = argparse.ArgumentParser(description="Train ECM-proto with multi-prototype memory cross-attention.")
    parser.add_argument("--site", default=defaults.site)
    parser.add_argument("--sensor-mode", choices=("s1s2", "s2"), default=defaults.sensor_mode)
    parser.add_argument("--train-years", nargs="+", type=int, default=list(defaults.train_years))
    parser.add_argument("--test-year", type=int, default=defaults.test_year)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--model-scale", choices=tuple(MODEL_SCALE_PRESETS.keys()), default=defaults.model_scale)
    parser.add_argument("--out-subdir", default=defaults.out_subdir)
    parser.add_argument("--experiment-site-dirname", default=defaults.experiment_site_dirname)
    parser.add_argument("--init-checkpoint", default=defaults.init_checkpoint)
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--time-shift-max-days", type=int, default=defaults.time_shift_max_days)
    parser.add_argument("--random-step-drop-prob", type=float, default=defaults.random_step_drop_prob)
    parser.add_argument("--train-subset-fraction", type=float, default=defaults.train_subset_fraction)
    parser.add_argument("--val-subset-fraction", type=float, default=defaults.val_subset_fraction)
    parser.add_argument("--test-subset-fraction", type=float, default=defaults.test_subset_fraction)
    parser.add_argument("--early-cutoff-weight", type=float, default=defaults.early_cutoff_weight)
    parser.add_argument("--mid-cutoff-weight", type=float, default=defaults.mid_cutoff_weight)
    parser.add_argument("--late-cutoff-weight", type=float, default=defaults.late_cutoff_weight)
    parser.add_argument("--prototypes-per-class", type=int, default=defaults.prototypes_per_class)
    parser.add_argument("--proto-temperature", type=float, default=defaults.proto_temperature)
    parser.add_argument("--proto-loss-weight", type=float, default=defaults.proto_loss_weight)
    parser.add_argument("--attn-loss-weight", type=float, default=defaults.attn_loss_weight)
    parser.add_argument("--proto-warmup-epochs", type=int, default=defaults.proto_warmup_epochs)
    parser.add_argument("--prototype-ema-momentum", type=float, default=defaults.prototype_ema_momentum)
    parser.add_argument("--kmeans-iters", type=int, default=defaults.kmeans_iters)
    parser.add_argument("--prototype-batch-size", type=int, default=defaults.prototype_batch_size)
    parser.add_argument("--fusion-mode", choices=("gated_residual", "proto_only"), default=defaults.fusion_mode)
    parser.add_argument("--future-query-tokens", type=int, default=defaults.future_query_tokens)
    parser.add_argument("--future-query-step-days", type=int, default=defaults.future_query_step_days)
    parser.add_argument("--future-query-max-doy", type=int, default=defaults.future_query_max_doy)
    args = parser.parse_args()
    cfg = ECMProtoTrainConfig(
        site=args.site,
        sensor_mode=args.sensor_mode,
        train_years=tuple(args.train_years),
        test_year=args.test_year,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_scale=args.model_scale,
        out_subdir=args.out_subdir,
        tag=defaults.tag,
        experiment_site_dirname=args.experiment_site_dirname,
        init_checkpoint=args.init_checkpoint,
        min_learning_rate=args.min_learning_rate,
        time_shift_max_days=args.time_shift_max_days,
        random_step_drop_prob=args.random_step_drop_prob,
        train_subset_fraction=args.train_subset_fraction,
        val_subset_fraction=args.val_subset_fraction,
        test_subset_fraction=args.test_subset_fraction,
        early_cutoff_weight=args.early_cutoff_weight,
        mid_cutoff_weight=args.mid_cutoff_weight,
        late_cutoff_weight=args.late_cutoff_weight,
        prototypes_per_class=args.prototypes_per_class,
        proto_temperature=args.proto_temperature,
        proto_loss_weight=args.proto_loss_weight,
        attn_loss_weight=args.attn_loss_weight,
        proto_warmup_epochs=args.proto_warmup_epochs,
        prototype_ema_momentum=args.prototype_ema_momentum,
        kmeans_iters=args.kmeans_iters,
        prototype_batch_size=args.prototype_batch_size,
        fusion_mode=args.fusion_mode,
        future_query_tokens=args.future_query_tokens,
        future_query_step_days=args.future_query_step_days,
        future_query_max_doy=args.future_query_max_doy,
    )
    return apply_model_scale(cfg, cfg.model_scale)


def _kmeans(features: Tensor, k: int, iters: int, seed: int) -> Tensor:
    n = features.shape[0]
    if n == 0:
        raise ValueError("cannot run kmeans on an empty class")
    k_eff = min(k, n)
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    init_idx = torch.randperm(n, generator=generator, device=features.device)[:k_eff]
    centroids = features[init_idx].clone()
    assignments = torch.zeros(n, dtype=torch.long, device=features.device)
    for _ in range(iters):
        distances = torch.cdist(features, centroids)
        assignments = distances.argmin(dim=1)
        new_centroids = []
        for idx in range(k_eff):
            mask = assignments == idx
            if mask.any():
                new_centroids.append(features[mask].mean(dim=0))
            else:
                replacement = torch.randint(0, n, (1,), generator=generator, device=features.device).item()
                new_centroids.append(features[replacement])
        centroids = torch.stack(new_centroids, dim=0)
    if k_eff == k:
        return assignments
    repeated = assignments.clone()
    return repeated.clamp_max(k - 1)


def _full_sequence_loader(split: dict[str, Tensor], batch_size: int, num_workers: int) -> DataLoader:
    ds = TensorDataset(split["x"].float(), split["obs_mask"].bool(), split["doy"].long(), split["y"].long())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def build_multi_prototypes(
    model: ECMProtoModel,
    split: dict[str, Tensor],
    cfg: ECMProtoTrainConfig,
    device: torch.device,
    logger: Logger,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    model.eval()
    loader = _full_sequence_loader(split, cfg.prototype_batch_size, cfg.num_workers)
    features_by_class = [[] for _ in range(cfg.num_classes)]
    with torch.no_grad():
        for x, obs_mask, doy, y in loader:
            x = x.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            doy = doy.to(device, non_blocking=True)
            features, _, _ = model.encode(x, obs_mask, doy)
            features = F.normalize(features.float(), dim=1).cpu()
            for class_id in range(cfg.num_classes):
                class_mask = y == class_id
                if class_mask.any():
                    features_by_class[class_id].append(features[class_mask])

    assignments_by_class: list[Tensor] = []
    cluster_counts: list[list[int]] = []
    for class_id, parts in enumerate(features_by_class):
        if not parts:
            raise ValueError(f"cannot build prototypes: class_id={class_id} has no training samples")
        class_features = torch.cat(parts, dim=0).to(device)
        assignments = _kmeans(class_features, cfg.prototypes_per_class, cfg.kmeans_iters, cfg.seed + class_id).cpu()
        assignments_by_class.append(assignments)
        cluster_counts.append(torch.bincount(assignments, minlength=cfg.prototypes_per_class).tolist())

    token_sums: Tensor | None = None
    mask_sums: Tensor | None = None
    class_offsets = [0 for _ in range(cfg.num_classes)]
    with torch.no_grad():
        for x, obs_mask, doy, y in loader:
            x = x.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            doy = doy.to(device, non_blocking=True)
            _, tokens, token_mask = model.encode(x, obs_mask, doy)
            if token_sums is None:
                token_sums = torch.zeros(
                    cfg.num_classes,
                    cfg.prototypes_per_class,
                    tokens.shape[1],
                    tokens.shape[2],
                    device=device,
                    dtype=tokens.dtype,
                )
                mask_sums = torch.zeros(
                    cfg.num_classes,
                    cfg.prototypes_per_class,
                    tokens.shape[1],
                    device=device,
                    dtype=tokens.dtype,
                )
            for class_id in range(cfg.num_classes):
                row_idx = torch.nonzero(y == class_id, as_tuple=False).flatten()
                if row_idx.numel() == 0:
                    continue
                start = class_offsets[class_id]
                stop = start + row_idx.numel()
                class_offsets[class_id] = stop
                proto_ids = assignments_by_class[class_id][start:stop].to(device)
                for proto_id in range(cfg.prototypes_per_class):
                    proto_rows = row_idx[proto_ids.cpu() == proto_id].to(device)
                    if proto_rows.numel() == 0:
                        continue
                    row_mask = token_mask[proto_rows].to(tokens.dtype)
                    token_sums[class_id, proto_id] += (tokens[proto_rows] * row_mask.unsqueeze(-1)).sum(dim=0)
                    mask_sums[class_id, proto_id] += row_mask.sum(dim=0)

    assert token_sums is not None and mask_sums is not None
    prototype_tokens = token_sums / mask_sums.unsqueeze(-1).clamp_min(1.0)
    prototype_token_mask = mask_sums > 0
    empty_proto = ~prototype_token_mask.any(dim=2)
    if empty_proto.any():
        prototype_token_mask[empty_proto, 0] = True
    prototype_tokens = prototype_tokens.reshape(cfg.num_classes * cfg.prototypes_per_class, prototype_tokens.shape[2], -1).detach()
    prototype_token_mask = prototype_token_mask.reshape(cfg.num_classes * cfg.prototypes_per_class, prototype_token_mask.shape[2]).detach()
    summary = {
        "source": "train full sequence",
        "prototypes_per_class": cfg.prototypes_per_class,
        "cluster_counts": cluster_counts,
        "prototype_shape": list(prototype_tokens.shape),
    }
    logger.log(f"rebuilt prototypes summary={summary}")
    return prototype_tokens, prototype_token_mask, summary


def prototype_attention_loss(prototype_attention: Tensor, y: Tensor, cfg: ECMProtoTrainConfig) -> Tensor:
    class_attention = prototype_attention.reshape(y.shape[0], cfg.num_classes, cfg.prototypes_per_class).sum(dim=2)
    return -torch.log(class_attention.gather(1, y.unsqueeze(1)).clamp_min(1e-6)).mean()


def baseline_logits(model: ECMProtoModel, x: Tensor, obs_mask: Tensor, doy: Tensor) -> tuple[Tensor, Tensor]:
    features, _, _ = model.encode(x, obs_mask, doy)
    return model.classifier(features), features


def initialize_from_baseline_checkpoint(model: ECMProtoModel, checkpoint_path: str, device: torch.device, logger: Logger) -> None:
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    source_state = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    target_state = model.state_dict()
    loaded_keys = []
    skipped_keys = []
    for key, value in source_state.items():
        if not (key.startswith("backbone.") or key.startswith("classifier.")):
            continue
        if key in target_state and target_state[key].shape == value.shape:
            target_state[key] = value.to(device=target_state[key].device, dtype=target_state[key].dtype)
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)
    model.load_state_dict(target_state)
    logger.log(
        f"initialized proto from baseline checkpoint={checkpoint_path} "
        f"loaded_keys={len(loaded_keys)} skipped_keys={len(skipped_keys)}"
    )


@torch.no_grad()
def update_ema_prototypes(
    model: ECMProtoModel,
    x: Tensor,
    obs_mask: Tensor,
    doy: Tensor,
    y: Tensor,
    prototype_tokens: Tensor,
    prototype_token_mask: Tensor,
    cfg: ECMProtoTrainConfig,
) -> tuple[Tensor, Tensor, dict[str, object]]:
    if not 0.0 <= cfg.prototype_ema_momentum < 1.0:
        raise ValueError("prototype_ema_momentum must be in [0, 1)")

    was_training = model.training
    model.eval()
    features, tokens, token_mask = model.encode(x, obs_mask, doy)
    proto_vectors = model._prototype_vectors(prototype_tokens, prototype_token_mask)
    updated = 0
    assigned = 0
    momentum = cfg.prototype_ema_momentum

    for class_id in range(cfg.num_classes):
        row_idx = torch.nonzero(y == class_id, as_tuple=False).flatten()
        if row_idx.numel() == 0:
            continue
        start = class_id * cfg.prototypes_per_class
        stop = start + cfg.prototypes_per_class
        class_proto_vectors = proto_vectors[start:stop]
        scores = F.normalize(features[row_idx].float(), dim=1) @ F.normalize(class_proto_vectors.float(), dim=1).t()
        proto_ids = scores.argmax(dim=1)
        assigned += int(row_idx.numel())

        for proto_id in range(cfg.prototypes_per_class):
            proto_rows = row_idx[proto_ids == proto_id]
            if proto_rows.numel() == 0:
                continue
            row_mask = token_mask[proto_rows].to(tokens.dtype)
            mask_sum = row_mask.sum(dim=0)
            avg_mask = mask_sum > 0
            if not avg_mask.any():
                continue
            avg_tokens = (tokens[proto_rows] * row_mask.unsqueeze(-1)).sum(dim=0) / mask_sum.unsqueeze(-1).clamp_min(1.0)
            bank_idx = start + proto_id
            updated_tokens = momentum * prototype_tokens[bank_idx] + (1.0 - momentum) * avg_tokens
            prototype_tokens[bank_idx] = torch.where(avg_mask.unsqueeze(-1), updated_tokens, prototype_tokens[bank_idx])
            prototype_token_mask[bank_idx] = prototype_token_mask[bank_idx] | avg_mask
            updated += 1

    if was_training:
        model.train()
    summary = {"source": "batch ema", "assigned_samples": assigned, "updated_prototypes": updated}
    return prototype_tokens.detach(), prototype_token_mask.detach(), summary


def evaluate_baseline_with_cutoff(
    model: ECMProtoModel,
    split: dict[str, Tensor],
    batch_size: int,
    device: torch.device,
    cutoff_doy: int,
    cfg: ECMProtoTrainConfig,
) -> dict[str, object]:
    model.eval()
    loader = make_loader(split["x"], split["obs_mask"], split["doy"], split["y"], batch_size, False, 0)
    preds_all = []
    targets_all = []
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for x, obs_mask, doy, y in loader:
            x, obs_mask, doy, y = move_batch_to_device(x, obs_mask, doy, y, device)
            x_cut, mask_cut = apply_doy_cutoff(x, obs_mask, doy, cutoff_doy, keep_first_observation_per_sensor=True)
            logits, _ = baseline_logits(model, x_cut, mask_cut, doy)
            loss = F.cross_entropy(logits.float(), y)
            total += y.size(0)
            total_loss += loss.item() * y.size(0)
            preds_all.append(logits.argmax(dim=1).cpu())
            targets_all.append(y.cpu())
    return compute_classification_metrics(torch.cat(preds_all), torch.cat(targets_all), total_loss, total)


def evaluate_baseline_sweep(
    model: ECMProtoModel,
    split: dict[str, Tensor],
    batch_size: int,
    device: torch.device,
    cfg: ECMProtoTrainConfig,
) -> dict[str, object]:
    results = []
    full_metrics = evaluate_baseline_with_cutoff(model, split, batch_size, device, cfg.early_max_doy, cfg)
    results.append({"cutoff_doy": "full_input_cutoff_270", **full_metrics})
    for cutoff in range(cfg.early_min_doy, cfg.early_max_doy + 1, 10):
        x_cut, mask_cut = apply_doy_cutoff(split["x"], split["obs_mask"], split["doy"], cutoff, True)
        metrics = evaluate_baseline_with_cutoff(model, split, batch_size, device, cutoff, cfg)
        visible_steps = int((split["doy"][0] <= cutoff).sum().item())
        visible_ratio = float((mask_cut.float().sum() / split["obs_mask"].float().sum()).item())
        results.append({"cutoff_doy": cutoff, "visible_steps": visible_steps, "visible_ratio": visible_ratio, **metrics})
    summary = summarize_cutoff_metrics(results, metric_key="accuracy", include_full=False)
    return {
        "metric_key": "accuracy",
        "mean_accuracy": float(summary["mean"]),
        "best_cutoff": int(summary["best_cutoff"]),
        "best_accuracy": float(summary["best_value"]),
        "cutoff_results": results,
        "prototype_results": [],
    }


def evaluate_with_cutoff(
    model: ECMProtoModel,
    split: dict[str, Tensor],
    prototype_tokens: Tensor,
    prototype_token_mask: Tensor,
    batch_size: int,
    device: torch.device,
    cutoff_doy: int,
    cfg: ECMProtoTrainConfig,
) -> tuple[dict[str, object], dict[str, float]]:
    model.eval()
    loader = make_loader(split["x"], split["obs_mask"], split["doy"], split["y"], batch_size, False, 0)
    preds_all = []
    proto_preds_all = []
    targets_all = []
    total_loss = 0.0
    total = 0
    aux_sums = {"proto_ce": 0.0, "attn_loss": 0.0, "true_class_attention": 0.0}
    with torch.no_grad():
        for x, obs_mask, doy, y in loader:
            x = x.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            doy = doy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            x_cut, mask_cut = apply_doy_cutoff(x, obs_mask, doy, cutoff_doy, keep_first_observation_per_sensor=True)
            out = model(
                x_cut,
                mask_cut,
                doy,
                prototype_tokens,
                prototype_token_mask,
                cfg.proto_temperature,
                query_cutoff_doy=cutoff_doy,
            )
            cls_loss = F.cross_entropy(out.logits.float(), y)
            proto_ce = F.cross_entropy(out.class_proto_logits.float(), y)
            attn_loss = prototype_attention_loss(out.prototype_attention, y, cfg)
            class_attention = out.prototype_attention.reshape(y.shape[0], cfg.num_classes, cfg.prototypes_per_class).sum(dim=2)
            total += y.size(0)
            total_loss += cls_loss.item() * y.size(0)
            aux_sums["proto_ce"] += proto_ce.item() * y.size(0)
            aux_sums["attn_loss"] += attn_loss.item() * y.size(0)
            aux_sums["true_class_attention"] += class_attention.gather(1, y.unsqueeze(1)).mean().item() * y.size(0)
            preds_all.append(out.logits.argmax(dim=1).cpu())
            proto_preds_all.append(out.class_proto_logits.argmax(dim=1).cpu())
            targets_all.append(y.cpu())
    targets = torch.cat(targets_all)
    preds = torch.cat(preds_all)
    proto_preds = torch.cat(proto_preds_all)
    metrics = compute_classification_metrics(preds, targets, total_loss, total)
    proto_metrics = compute_classification_metrics(proto_preds, targets)
    aux = {key: value / max(total, 1) for key, value in aux_sums.items()}
    aux["proto_accuracy"] = float((proto_preds == targets).float().mean().item())
    aux["proto_macro_f1"] = float(proto_metrics["macro_f1"])
    return metrics, aux


def evaluate_sweep(
    model: ECMProtoModel,
    split: dict[str, Tensor],
    prototype_tokens: Tensor,
    prototype_token_mask: Tensor,
    batch_size: int,
    device: torch.device,
    cfg: ECMProtoTrainConfig,
) -> dict[str, object]:
    results = []
    aux_results = []
    full_metrics, full_aux = evaluate_with_cutoff(
        model, split, prototype_tokens, prototype_token_mask, batch_size, device, cfg.early_max_doy, cfg
    )
    results.append({"cutoff_doy": "full_input_cutoff_270", **full_metrics})
    aux_results.append({"cutoff_doy": "full_input_cutoff_270", **full_aux})
    for cutoff in range(cfg.early_min_doy, cfg.early_max_doy + 1, 10):
        x_cut, mask_cut = apply_doy_cutoff(split["x"], split["obs_mask"], split["doy"], cutoff, True)
        metrics, aux = evaluate_with_cutoff(model, split, prototype_tokens, prototype_token_mask, batch_size, device, cutoff, cfg)
        visible_steps = int((split["doy"][0] <= cutoff).sum().item())
        visible_ratio = float((mask_cut.float().sum() / split["obs_mask"].float().sum()).item())
        results.append({"cutoff_doy": cutoff, "visible_steps": visible_steps, "visible_ratio": visible_ratio, **metrics})
        aux_results.append({"cutoff_doy": cutoff, **aux})
    summary = summarize_cutoff_metrics(results, metric_key="accuracy", include_full=False)
    return {
        "metric_key": "accuracy",
        "mean_accuracy": float(summary["mean"]),
        "best_cutoff": int(summary["best_cutoff"]),
        "best_accuracy": float(summary["best_value"]),
        "cutoff_results": results,
        "prototype_results": aux_results,
    }


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device, _ = resolve_device()

    out_dir = resolve_experiment_site_dir(cfg) / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / f"{cfg.tag}_train.log")
    logger.log(f"starting ECM-proto config={asdict(cfg)} device={device}")
    logger.log(
        "prototype_source=train full sequence; query=train/test early sequence; "
        f"memory=class multi-prototype tokens; fusion=cross_attention mode={cfg.fusion_mode}"
    )

    prepared = prepare_train_val_test_data(cfg, logger)
    cfg = replace(cfg, num_classes=prepared.num_classes)
    logger.log(
        f"ready train={prepared.train['y'].numel()} val={prepared.val['y'].numel()} test={prepared.test['y'].numel()} "
        f"train_dist={class_distribution(prepared.train['y'])} val_dist={class_distribution(prepared.val['y'])} "
        f"test_dist={class_distribution(prepared.test['y'])}"
    )

    train_loader = make_loader(
        prepared.train["x"], prepared.train["obs_mask"], prepared.train["doy"], prepared.train["y"], cfg.batch_size, True, cfg.num_workers
    )
    model = ECMProtoModel(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
        prototypes_per_class=cfg.prototypes_per_class,
        fusion_mode=cfg.fusion_mode,
        future_query_tokens=cfg.future_query_tokens,
        future_query_step_days=cfg.future_query_step_days,
        future_query_max_doy=cfg.future_query_max_doy,
    ).to(device)
    if cfg.init_checkpoint:
        initialize_from_baseline_checkpoint(model, cfg.init_checkpoint, device, logger)
    effective_warmup_epochs = 0 if cfg.init_checkpoint else cfg.proto_warmup_epochs
    logger.log(f"effective_proto_warmup_epochs={effective_warmup_epochs}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs * len(train_loader), eta_min=cfg.min_learning_rate
    )

    best = {"epoch": -1, "val_mean_accuracy": -1.0}
    best_path = out_dir / f"{cfg.tag}_best.pt"
    history = []
    prototype_history = []
    prototype_tokens: Tensor | None = None
    prototype_token_mask: Tensor | None = None

    for epoch in range(1, cfg.epochs + 1):
        proto_active = epoch > effective_warmup_epochs
        if proto_active and prototype_tokens is None:
            prototype_tokens, prototype_token_mask, proto_summary = build_multi_prototypes(model, prepared.train, cfg, device, logger)
            prototype_history.append({"epoch": epoch, **proto_summary})

        model.train()
        aux_scale = 1.0 if proto_active else 0.0
        totals = {"loss": 0.0, "cls_ce": 0.0, "proto_ce": 0.0, "attn_loss": 0.0, "true_class_attention": 0.0}
        total = 0
        correct = 0
        proto_correct = 0
        for batch_idx, (x, obs_mask, doy, y) in enumerate(train_loader, start=1):
            x, obs_mask, doy, y = move_batch_to_device(x, obs_mask, doy, y, device)
            aug_x, aug_mask, encoded_doy, time_shifts = apply_time_series_augmentation(
                x, obs_mask, doy, cfg.time_shift_max_days, cfg.random_step_drop_prob
            )
            early_mask, cutoffs = sample_random_early_mask(
                doy=doy,
                obs_mask=aug_mask,
                min_doy=cfg.early_min_doy,
                max_doy=cfg.early_max_doy,
                split_doy=cfg.early_split_doy,
                early_sampling_ratio=cfg.early_sampling_ratio,
            )
            early_x = torch.where(early_mask, aug_x, torch.zeros_like(aug_x))
            optimizer.zero_grad(set_to_none=True)
            weights = cutoff_sample_weights(cutoffs, cfg).to(device)
            if proto_active:
                assert prototype_tokens is not None and prototype_token_mask is not None
                out = model(
                    early_x,
                    early_mask,
                    encoded_doy,
                    prototype_tokens,
                    prototype_token_mask,
                    cfg.proto_temperature,
                    query_cutoff_doy=cutoffs,
                )
                logits = out.logits
                proto_ce = F.cross_entropy(out.class_proto_logits.float(), y)
                attn_loss = prototype_attention_loss(out.prototype_attention, y, cfg)
                class_attention = out.prototype_attention.reshape(y.shape[0], cfg.num_classes, cfg.prototypes_per_class).sum(dim=2)
                proto_correct += (out.class_proto_logits.argmax(dim=1) == y).sum().item()
            else:
                logits, _ = baseline_logits(model, early_x, early_mask, encoded_doy)
                proto_ce = logits.new_zeros(())
                attn_loss = logits.new_zeros(())
                class_attention = logits.new_zeros((y.shape[0], cfg.num_classes))
            cls_ce = weighted_ce(logits, y, weights)
            loss = cls_ce + aux_scale * (cfg.proto_loss_weight * proto_ce + cfg.attn_loss_weight * attn_loss)
            loss.backward()
            optimizer.step()
            scheduler.step()
            if proto_active:
                assert prototype_tokens is not None and prototype_token_mask is not None
                prototype_tokens, prototype_token_mask, _ = update_ema_prototypes(
                    model=model,
                    x=x,
                    obs_mask=obs_mask,
                    doy=doy,
                    y=y,
                    prototype_tokens=prototype_tokens,
                    prototype_token_mask=prototype_token_mask,
                    cfg=cfg,
                )

            batch_size_i = y.size(0)
            total += batch_size_i
            totals["loss"] += loss.item() * batch_size_i
            totals["cls_ce"] += cls_ce.item() * batch_size_i
            totals["proto_ce"] += proto_ce.item() * batch_size_i
            totals["attn_loss"] += attn_loss.item() * batch_size_i
            totals["true_class_attention"] += class_attention.gather(1, y.unsqueeze(1)).mean().item() * batch_size_i
            correct += (logits.argmax(dim=1) == y).sum().item()
            if batch_idx == 1 or batch_idx % cfg.log_interval == 0 or batch_idx == len(train_loader):
                logger.log(
                    f"epoch {epoch}/{cfg.epochs} batch {batch_idx}/{len(train_loader)} lr={optimizer.param_groups[0]['lr']:.8f} "
                    f"loss={totals['loss']/max(total,1):.4f} cls={totals['cls_ce']/max(total,1):.4f} "
                    f"proto={totals['proto_ce']/max(total,1):.4f} attn={totals['attn_loss']/max(total,1):.4f} "
                    f"acc={correct/max(total,1):.4f} proto_acc={proto_correct/max(total,1):.4f} "
                    f"true_attn={totals['true_class_attention']/max(total,1):.4f} aux_scale={aux_scale:.1f} "
                    f"cutoff_min={int(cutoffs.min())} cutoff_max={int(cutoffs.max())} shift_mean={float(time_shifts.float().mean()):.2f}"
                )

        if proto_active:
            assert prototype_tokens is not None and prototype_token_mask is not None
            val_eval = evaluate_sweep(model, prepared.val, prototype_tokens, prototype_token_mask, cfg.batch_size, device, cfg)
        else:
            val_eval = evaluate_baseline_sweep(model, prepared.val, cfg.batch_size, device, cfg)
        train_metrics = {key: value / max(total, 1) for key, value in totals.items()}
        train_metrics["accuracy"] = correct / max(total, 1)
        train_metrics["proto_accuracy"] = proto_correct / max(total, 1)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_eval})
        logger.log(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_mean_acc={val_eval['mean_accuracy']:.4f} val_best_cutoff={val_eval['best_cutoff']} aux_scale={aux_scale:.1f}"
        )
        if proto_active and val_eval["mean_accuracy"] > best["val_mean_accuracy"]:
            assert prototype_tokens is not None and prototype_token_mask is not None
            best = {"epoch": epoch, "val_mean_accuracy": float(val_eval["mean_accuracy"])}
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "prototype_tokens": prototype_tokens.detach().cpu(),
                    "prototype_token_mask": prototype_token_mask.detach().cpu(),
                    "prototype_summary": prototype_history[-1] if prototype_history else {},
                },
                best_path,
            )
            logger.log(f"new best checkpoint epoch={epoch} val_mean_acc={val_eval['mean_accuracy']:.4f}")

    if best["epoch"] < 0:
        prototype_tokens, prototype_token_mask, proto_summary = build_multi_prototypes(model, prepared.train, cfg, device, logger)
        prototype_history.append({"epoch": cfg.epochs, **proto_summary})
        best = {"epoch": cfg.epochs, "val_mean_accuracy": -1.0}
        torch.save(
            {
                "model": model.state_dict(),
                "config": asdict(cfg),
                "prototype_tokens": prototype_tokens.detach().cpu(),
                "prototype_token_mask": prototype_token_mask.detach().cpu(),
                "prototype_summary": prototype_history[-1],
            },
            best_path,
        )
        logger.log("saved fallback checkpoint after building prototype bank because no proto-active epoch was run")

    checkpoint = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    prototype_tokens = checkpoint["prototype_tokens"].to(device)
    prototype_token_mask = checkpoint["prototype_token_mask"].to(device)
    final_proto_summary = dict(checkpoint.get("prototype_summary", {}))
    final_proto_summary["source"] = "best checkpoint cached train prototype bank"
    logger.log(f"loaded best checkpoint prototype bank summary={final_proto_summary}")
    test_eval = evaluate_sweep(model, prepared.test, prototype_tokens, prototype_token_mask, cfg.batch_size, device, cfg)
    summary = {
        "config": asdict(cfg),
        "structure": {
            "type": "ECM-proto",
            "encoder": "shared DualBranchS1S2Transformer",
            "prototype_source": "2020-2023 train full sequence",
            "query_source": "random early-truncated train sequence and fixed-cutoff 2024 test sequence",
            "prototype_mode": "multi-prototype per class by kmeans over full-season pooled features",
            "fusion_mode": cfg.fusion_mode,
            "future_query_tokens": int(cfg.future_query_tokens),
            "future_query_step_days": int(cfg.future_query_step_days),
            "future_query_max_doy": int(cfg.future_query_max_doy),
            "query_mode": "valid early tokens plus learnable future query tokens at cutoff+step...cutoff+M*step",
            "prototype_bank_update": (
                f"full train-set initialization after {effective_warmup_epochs} warmup epochs; "
                f"then batch EMA updates with momentum={cfg.prototype_ema_momentum}; "
                "final test loads the cached train prototype bank from the best checkpoint"
            ),
            "future_test_usage": "2024 full sequence is not used; test inputs are cutoff-truncated early views only",
        },
        "prototype_history": prototype_history,
        "final_prototypes": final_proto_summary,
        "train_samples": int(prepared.train["y"].numel()),
        "val_samples": int(prepared.val["y"].numel()),
        "test_samples": int(prepared.test["y"].numel()),
        "train_distribution": class_distribution(prepared.train["y"]),
        "val_distribution": class_distribution(prepared.val["y"]),
        "test_distribution": class_distribution(prepared.test["y"]),
        "best_val": {
            "epoch": int(best["epoch"]),
            "selection_metric": f"val_mean_accuracy_{cfg.early_min_doy}_{cfg.early_max_doy}_step10",
            "mean_accuracy": float(best["val_mean_accuracy"]),
        },
        "final_test": test_eval,
        "history": history,
    }
    summary_path = out_dir / f"{cfg.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.log(f"summary saved to {summary_path}")


if __name__ == "__main__":
    main()
