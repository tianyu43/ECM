from __future__ import annotations

try:
    from train._bootstrap import ensure_repo_root_on_path
except ModuleNotFoundError:
    from _bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from config import ECMBaselineTrainConfig, MODEL_SCALE_PRESETS, apply_model_scale
from model.ecm_baseline import ECMBaselineModel
from utils.data_pipeline import prepare_train_val_test_data
from utils.helper import apply_doy_cutoff, evaluate_truncated_model, summarize_cutoff_metrics
from utils.training_common import (
    Logger,
    apply_time_series_augmentation,
    class_distribution,
    evaluate,
    load_checkpoint,
    make_loader,
    move_batch_to_device,
    resolve_device,
    sample_random_early_mask,
    seed_everything,
)


def parse_args() -> ECMBaselineTrainConfig:
    defaults = ECMBaselineTrainConfig()
    parser = argparse.ArgumentParser(description="Train ECM baseline with early-view supervision.")
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
    parser.add_argument("--min-learning-rate", type=float, default=defaults.min_learning_rate)
    parser.add_argument("--time-shift-max-days", type=int, default=defaults.time_shift_max_days)
    parser.add_argument("--random-step-drop-prob", type=float, default=defaults.random_step_drop_prob)
    parser.add_argument("--train-subset-fraction", type=float, default=defaults.train_subset_fraction)
    parser.add_argument("--val-subset-fraction", type=float, default=defaults.val_subset_fraction)
    parser.add_argument("--test-subset-fraction", type=float, default=defaults.test_subset_fraction)
    parser.add_argument("--init-checkpoint", default=defaults.init_checkpoint)
    parser.add_argument("--early-cutoff-weight", type=float, default=defaults.early_cutoff_weight)
    parser.add_argument("--mid-cutoff-weight", type=float, default=defaults.mid_cutoff_weight)
    parser.add_argument("--late-cutoff-weight", type=float, default=defaults.late_cutoff_weight)
    parser.add_argument("--train-all-fixed-cutoffs", action="store_true", default=defaults.train_all_fixed_cutoffs)
    parser.add_argument("--fixed-cutoff-step", type=int, default=defaults.fixed_cutoff_step)
    args = parser.parse_args()

    cfg = ECMBaselineTrainConfig(
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
        min_learning_rate=args.min_learning_rate,
        time_shift_max_days=args.time_shift_max_days,
        random_step_drop_prob=args.random_step_drop_prob,
        train_subset_fraction=args.train_subset_fraction,
        val_subset_fraction=args.val_subset_fraction,
        test_subset_fraction=args.test_subset_fraction,
        init_checkpoint=args.init_checkpoint,
        early_cutoff_weight=args.early_cutoff_weight,
        mid_cutoff_weight=args.mid_cutoff_weight,
        late_cutoff_weight=args.late_cutoff_weight,
        train_all_fixed_cutoffs=args.train_all_fixed_cutoffs,
        fixed_cutoff_step=args.fixed_cutoff_step,
    )
    return apply_model_scale(cfg, cfg.model_scale)


def resolve_experiment_site_dir(cfg: ECMBaselineTrainConfig) -> Path:
    base_dir = cfg.experiment_site_dir()
    if cfg.experiment_site_dirname is None:
        return base_dir
    return base_dir.parent / cfg.experiment_site_dirname


def make_early_view_loader(split: dict[str, torch.Tensor], batch_size: int, cutoff_doy: int, num_workers: int):
    early_x, early_mask = apply_doy_cutoff(
        split["x"],
        split["obs_mask"],
        split["doy"],
        cutoff_doy,
        keep_first_observation_per_sensor=True,
    )
    return make_loader(early_x, early_mask, split["doy"], split["y"], batch_size, False, num_workers)


def evaluate_fixed_early_sweep(
    model: nn.Module,
    split: dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    min_doy: int,
    max_doy: int,
) -> dict[str, object]:
    cutoffs = list(range(min_doy, max_doy + 1, 10))
    results = evaluate_truncated_model(
        model=model,
        split=split,
        batch_size=batch_size,
        device=device,
        cutoffs=cutoffs,
        keep_first_observation_per_sensor=True,
    )
    summary = summarize_cutoff_metrics(results, metric_key="accuracy", include_full=False)
    return {
        "metric_key": "accuracy",
        "mean_accuracy": float(summary["mean"]),
        "best_cutoff": int(summary["best_cutoff"]),
        "best_accuracy": float(summary["best_value"]),
        "cutoff_results": results,
    }


def cutoff_sample_weights(cutoffs: torch.Tensor, cfg: ECMBaselineTrainConfig) -> torch.Tensor:
    weights = torch.full_like(cutoffs, cfg.late_cutoff_weight, dtype=torch.float32)
    weights = torch.where(cutoffs <= 190, torch.full_like(weights, cfg.early_cutoff_weight), weights)
    weights = torch.where((cutoffs > 190) & (cutoffs <= 230), torch.full_like(weights, cfg.mid_cutoff_weight), weights)
    return weights / weights.mean().clamp_min(1e-6)


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    losses = F.cross_entropy(logits.float(), y, reduction="none")
    return (losses * weights).mean()


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)
    device, use_cuda = resolve_device()

    out_dir = resolve_experiment_site_dir(cfg) / cfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / f"{cfg.tag}_train.log")
    logger.log(f"starting ECM-baseline config={asdict(cfg)} device={device} cuda={use_cuda}")
    logger.log("doy_shift_policy=encode_only mask_doy=true_doy model_doy=shifted_doy")

    prepared = prepare_train_val_test_data(cfg, logger)
    logger.log(
        f"ready train={prepared.train['y'].numel()} val={prepared.val['y'].numel()} test={prepared.test['y'].numel()} "
        f"train_dist={class_distribution(prepared.train['y'])} val_dist={class_distribution(prepared.val['y'])} "
        f"test_dist={class_distribution(prepared.test['y'])}"
    )

    cfg = replace(cfg, num_classes=prepared.num_classes)
    eval_early_cutoff = cfg.early_max_doy
    train_loader = make_loader(
        prepared.train["x"], prepared.train["obs_mask"], prepared.train["doy"], prepared.train["y"], cfg.batch_size, True, cfg.num_workers
    )
    test_loader = make_loader(
        prepared.test["x"], prepared.test["obs_mask"], prepared.test["doy"], prepared.test["y"], cfg.batch_size, False, 0
    )
    test_loader_early = make_early_view_loader(prepared.test, cfg.batch_size, eval_early_cutoff, 0)

    model = ECMBaselineModel(
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_layers=cfg.num_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        num_classes=cfg.num_classes,
    ).to(device)
    if cfg.init_checkpoint:
        checkpoint = load_checkpoint(cfg.init_checkpoint, map_location=device)
        model.load_state_dict(checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint)))
        logger.log(f"initialized from checkpoint={cfg.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs * len(train_loader),
        eta_min=cfg.min_learning_rate,
    )

    best = {"epoch": -1, "val_mean_accuracy": -1.0}
    history = []
    best_path = out_dir / f"{cfg.tag}_best.pt"

    for epoch in range(1, cfg.epochs + 1):
        logger.log(f"epoch {epoch}/{cfg.epochs} train start")
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        num_batches = len(train_loader)

        for batch_idx, (x, obs_mask, doy, y) in enumerate(train_loader, start=1):
            x, obs_mask, doy, y = move_batch_to_device(x, obs_mask, doy, y, device)
            aug_x, aug_mask, encoded_doy, time_shifts = apply_time_series_augmentation(
                x=x,
                obs_mask=obs_mask,
                doy=doy,
                time_shift_max_days=cfg.time_shift_max_days,
                random_step_drop_prob=cfg.random_step_drop_prob,
            )
            optimizer.zero_grad(set_to_none=True)
            if cfg.train_all_fixed_cutoffs:
                cutoff_values = list(range(cfg.early_min_doy, cfg.early_max_doy + 1, cfg.fixed_cutoff_step))
                loss = y.new_zeros((), dtype=torch.float32)
                batch_correct = 0
                batch_total = 0
                cutoff_min = cutoff_values[0]
                cutoff_max = cutoff_values[-1]
                for cutoff_value in cutoff_values:
                    cutoffs = torch.full((y.shape[0],), cutoff_value, device=device, dtype=doy.dtype)
                    keep = doy <= cutoffs.unsqueeze(1)
                    early_mask = aug_mask & keep.unsqueeze(-1)
                    early_x = torch.where(early_mask, aug_x, torch.zeros_like(aug_x))
                    logits = model(early_x, obs_mask=early_mask, doy=encoded_doy)
                    weights = cutoff_sample_weights(cutoffs, cfg).to(device)
                    cutoff_loss = weighted_ce(logits, y, weights) / len(cutoff_values)
                    cutoff_loss.backward()
                    loss = loss + cutoff_loss.detach()
                    batch_correct += (logits.argmax(dim=1) == y).sum().item()
                    batch_total += y.size(0)
            else:
                early_mask, cutoffs = sample_random_early_mask(
                    doy=doy,
                    obs_mask=aug_mask,
                    min_doy=cfg.early_min_doy,
                    max_doy=cfg.early_max_doy,
                    split_doy=cfg.early_split_doy,
                    early_sampling_ratio=cfg.early_sampling_ratio,
                )
                early_x = torch.where(early_mask, aug_x, torch.zeros_like(aug_x))
                logits = model(early_x, obs_mask=early_mask, doy=encoded_doy)
                weights = cutoff_sample_weights(cutoffs, cfg).to(device)
                loss = weighted_ce(logits, y, weights)
                loss.backward()
                batch_correct = (logits.argmax(dim=1) == y).sum().item()
                batch_total = y.size(0)
                cutoff_min = int(cutoffs.min())
                cutoff_max = int(cutoffs.max())
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * batch_total
            total += batch_total
            correct += batch_correct

            if batch_idx == 1 or batch_idx % cfg.log_interval == 0 or batch_idx == num_batches:
                logger.log(
                    f"epoch {epoch} train batch {batch_idx}/{num_batches} "
                    f"lr={optimizer.param_groups[0]['lr']:.8f} loss={total_loss / max(total, 1):.4f} "
                    f"acc={correct / max(total, 1):.4f} cutoff_min={cutoff_min} "
                    f"cutoff_max={cutoff_max} shift_mean={float(time_shifts.float().mean()):.2f}"
                )

        val_early = evaluate_fixed_early_sweep(model, prepared.val, cfg.batch_size, device, cfg.early_min_doy, cfg.early_max_doy)
        train_metrics = {"loss": total_loss / max(total, 1), "accuracy": correct / max(total, 1)}
        history.append({"epoch": epoch, "train": train_metrics, "val_early_fixed": val_early})
        logger.log(
            f"epoch={epoch} lr={optimizer.param_groups[0]['lr']:.8f} train_loss={train_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_early_mean_acc={val_early['mean_accuracy']:.4f} "
            f"val_early_best_cutoff={val_early['best_cutoff']} val_early_best_acc={val_early['best_accuracy']:.4f}"
        )

        if val_early["mean_accuracy"] > best["val_mean_accuracy"]:
            best["epoch"] = epoch
            best["val_mean_accuracy"] = val_early["mean_accuracy"]
            torch.save({"model": model.state_dict(), "config": asdict(cfg)}, best_path)
            logger.log(f"new best checkpoint epoch={epoch} val_early_mean_acc={val_early['mean_accuracy']:.4f}")

    checkpoint = load_checkpoint(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    logger.log("best checkpoint loaded, starting final test evaluation")

    test_full = evaluate(model, test_loader, device)
    test_early = evaluate(model, test_loader_early, device)
    truncated_results = evaluate_truncated_model(
        model=model,
        split=prepared.test,
        batch_size=cfg.batch_size,
        device=device,
        cutoffs=list(range(cfg.early_min_doy, cfg.early_max_doy + 1, 10)),
        keep_first_observation_per_sensor=True,
    )

    logger.log(f"best_epoch={best['epoch']}")
    logger.log(f"best_val_early_mean_accuracy={best['val_mean_accuracy']:.6f}")
    logger.log(f"final_test_full_macro_f1={test_full['macro_f1']:.6f}")
    logger.log(f"final_test_early_macro_f1@{eval_early_cutoff}={test_early['macro_f1']:.6f}")

    summary = {
        "config": asdict(cfg),
        "structure": {
            "type": "ECM-baseline",
            "encoder": "DualBranchS1S2Transformer",
            "classifier_input": "backbone_features_from_early_view",
        },
        "train_cache_paths": prepared.train_cache_paths,
        "test_cache": prepared.test_cache_path,
        "test_cache_paths": prepared.test_cache_paths,
        "train_samples": int(prepared.train["y"].numel()),
        "val_samples": int(prepared.val["y"].numel()),
        "test_samples": int(prepared.test["y"].numel()),
        "train_distribution": class_distribution(prepared.train["y"]),
        "val_distribution": class_distribution(prepared.val["y"]),
        "test_distribution": class_distribution(prepared.test["y"]),
        "max_steps": int(prepared.train["x"].shape[1]),
        "best_val": {
            "epoch": int(best["epoch"]),
            "selection_metric": f"val_early_mean_accuracy_{cfg.early_min_doy}_{cfg.early_max_doy}_step10",
            "mean_accuracy": float(best["val_mean_accuracy"]),
        },
        "final_test_full": test_full,
        "final_test_early": test_early,
        "final_test_early_cutoff_doy": int(eval_early_cutoff),
        "truncated_results": truncated_results,
        "history": history,
    }
    summary_path = out_dir / f"{cfg.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
