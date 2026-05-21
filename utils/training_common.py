from __future__ import annotations

from typing import Any
from datetime import datetime
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> tuple[torch.device, bool]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    return device, use_cuda


def move_batch_to_device(
    x: Tensor,
    obs_mask: Tensor,
    doy: Tensor,
    y: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        x.to(device, non_blocking=True),
        obs_mask.to(device, non_blocking=True),
        doy.to(device, non_blocking=True),
        y.to(device, non_blocking=True),
    )


def load_checkpoint(path: str | Path, map_location: torch.device | str) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def compute_channel_stats(x: Tensor, obs_mask: Tensor) -> tuple[Tensor, Tensor]:
    mask = obs_mask.float()
    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
    mean = (x * mask).sum(dim=(0, 1)) / denom
    var = (((x - mean.view(1, 1, -1)) * mask) ** 2).sum(dim=(0, 1)) / denom
    std = var.sqrt().clamp_min(1e-6)
    return mean, std


def normalize_batch(x: Tensor, obs_mask: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    x_norm = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    return torch.where(obs_mask, x_norm, torch.zeros_like(x_norm))


def make_loader(
    batch_x: Tensor,
    batch_mask: Tensor,
    batch_doy: Tensor,
    batch_y: Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    ds = TensorDataset(batch_x.float(), batch_mask.bool(), batch_doy.long(), batch_y.long())
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
    )


def select_sensor_channels(batch_x: Tensor, batch_mask: Tensor, sensor_mode: str) -> tuple[Tensor, Tensor]:
    if sensor_mode == "s1s2":
        return batch_x, batch_mask
    if sensor_mode == "s2":
        if batch_x.shape[-1] < 10:
            raise ValueError(f"S2-only mode expects at least 10 channels, got {batch_x.shape[-1]}")
        return batch_x[..., -10:], batch_mask[..., -10:]
    raise ValueError(f"unsupported sensor_mode={sensor_mode}")


def select_sensor_split(split: dict[str, Tensor], sensor_mode: str) -> dict[str, Tensor]:
    batch_x, batch_mask = select_sensor_channels(split["x"], split["obs_mask"], sensor_mode)
    return {
        "x": batch_x,
        "obs_mask": batch_mask,
        "doy": split["doy"],
        "y": split["y"],
        "coords": split["coords"],
    }


def apply_time_series_augmentation(
    x: Tensor,
    obs_mask: Tensor,
    doy: Tensor,
    time_shift_max_days: int = 0,
    random_step_drop_prob: float = 0.0,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    batch_size, steps, _ = x.shape
    aug_doy = doy
    time_shifts = torch.zeros(batch_size, device=doy.device, dtype=doy.dtype)

    if time_shift_max_days > 0:
        time_shifts = torch.randint(
            low=-time_shift_max_days,
            high=time_shift_max_days + 1,
            size=(batch_size,),
            device=doy.device,
            dtype=doy.dtype,
        )
        aug_doy = (doy + time_shifts.unsqueeze(1)).clamp_(1, 366)

    aug_mask = obs_mask
    if random_step_drop_prob > 0:
        keep_steps = torch.rand(batch_size, steps, device=x.device) >= random_step_drop_prob
        drop_mask = keep_steps.unsqueeze(-1)
        if obs_mask.shape[-1] > 10:
            aug_mask = obs_mask.clone()
            aug_mask[:, :, 2:] = obs_mask[:, :, 2:] & drop_mask
        else:
            aug_mask = obs_mask & drop_mask
        empty_rows = ~aug_mask.any(dim=2).any(dim=1)
        if empty_rows.any():
            token_mask = obs_mask.any(dim=-1)
            valid_rows = token_mask.any(dim=1)
            first_valid_idx = token_mask.float().argmax(dim=1)
            restore_rows = torch.nonzero(empty_rows & valid_rows, as_tuple=False).flatten()
            if restore_rows.numel() > 0:
                aug_mask = aug_mask.clone()
                for row_idx in restore_rows.tolist():
                    step_idx = int(first_valid_idx[row_idx].item())
                    aug_mask[row_idx, step_idx] = obs_mask[row_idx, step_idx]

    aug_x = torch.where(aug_mask, x, torch.zeros_like(x))
    return aug_x, aug_mask, aug_doy, time_shifts


def sample_random_early_mask(
    doy: Tensor,
    obs_mask: Tensor,
    min_doy: int,
    max_doy: int,
    split_doy: int | None = None,
    early_sampling_ratio: float = 0.5,
) -> tuple[Tensor, Tensor]:
    if split_doy is None or split_doy < min_doy or split_doy >= max_doy:
        cutoffs = torch.randint(min_doy, max_doy + 1, (doy.shape[0],), device=doy.device)
    else:
        choose_early = torch.rand(doy.shape[0], device=doy.device) < early_sampling_ratio
        early_cutoffs = torch.randint(min_doy, split_doy + 1, (doy.shape[0],), device=doy.device)
        late_cutoffs = torch.randint(split_doy + 1, max_doy + 1, (doy.shape[0],), device=doy.device)
        cutoffs = torch.where(choose_early, early_cutoffs, late_cutoffs)
    keep = doy <= cutoffs.unsqueeze(1)
    early_mask = obs_mask & keep.unsqueeze(-1)
    return early_mask, cutoffs


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    preds_all = []
    targets_all = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, obs_mask, doy, y in loader:
            x = x.to(device, non_blocking=True)
            obs_mask = obs_mask.to(device, non_blocking=True)
            doy = doy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x, obs_mask=obs_mask, doy=doy)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, y)
            pred = logits.argmax(dim=1)

            total_loss += loss.item() * y.size(0)
            total += y.size(0)
            preds_all.append(pred.cpu())
            targets_all.append(y.cpu())

    preds = torch.cat(preds_all)
    targets = torch.cat(targets_all)
    return compute_classification_metrics(
        preds=preds,
        targets=targets,
        total_loss=total_loss,
        total_samples=total,
    )


def compute_classification_metrics(
    preds: Tensor,
    targets: Tensor,
    total_loss: float | None = None,
    total_samples: int | None = None,
) -> dict[str, object]:
    if preds.numel() != targets.numel():
        raise ValueError(f"preds and targets must have the same length, got {preds.numel()} and {targets.numel()}")
    if targets.numel() == 0:
        raise ValueError("cannot compute classification metrics for an empty target tensor")

    num_classes = int(targets.max().item()) + 1
    conf = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(targets, preds):
        conf[t, p] += 1

    recalls = []
    precisions = []
    f1s = []
    for idx in range(num_classes):
        tp = conf[idx, idx].item()
        fn = conf[idx, :].sum().item() - tp
        fp = conf[:, idx].sum().item() - tp
        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)

    metrics: dict[str, object] = {
        "macro_recall": sum(recalls) / len(recalls),
        "macro_precision": sum(precisions) / len(precisions),
        "macro_f1": sum(f1s) / len(f1s),
        "confusion_matrix": conf.tolist(),
    }
    if total_loss is not None and total_samples is not None:
        correct = (preds == targets).sum().item()
        metrics["loss"] = total_loss / max(total_samples, 1)
        metrics["accuracy"] = correct / max(total_samples, 1)
    return metrics


def class_distribution(y: Tensor, num_classes: int | None = None) -> list[int]:
    inferred = int(y.max().item()) + 1 if y.numel() > 0 else 0
    size = max(inferred, num_classes or 0)
    return torch.bincount(y.long(), minlength=size).tolist()


def pad_split(split: dict[str, Tensor], target_steps: int) -> dict[str, Tensor]:
    current_steps = split["x"].shape[1]
    if current_steps == target_steps:
        return split
    pad_steps = target_steps - current_steps
    if pad_steps < 0:
        raise ValueError(f"target_steps={target_steps} smaller than current_steps={current_steps}")

    x_pad = torch.zeros((split["x"].shape[0], pad_steps, split["x"].shape[2]), dtype=split["x"].dtype)
    mask_pad = torch.zeros((split["obs_mask"].shape[0], pad_steps, split["obs_mask"].shape[2]), dtype=split["obs_mask"].dtype)
    doy_last = split["doy"][:, -1:]
    doy_pad = doy_last.expand(-1, pad_steps).clone()
    return {
        "x": torch.cat([split["x"], x_pad], dim=1),
        "obs_mask": torch.cat([split["obs_mask"], mask_pad], dim=1),
        "doy": torch.cat([split["doy"], doy_pad], dim=1),
        "y": split["y"],
        "coords": split["coords"],
    }


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
