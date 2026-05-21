from __future__ import annotations

from pathlib import Path

import h5py
import torch
from torch import Tensor, nn

from utils.training_common import evaluate, make_loader


def concat_splits(splits: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    return {
        "x": torch.cat([item["x"] for item in splits], dim=0),
        "obs_mask": torch.cat([item["obs_mask"] for item in splits], dim=0),
        "doy": torch.cat([item["doy"] for item in splits], dim=0),
        "y": torch.cat([item["y"] for item in splits], dim=0),
        "coords": torch.cat([item["coords"] for item in splits], dim=0),
    }


def apply_doy_cutoff(
    x: Tensor,
    obs_mask: Tensor,
    doy: Tensor,
    cutoff: int,
    keep_first_observation_per_sensor: bool = False,
) -> tuple[Tensor, Tensor]:
    keep = (doy <= cutoff).unsqueeze(-1)
    truncated_mask = obs_mask & keep
    if keep_first_observation_per_sensor:
        if obs_mask.shape[-1] == 12:
            sensor_slices = [(0, 2), (2, obs_mask.shape[-1])]
        else:
            sensor_slices = [(0, obs_mask.shape[-1])]
        for start, end in sensor_slices:
            sensor_full = obs_mask[:, :, start:end]
            sensor_truncated = truncated_mask[:, :, start:end]
            empty_rows = ~sensor_truncated.any(dim=2).any(dim=1)
            if empty_rows.any():
                token_mask = sensor_full.any(dim=-1)
                first_valid_idx = token_mask.float().argmax(dim=1)
                valid_sensor_rows = token_mask.any(dim=1)
                for row_idx in torch.nonzero(empty_rows & valid_sensor_rows, as_tuple=False).flatten():
                    step_idx = int(first_valid_idx[row_idx].item())
                    truncated_mask[row_idx, step_idx, start:end] = obs_mask[row_idx, step_idx, start:end]
    truncated_x = torch.where(truncated_mask, x, torch.zeros_like(x))
    return truncated_x, truncated_mask


def evaluate_truncated_model(
    model: nn.Module,
    split: dict[str, Tensor],
    batch_size: int,
    device: torch.device,
    cutoffs: list[int] | None = None,
    keep_first_observation_per_sensor: bool = False,
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    full_loader = make_loader(split["x"], split["obs_mask"], split["doy"], split["y"], batch_size, False, 0)
    full_metrics = evaluate(model, full_loader, device)
    results.append({"cutoff_doy": "full", **full_metrics})

    for cutoff in cutoffs or list(range(150, 271, 10)):
        x_cut, mask_cut = apply_doy_cutoff(
            split["x"],
            split["obs_mask"],
            split["doy"],
            cutoff,
            keep_first_observation_per_sensor=keep_first_observation_per_sensor,
        )
        loader = make_loader(x_cut, mask_cut, split["doy"], split["y"], batch_size, False, 0)
        metrics = evaluate(model, loader, device)
        visible_steps = int((split["doy"][0] <= cutoff).sum().item())
        visible_ratio = float((mask_cut.float().sum() / split["obs_mask"].float().sum()).item())
        results.append(
            {
                "cutoff_doy": cutoff,
                "visible_steps": visible_steps,
                "visible_ratio": visible_ratio,
                **metrics,
            }
        )
    return results


def summarize_cutoff_metrics(
    results: list[dict[str, float | int | str]],
    metric_key: str,
    include_full: bool = False,
) -> dict[str, object]:
    selected = [
        item for item in results
        if include_full or isinstance(item["cutoff_doy"], int)
    ]
    values = [float(item[metric_key]) for item in selected]
    cutoffs = [item["cutoff_doy"] for item in selected]
    if not values:
        raise ValueError("no cutoff results available for summarization")
    best_idx = max(range(len(values)), key=lambda idx: values[idx])
    return {
        "metric_key": metric_key,
        "include_full": include_full,
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "best_cutoff": cutoffs[best_idx],
        "best_value": values[best_idx],
        "num_cutoffs": len(values),
    }


def score_truncated_results(results: list[dict[str, float | int | str]]) -> dict[str, float]:
    by_cutoff = {item["cutoff_doy"]: item for item in results}
    doy150 = float(by_cutoff[150]["macro_f1"])
    doy180 = float(by_cutoff[180]["macro_f1"])
    doy210 = float(by_cutoff[210]["macro_f1"])
    full_f1 = float(by_cutoff["full"]["macro_f1"])
    early_mean = (doy150 + doy180 + doy210) / 3.0
    overall = 0.8 * early_mean + 0.2 * full_f1
    return {
        "early_mean_f1": early_mean,
        "overall_score": overall,
        "full_f1": full_f1,
        "doy150_f1": doy150,
        "doy180_f1": doy180,
        "doy210_f1": doy210,
    }


def load_history_split(history_h5_path: Path, split_key: str) -> dict[int, dict[str, Tensor]]:
    out: dict[int, dict[str, Tensor]] = {}
    with h5py.File(history_h5_path, "r") as f:
        hist_root = f[split_key]["history"]
        for year in hist_root.keys():
            grp = hist_root[year]
            out[int(year)] = {
                "x": torch.from_numpy(grp["x"][:]),
                "obs_mask": torch.from_numpy(grp["obs_mask"][:]),
                "doy": torch.from_numpy(grp["doy"][:]),
            }
    return out
