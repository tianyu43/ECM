from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from config import ProjectPaths, project_paths
from utils.h5_timeseries import load_split_from_h5
from utils.helper import concat_splits
from utils.training_common import (
    Logger,
    class_distribution,
    compute_channel_stats,
    normalize_batch,
    pad_split,
    select_sensor_split,
)


@dataclass
class PreparedSplitData:
    train: dict[str, Tensor]
    val: dict[str, Tensor]
    test: dict[str, Tensor]
    train_cache_paths: list[str]
    test_cache_paths: list[str]
    test_cache_path: str
    max_steps: int
    channel_mean: Tensor
    channel_std: Tensor
    num_classes: int


def prepare_train_val_test_data(cfg, logger: Logger | None = None) -> PreparedSplitData:
    if os.environ.get("ECM_RESPLIT_ALL_SAMPLES") == "1":
        prepared = prepare_all_samples_train_val_test_data(
            site=cfg.site,
            site_token=cfg.site_token,
            train_years=cfg.train_years,
            test_years=(cfg.test_year,),
            train_fraction=float(os.environ.get("ECM_TRAIN_FRACTION", "0.9")),
            seed=cfg.seed,
            logger=logger,
        )
    else:
        prepared = prepare_multiyear_split_data(
            site=cfg.site,
            site_token=cfg.site_token,
            train_years=cfg.train_years,
            test_year=cfg.test_year,
            train_subset_fraction=cfg.train_subset_fraction,
            val_subset_fraction=cfg.val_subset_fraction,
            test_subset_fraction=cfg.test_subset_fraction,
            logger=logger,
        )
    return PreparedSplitData(
        train=select_sensor_split(prepared.train, cfg.sensor_mode),
        val=select_sensor_split(prepared.val, cfg.sensor_mode),
        test=select_sensor_split(prepared.test, cfg.sensor_mode),
        train_cache_paths=prepared.train_cache_paths,
        test_cache_paths=prepared.test_cache_paths,
        test_cache_path=prepared.test_cache_path,
        max_steps=prepared.max_steps,
        channel_mean=prepared.channel_mean,
        channel_std=prepared.channel_std,
        num_classes=prepared.num_classes,
    )


def _year_cache_path(repo_paths: ProjectPaths, site: str, site_token: str, year: int) -> Path:
    cache_root = os.environ.get("ECM_CACHE_ROOT")
    site_root = Path(cache_root) / site if cache_root else repo_paths.processed_data_dir / site
    template = os.environ.get("ECM_CACHE_FILENAME_TEMPLATE", "{site_token}_{year}_quality_randomsplit_timeseries.h5")
    filename = template.format(site=site, site_token=site_token, year=year)
    return site_root / str(year) / filename


def prepare_multiyear_split_data(
    *,
    site: str,
    site_token: str,
    train_years: tuple[int, ...],
    test_year: int,
    test_years: tuple[int, ...] | None = None,
    train_subset_fraction: float = 1.0,
    val_subset_fraction: float = 1.0,
    test_subset_fraction: float = 1.0,
    logger: Logger | None = None,
    paths: ProjectPaths | None = None,
) -> PreparedSplitData:
    repo_paths = paths or project_paths()
    raw_train_splits = []
    raw_val_splits = []
    raw_test_splits = []
    train_cache_paths: list[str] = []
    test_cache_paths: list[str] = []
    max_steps = 0

    for year in train_years:
        cache_path = _year_cache_path(repo_paths, site, site_token, year)
        train_split = load_split_from_h5(cache_path, "train")
        val_split = load_split_from_h5(cache_path, "val")
        if train_subset_fraction < 1.0:
            subset_idx = _stratified_subset_indices(
                train_split["y"],
                subset_fraction=train_subset_fraction,
                seed=year,
            )
            train_split = _subset_split(train_split, subset_idx)
        if val_subset_fraction < 1.0:
            subset_idx = _stratified_subset_indices(
                val_split["y"],
                subset_fraction=val_subset_fraction,
                seed=year + 10_000,
            )
            val_split = _subset_split(val_split, subset_idx)
        raw_train_splits.append(train_split)
        raw_val_splits.append(val_split)
        train_cache_paths.append(str(cache_path))
        max_steps = max(max_steps, train_split["x"].shape[1], val_split["x"].shape[1])
        if logger is not None:
            logger.log(
                f"loaded year={year} train_shape={tuple(train_split['x'].shape)} val_shape={tuple(val_split['x'].shape)} "
                f"train_dist={class_distribution(train_split['y'])} val_dist={class_distribution(val_split['y'])}"
            )

    resolved_test_years = test_years or (test_year,)
    for year in resolved_test_years:
        test_cache = _year_cache_path(repo_paths, site, site_token, year)
        raw_test = load_split_from_h5(test_cache, "test")
        if test_subset_fraction < 1.0:
            subset_idx = _stratified_subset_indices(
                raw_test["y"],
                subset_fraction=test_subset_fraction,
                seed=year + 20_000,
            )
            raw_test = _subset_split(raw_test, subset_idx)
        raw_test_splits.append(raw_test)
        test_cache_paths.append(str(test_cache))
        max_steps = max(max_steps, raw_test["x"].shape[1])
        if logger is not None:
            logger.log(
                f"loaded test_year={year} test_shape={tuple(raw_test['x'].shape)} "
                f"test_dist={class_distribution(raw_test['y'])} max_steps={max_steps}"
            )

    train = concat_splits([pad_split(item, max_steps) for item in raw_train_splits])
    val = concat_splits([pad_split(item, max_steps) for item in raw_val_splits])
    test = concat_splits([pad_split(item, max_steps) for item in raw_test_splits])
    train = _filter_allowed_labels(train)
    val = _filter_allowed_labels(val)
    test = _filter_allowed_labels(test)
    _validate_non_empty_splits(train=train, val=val, test=test)
    if logger is not None:
        logger.log(
            f"concatenated train_shape={tuple(train['x'].shape)} val_shape={tuple(val['x'].shape)} "
            f"test_shape={tuple(test['x'].shape)}"
        )

    channel_mean, channel_std = compute_channel_stats(train["x"], train["obs_mask"])
    train["x"] = normalize_batch(train["x"], train["obs_mask"], channel_mean, channel_std)
    val["x"] = normalize_batch(val["x"], val["obs_mask"], channel_mean, channel_std)
    test["x"] = normalize_batch(test["x"], test["obs_mask"], channel_mean, channel_std)

    return PreparedSplitData(
        train=train,
        val=val,
        test=test,
        train_cache_paths=train_cache_paths,
        test_cache_paths=test_cache_paths,
        test_cache_path=test_cache_paths[0],
        max_steps=max_steps,
        channel_mean=channel_mean,
        channel_std=channel_std,
        num_classes=int(train["y"].max().item()) + 1,
    )


def _subset_split(split: dict[str, Tensor], indices: Tensor) -> dict[str, Tensor]:
    return {
        "x": split["x"][indices],
        "obs_mask": split["obs_mask"][indices],
        "doy": split["doy"][indices],
        "y": split["y"][indices],
        "coords": split["coords"][indices],
    }


def _validate_non_empty_splits(**splits: dict[str, Tensor]) -> None:
    empty = [name for name, split in splits.items() if split["y"].numel() == 0]
    if empty:
        raise ValueError(f"empty split after loading/filtering: {', '.join(empty)}")


def _filter_allowed_labels(split: dict[str, Tensor]) -> dict[str, Tensor]:
    raw = os.environ.get("ECM_ALLOWED_LABELS")
    if not raw:
        return split
    allowed = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not allowed:
        return split
    keep = torch.zeros_like(split["y"], dtype=torch.bool)
    for label in allowed:
        keep |= split["y"] == label
    filtered = _subset_split(split, torch.nonzero(keep, as_tuple=False).flatten())
    mapping = {label: idx for idx, label in enumerate(sorted(allowed))}
    remapped = filtered["y"].clone()
    for old, new in mapping.items():
        remapped[filtered["y"] == old] = new
    filtered["y"] = remapped
    return filtered


def _stratified_subset_indices(labels: Tensor, subset_fraction: float, seed: int) -> Tensor:
    if subset_fraction >= 1.0:
        return torch.arange(labels.shape[0], dtype=torch.long)
    if subset_fraction <= 0.0:
        raise ValueError(f"subset_fraction must be > 0, got {subset_fraction}")

    rng = torch.Generator()
    rng.manual_seed(seed)
    parts: list[Tensor] = []
    for class_id in sorted(labels.unique().tolist()):
        class_idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        perm = class_idx[torch.randperm(class_idx.numel(), generator=rng)]
        keep_n = max(1, int(round(class_idx.numel() * subset_fraction)))
        keep_n = min(keep_n, class_idx.numel())
        parts.append(perm[:keep_n])
    subset_idx = torch.cat(parts) if parts else torch.empty(0, dtype=torch.long)
    if subset_idx.numel() > 0:
        subset_idx = subset_idx[torch.randperm(subset_idx.numel(), generator=rng)]
    return subset_idx


def _stratified_train_val_indices(labels: Tensor, train_fraction: float, seed: int) -> tuple[Tensor, Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    train_parts: list[Tensor] = []
    val_parts: list[Tensor] = []
    for class_id in sorted(labels.unique().tolist()):
        class_idx = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        perm = class_idx[torch.randperm(class_idx.numel(), generator=rng)]
        train_n = int(round(class_idx.numel() * train_fraction))
        train_n = min(max(train_n, 1), class_idx.numel() - 1) if class_idx.numel() > 1 else class_idx.numel()
        train_parts.append(perm[:train_n])
        val_parts.append(perm[train_n:])
    train_idx = torch.cat(train_parts) if train_parts else torch.empty(0, dtype=torch.long)
    val_idx = torch.cat(val_parts) if val_parts else torch.empty(0, dtype=torch.long)
    if train_idx.numel() > 0:
        train_idx = train_idx[torch.randperm(train_idx.numel(), generator=rng)]
    if val_idx.numel() > 0:
        val_idx = val_idx[torch.randperm(val_idx.numel(), generator=rng)]
    return train_idx, val_idx


def prepare_all_samples_train_val_test_data(
    *,
    site: str,
    site_token: str,
    train_years: tuple[int, ...],
    test_years: tuple[int, ...],
    train_fraction: float,
    seed: int,
    logger: Logger | None = None,
    paths: ProjectPaths | None = None,
) -> PreparedSplitData:
    repo_paths = paths or project_paths()
    train_year_splits: list[dict[str, Tensor]] = []
    test_year_splits: list[dict[str, Tensor]] = []
    train_cache_paths: list[str] = []
    test_cache_paths: list[str] = []
    max_steps = 0

    for year in train_years:
        cache_path = _year_cache_path(repo_paths, site, site_token, year)
        year_splits = [load_split_from_h5(cache_path, split_name) for split_name in ("train", "val", "test")]
        merged = concat_splits(year_splits)
        train_year_splits.append(merged)
        train_cache_paths.append(str(cache_path))
        max_steps = max(max_steps, merged["x"].shape[1])
        if logger is not None:
            logger.log(
                f"loaded all samples year={year} shape={tuple(merged['x'].shape)} "
                f"dist={class_distribution(merged['y'])}"
            )

    for year in test_years:
        cache_path = _year_cache_path(repo_paths, site, site_token, year)
        year_splits = [load_split_from_h5(cache_path, split_name) for split_name in ("train", "val", "test")]
        merged = concat_splits(year_splits)
        test_year_splits.append(merged)
        test_cache_paths.append(str(cache_path))
        max_steps = max(max_steps, merged["x"].shape[1])
        if logger is not None:
            logger.log(
                f"loaded all test samples year={year} shape={tuple(merged['x'].shape)} "
                f"dist={class_distribution(merged['y'])}"
            )

    all_train = concat_splits([pad_split(item, max_steps) for item in train_year_splits])
    all_test = concat_splits([pad_split(item, max_steps) for item in test_year_splits])
    train_idx, val_idx = _stratified_train_val_indices(all_train["y"], train_fraction=train_fraction, seed=seed)
    train = _subset_split(all_train, train_idx)
    val = _subset_split(all_train, val_idx)
    test = all_test
    train = _filter_allowed_labels(train)
    val = _filter_allowed_labels(val)
    test = _filter_allowed_labels(test)
    _validate_non_empty_splits(train=train, val=val, test=test)
    if logger is not None:
        logger.log(
            f"resplit all-train samples train_shape={tuple(train['x'].shape)} val_shape={tuple(val['x'].shape)} "
            f"test_shape={tuple(test['x'].shape)}"
        )

    channel_mean, channel_std = compute_channel_stats(train["x"], train["obs_mask"])
    train["x"] = normalize_batch(train["x"], train["obs_mask"], channel_mean, channel_std)
    val["x"] = normalize_batch(val["x"], val["obs_mask"], channel_mean, channel_std)
    test["x"] = normalize_batch(test["x"], test["obs_mask"], channel_mean, channel_std)

    return PreparedSplitData(
        train=train,
        val=val,
        test=test,
        train_cache_paths=train_cache_paths,
        test_cache_paths=test_cache_paths,
        test_cache_path=test_cache_paths[0],
        max_steps=max_steps,
        channel_mean=channel_mean,
        channel_std=channel_std,
        num_classes=int(train["y"].max().item()) + 1,
    )
