from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import rasterio

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "config").is_dir() and (parent / "utils").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import project_paths
from utils.h5_timeseries import save_metadata_to_h5, save_split_batch_to_h5
from utils.timeseries_dataset import (
    VALID_OBSERVATION_RULE,
    QualitySelectionStats,
    _build_feature_arrays,
    _compute_sensor_valid_observation_counts,
    _load_sensor_count_cache,
    _save_sensor_count_cache,
    _slice_pixel_batch,
    build_timeline,
    list_scenes,
    map_cdl_to_site_classes,
    resolve_label_path,
)
from utils.training_common import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build no-balance fixed-size quality caches from GeoTIFF files."
    )
    parser.add_argument("--sites", nargs="+", default=["site_IA", "site_OH", "site_AK"])
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--train-count", type=int, default=25000)
    parser.add_argument("--val-count", type=int, default=25000)
    parser.add_argument("--test-count", type=int, default=25000)
    parser.add_argument("--train-top-fraction", type=float, default=0.05)
    parser.add_argument("--val-test-top-fraction", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=None)
    return parser.parse_args()


def class_distribution(y) -> list[int]:
    values = y.numpy()
    max_class = int(values.max()) if values.size else -1
    return [int((values == idx).sum()) for idx in range(max_class + 1)]


def relabeled_same_class_8_neighbor_mask(raw_labels: np.ndarray, mapped_labels: np.ndarray) -> np.ndarray:
    if raw_labels.shape != mapped_labels.shape:
        raise ValueError(f"raw/mapped label shape mismatch: {raw_labels.shape} vs {mapped_labels.shape}")
    raw_valid = raw_labels > 0
    mapped_i64 = mapped_labels.astype(np.int64, copy=False)
    padded_mapped = np.pad(mapped_i64, pad_width=1, mode="constant", constant_values=-9999)
    padded_raw_valid = np.pad(raw_valid, pad_width=1, mode="constant", constant_values=False)
    height, width = mapped_i64.shape
    same = raw_valid.copy()
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            neighbor_mapped = padded_mapped[
                1 + row_offset : 1 + row_offset + height,
                1 + col_offset : 1 + col_offset + width,
            ]
            neighbor_raw_valid = padded_raw_valid[
                1 + row_offset : 1 + row_offset + height,
                1 + col_offset : 1 + col_offset + width,
            ]
            same &= neighbor_raw_valid & (neighbor_mapped == mapped_i64)
    return same


def load_or_compute_counts(
    *,
    cache_dir: Path,
    year: int,
    label_path: Path,
    scenes,
    sensor: str,
    valid_mask: np.ndarray,
    logger: Logger,
) -> np.ndarray:
    counts = _load_sensor_count_cache(
        cache_dir=cache_dir,
        year=year,
        label_path=label_path,
        scenes=scenes,
        sensor=sensor,
    )
    if counts is not None:
        logger.log(f"year={year} loaded cached {sensor} valid observation counts rule={VALID_OBSERVATION_RULE}")
        return counts
    logger.log(
        f"year={year} counting {sensor} valid observations rule={VALID_OBSERVATION_RULE} "
        f"candidate_pixels={int(valid_mask.sum())}"
    )
    counts = _compute_sensor_valid_observation_counts(
        scenes=scenes,
        sensor=sensor,
        label_mask=valid_mask,
        progress_hook=logger.log,
    )
    _save_sensor_count_cache(
        cache_dir=cache_dir,
        year=year,
        label_path=label_path,
        scenes=scenes,
        sensor=sensor,
        counts=counts,
    )
    return counts


def sample_fixed_split(
    *,
    labels: np.ndarray,
    mapped_labels: np.ndarray,
    valid_mask: np.ndarray,
    s2_counts: np.ndarray,
    site: str,
    year: int,
    train_count: int,
    val_count: int,
    test_count: int,
    train_top_fraction: float,
    val_test_top_fraction: float,
    seed: int,
    raster_width: int,
    logger: Logger,
):
    valid_s2_counts = s2_counts[valid_mask].astype(np.float32)
    if valid_s2_counts.size == 0:
        raise ValueError(f"No candidate pixels for site={site} year={year}")
    train_threshold = float(np.quantile(valid_s2_counts, 1.0 - train_top_fraction))
    val_test_threshold = float(np.quantile(valid_s2_counts, 1.0 - val_test_top_fraction))
    train_pool_mask = valid_mask & (s2_counts >= train_threshold)
    val_test_pool_mask = valid_mask & (s2_counts >= val_test_threshold)
    train_pool_coords = np.argwhere(train_pool_mask)
    val_test_pool_coords = np.argwhere(val_test_pool_mask)
    if train_pool_coords.shape[0] < train_count:
        raise ValueError(
            f"Not enough train pixels in top {train_top_fraction:.2%} S2 pool for {site} {year}: "
            f"need={train_count}, pool={train_pool_coords.shape[0]}, threshold={train_threshold}"
        )
    if val_test_pool_coords.shape[0] < train_count + val_count + test_count:
        logger.log(
            f"warning: top {val_test_top_fraction:.2%} val/test pool smaller than all requested before train exclusion: "
            f"pool={val_test_pool_coords.shape[0]} requested_total={train_count + val_count + test_count}"
        )

    rng = np.random.default_rng(seed + year)
    train_idx = rng.choice(train_pool_coords.shape[0], size=train_count, replace=False)
    train_coords = train_pool_coords[train_idx]
    taken = set((train_coords[:, 0].astype(np.int64) * int(raster_width) + train_coords[:, 1].astype(np.int64)).tolist())
    remaining = []
    for row, col in val_test_pool_coords.tolist():
        flat = int(row) * int(raster_width) + int(col)
        if flat not in taken:
            remaining.append((row, col))
    remaining_coords = np.asarray(remaining, dtype=np.int64)
    if remaining_coords.shape[0] < val_count + test_count:
        raise ValueError(
            f"Not enough val/test pixels in top {val_test_top_fraction:.2%} S2 pool after train exclusion for {site} {year}: "
            f"need={val_count + test_count}, remaining={remaining_coords.shape[0]}, threshold={val_test_threshold}"
        )
    val_test_idx = rng.choice(remaining_coords.shape[0], size=val_count + test_count, replace=False)
    val_test_coords = remaining_coords[val_test_idx]
    val_coords = val_test_coords[:val_count]
    test_coords = val_test_coords[val_count:]
    selected_coords = np.concatenate([train_coords, val_coords, test_coords], axis=0)
    selected_y = mapped_labels[selected_coords[:, 0], selected_coords[:, 1]].astype(np.int64, copy=False)
    _, class_names = map_cdl_to_site_classes(labels[selected_coords[:, 0], selected_coords[:, 1]], site)
    train_quality = s2_counts[train_coords[:, 0], train_coords[:, 1]].astype(np.float32)
    logger.log(
        f"year={year} nobalance pools candidate={int(valid_mask.sum())} "
        f"train_top={train_top_fraction:.2%} train_threshold={train_threshold:.4f} train_pool={train_pool_coords.shape[0]} "
        f"val_test_top={val_test_top_fraction:.2%} val_test_threshold={val_test_threshold:.4f} "
        f"val_test_pool={val_test_pool_coords.shape[0]} val_test_after_train={remaining_coords.shape[0]} "
        f"selected train={train_count} val={val_count} test={test_count}"
    )
    stats = QualitySelectionStats(
        candidate_count=int(valid_mask.sum()),
        selected_count=int(train_count),
        train_count=int(train_count),
        val_count=int(val_count),
        test_count=int(test_count),
        quality_mean_candidates=float(valid_s2_counts.mean()),
        quality_mean_selected=float(train_quality.mean()) if train_quality.size else 0.0,
        quality_min_selected=float(train_quality.min()) if train_quality.size else 0.0,
        quality_max_selected=float(train_quality.max()) if train_quality.size else 0.0,
    )
    return selected_coords, selected_y, class_names, stats, {
        "train_threshold": train_threshold,
        "val_test_threshold": val_test_threshold,
        "train_pool_count": int(train_pool_coords.shape[0]),
        "val_test_pool_count": int(val_test_pool_coords.shape[0]),
        "val_test_after_train_count": int(remaining_coords.shape[0]),
    }


def build_year(
    *,
    data_root: Path,
    site: str,
    year: int,
    site_root: Path,
    train_count: int,
    val_count: int,
    test_count: int,
    train_top_fraction: float,
    val_test_top_fraction: float,
    seed: int,
    logger: Logger,
) -> dict[str, object]:
    year_dir = site_root / str(year)
    year_dir.mkdir(parents=True, exist_ok=True)
    site_token = site.split("_", 1)[-1].lower()
    h5_path = year_dir / f"{site_token}_{year}_nobalance_quality_timeseries.h5"
    label_path = resolve_label_path(data_root, site, year)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)
        raster_width = ds.width
    mapped_labels, class_names = map_cdl_to_site_classes(labels, site)
    valid_mask = relabeled_same_class_8_neighbor_mask(labels, mapped_labels)
    total_candidate = int(valid_mask.sum())
    if total_candidate <= 0:
        raise ValueError(f"No relabeled 8-neighbor candidate pixels for site={site} year={year}")
    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    s2_counts = load_or_compute_counts(
        cache_dir=year_dir,
        year=year,
        label_path=label_path,
        scenes=s2_scenes,
        sensor="S2",
        valid_mask=valid_mask,
        logger=logger,
    )
    selected_coords, selected_y, class_names, stats, pool_info = sample_fixed_split(
        labels=labels,
        mapped_labels=mapped_labels,
        valid_mask=valid_mask,
        s2_counts=s2_counts,
        site=site,
        year=year,
        train_count=train_count,
        val_count=val_count,
        test_count=test_count,
        train_top_fraction=train_top_fraction,
        val_test_top_fraction=val_test_top_fraction,
        seed=seed,
        raster_width=raster_width,
        logger=logger,
    )
    timeline = [date_token for date_token, _ in build_timeline(s1_scenes, s2_scenes)]
    logger.log(
        f"year={year} building feature arrays selected={selected_coords.shape[0]} "
        f"timeline={len(timeline)} s1={len(s1_scenes)} s2={len(s2_scenes)}"
    )
    all_x, all_obs_mask, all_doy, _ = _build_feature_arrays(
        coords_rc=selected_coords,
        s1_scenes=s1_scenes,
        s2_scenes=s2_scenes,
        s2_scale=10000.0,
        progress_hook=logger.log,
    )
    if h5_path.exists():
        h5_path.unlink()
        logger.log(f"removed existing cache {h5_path}")
    split_train = _slice_pixel_batch(all_x, all_obs_mask, all_doy, selected_y, selected_coords, class_names, timeline, 0, train_count)
    split_val = _slice_pixel_batch(all_x, all_obs_mask, all_doy, selected_y, selected_coords, class_names, timeline, train_count, train_count + val_count)
    split_test = _slice_pixel_batch(all_x, all_obs_mask, all_doy, selected_y, selected_coords, class_names, timeline, train_count + val_count, train_count + val_count + test_count)
    save_split_batch_to_h5(h5_path, "train", split_train.x, split_train.obs_mask, split_train.doy, split_train.y, split_train.coords)
    save_split_batch_to_h5(h5_path, "val", split_val.x, split_val.obs_mask, split_val.doy, split_val.y, split_val.coords)
    save_split_batch_to_h5(h5_path, "test", split_test.x, split_test.obs_mask, split_test.doy, split_test.y, split_test.coords)
    metadata = {
        "site": site,
        "year": int(year),
        "sample_mode": "nobalance_quality_fixed",
        "valid_observation_rule": VALID_OBSERVATION_RULE,
        "label_neighbor_rule": "relabel_then_same_class_8_neighbor_raw_positive_3x3",
        "train_count": int(train_count),
        "val_count": int(val_count),
        "test_count": int(test_count),
        "train_top_fraction": float(train_top_fraction),
        "val_test_top_fraction": float(val_test_top_fraction),
        "candidate_count": int(total_candidate),
        "train_s2_valid_count_threshold": float(pool_info["train_threshold"]),
        "val_test_s2_valid_count_threshold": float(pool_info["val_test_threshold"]),
        "time_steps": int(all_x.shape[1]),
        "channels": int(all_x.shape[2]),
        "num_classes": int(len(class_names)),
        "class_names_json": json.dumps(class_names),
    }
    save_metadata_to_h5(h5_path, metadata)
    manifest = {
        "site": site,
        "year": int(year),
        "h5_path": str(h5_path),
        "sample_mode": "nobalance_quality_fixed",
        "valid_observation_rule": VALID_OBSERVATION_RULE,
        "label_neighbor_rule": "CDL relabel first, then same mapped class in full 3x3 neighborhood; raw CDL must be positive in the 3x3 neighborhood",
        "class_names": class_names,
        "candidate_count": int(total_candidate),
        "train_samples": int(train_count),
        "val_samples": int(val_count),
        "test_samples": int(test_count),
        "train_distribution": class_distribution(split_train.y),
        "val_distribution": class_distribution(split_val.y),
        "test_distribution": class_distribution(split_test.y),
        "time_steps": int(all_x.shape[1]),
        "channels": int(all_x.shape[2]),
        "timeline": timeline,
        "quality_selection": {
            "applies_to": f"train_top{train_top_fraction:g}_val_test_top{val_test_top_fraction:g}",
            "train_selection_rule": (
                f"uniform random within pixels in the top {train_top_fraction:.0%} by S2 valid observation count"
            ),
            "val_test_selection_rule": (
                f"uniform random within pixels in the top {val_test_top_fraction:.0%} by S2 valid observation count, "
                "excluding train"
            ),
            "train_s2_valid_count_threshold": float(pool_info["train_threshold"]),
            "val_test_s2_valid_count_threshold": float(pool_info["val_test_threshold"]),
            "train_eligible_count": int(pool_info["train_pool_count"]),
            "val_test_eligible_count": int(pool_info["val_test_pool_count"]),
            "val_test_after_train_count": int(pool_info["val_test_after_train_count"]),
            "quality_mean_candidates": stats.quality_mean_candidates,
            "quality_mean_selected": stats.quality_mean_selected,
            "quality_min_selected": stats.quality_min_selected,
            "quality_max_selected": stats.quality_max_selected,
        },
    }
    manifest_path = year_dir / f"{site_token}_{year}_nobalance_quality_timeseries_manifest.json"
    with manifest_path.open("w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2)
    logger.log(
        f"year={year} done h5={h5_path.name} train_dist={manifest['train_distribution']} "
        f"val_dist={manifest['val_distribution']} test_dist={manifest['test_distribution']}"
    )
    return manifest


def main() -> None:
    args = parse_args()
    if min(args.train_count, args.val_count, args.test_count) <= 0:
        raise ValueError("train/val/test counts must be positive")
    if not (0 < args.train_top_fraction <= 1 and 0 < args.val_test_top_fraction <= 1):
        raise ValueError("top fractions must be in (0, 1]")
    paths = project_paths()
    out_root = args.out_root or (paths.root / "Data" / "Data_preprocess" / "nobalance")
    out_root.mkdir(parents=True, exist_ok=True)
    all_summary = []
    for site in args.sites:
        site_root = out_root / site
        site_root.mkdir(parents=True, exist_ok=True)
        root_logger = Logger(site_root / "build_feature_cache_from_tif_nobalance.log")
        root_logger.log(f"building nobalance cache site={site} config={vars(args)} out_root={out_root}")
        site_summary = []
        for year in args.years:
            year_logger = Logger(site_root / str(year) / f"build_{year}_nobalance_quality_feature_cache.log")
            manifest = build_year(
                data_root=paths.raw_data_dir,
                site=site,
                year=year,
                site_root=site_root,
                train_count=args.train_count,
                val_count=args.val_count,
                test_count=args.test_count,
                train_top_fraction=args.train_top_fraction,
                val_test_top_fraction=args.val_test_top_fraction,
                seed=args.seed,
                logger=year_logger,
            )
            site_summary.append(manifest)
            all_summary.append(manifest)
            root_logger.log(
                f"year={year} complete train={manifest['train_samples']} val={manifest['val_samples']} "
                f"test={manifest['test_samples']} h5={manifest['h5_path']}"
            )
        with (site_root / "multiyear_nobalance_quality_manifest.json").open("w", encoding="ascii") as f:
            json.dump({"site": site, "years": site_summary}, f, indent=2)
        root_logger.log("site nobalance cache build complete")
    with (out_root / "all_sites_nobalance_quality_manifest.json").open("w", encoding="ascii") as f:
        json.dump({"sites": args.sites, "years": args.years, "items": all_summary}, f, indent=2)


if __name__ == "__main__":
    main()
