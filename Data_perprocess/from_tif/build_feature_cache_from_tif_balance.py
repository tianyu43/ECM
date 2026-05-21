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
    QualitySelectionStats,
    _build_feature_arrays,
    _compute_sensor_valid_observation_counts,
    _load_sensor_count_cache,
    _save_sensor_count_cache,
    _slice_pixel_batch,
    build_timeline,
    cdl_same_class_8_neighbor_mask,
    list_scenes,
    map_cdl_to_site_classes,
    resolve_confidence_path,
    resolve_label_path,
)
from utils.training_common import Logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build class-balanced site/year feature caches from GeoTIFF files."
    )
    parser.add_argument("--site", default="site_IA")
    parser.add_argument("--years", nargs="+", type=int, default=[2020, 2021, 2022, 2023, 2024])
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--allowed-labels",
        nargs="*",
        type=int,
        default=[],
        help="Optional raw-label filter. Defaults to all labels > 0, preserving the site others class when configured.",
    )
    parser.add_argument("--top-fraction", type=float, default=0.25)
    parser.add_argument("--per-class-train", type=int, default=4500)
    parser.add_argument("--per-class-val", type=int, default=500)
    parser.add_argument("--per-class-test", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path, default=None, help="Processed data root. Defaults to ECM_PROCESSED_DATA_DIR/from_tif/site.")
    return parser.parse_args()


def class_distribution(y) -> list[int]:
    values = y.numpy()
    max_class = int(values.max()) if values.size else -1
    return [int((values == idx).sum()) for idx in range(max_class + 1)]


def _load_or_compute_counts(
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
        logger.log(f"year={year} loaded cached {sensor} valid observation counts")
        return counts

    logger.log(f"year={year} counting {sensor} valid observations total_valid={int(valid_mask.sum())}")
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


def _sample_balanced_positions(
    *,
    pool_y: np.ndarray,
    class_ids: np.ndarray,
    per_class_count: int,
    rng: np.random.Generator,
    pool_name: str,
) -> np.ndarray:
    if per_class_count < 0:
        raise ValueError(f"{pool_name} per-class count must be non-negative, got {per_class_count}")
    if per_class_count == 0:
        return np.empty(0, dtype=np.int64)

    parts: list[np.ndarray] = []
    for class_id in class_ids.tolist():
        class_positions = np.flatnonzero(pool_y == class_id)
        if class_positions.size < per_class_count:
            raise ValueError(
                f"Not enough {pool_name} samples for class={class_id}: "
                f"required={per_class_count}, found={class_positions.size}"
            )
        parts.append(rng.choice(class_positions, size=per_class_count, replace=False))

    selected = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    if selected.size:
        selected = selected[rng.permutation(selected.size)]
    return selected


def _coords_to_flat(coords: np.ndarray, width: int) -> np.ndarray:
    return coords[:, 0].astype(np.int64) * int(width) + coords[:, 1].astype(np.int64)


def _filter_out_flat(coords: np.ndarray, y: np.ndarray, taken_flat: set[int], width: int) -> tuple[np.ndarray, np.ndarray]:
    flat = _coords_to_flat(coords, width)
    keep = np.fromiter((int(item) not in taken_flat for item in flat), dtype=bool, count=flat.size)
    return coords[keep], y[keep]


def _build_and_save_year(
    *,
    data_root: Path,
    site: str,
    year: int,
    cache_dir: Path,
    h5_path: Path,
    allowed_labels: tuple[int, ...],
    min_confidence: float,
    top_fraction: float,
    train_per_class: int,
    val_per_class: int,
    test_per_class: int,
    seed: int,
    logger: Logger,
) -> dict[str, object]:
    label_path = resolve_label_path(data_root, site, year)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)
        raster_width = ds.width

    valid_mask = (labels > 0) & cdl_same_class_8_neighbor_mask(labels)
    if allowed_labels:
        valid_mask &= np.isin(labels, np.asarray(allowed_labels, dtype=labels.dtype))
    if min_confidence > 0:
        conf_path = resolve_confidence_path(data_root, site, year)
        with rasterio.open(conf_path) as ds:
            confidence = ds.read(1)
        valid_mask &= confidence > float(min_confidence)

    total_valid = int(valid_mask.sum())
    if total_valid <= 0:
        raise ValueError(f"No valid pixels found for site={site} year={year}")

    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    s2_counts = _load_or_compute_counts(
        cache_dir=cache_dir,
        year=year,
        label_path=label_path,
        scenes=s2_scenes,
        sensor="S2",
        valid_mask=valid_mask,
        logger=logger,
    )

    valid_labels = labels[valid_mask]
    valid_y, class_names = map_cdl_to_site_classes(valid_labels, site)
    class_ids = np.arange(len(class_names), dtype=np.int64)

    valid_s2_counts = s2_counts[valid_mask].astype(np.float32)
    train_threshold = float(np.quantile(valid_s2_counts, 1.0 - float(top_fraction)))
    val_test_threshold = float(np.quantile(valid_s2_counts, 0.25))
    train_pool_mask = valid_mask & (s2_counts >= train_threshold)
    val_test_pool_mask = valid_mask & (s2_counts >= val_test_threshold)
    train_pool_coords = np.argwhere(train_pool_mask)
    train_pool_raw_labels = labels[train_pool_mask]
    train_pool_y, _ = map_cdl_to_site_classes(train_pool_raw_labels, site)
    val_test_pool_coords = np.argwhere(val_test_pool_mask)
    val_test_pool_raw_labels = labels[val_test_pool_mask]
    val_test_pool_y, _ = map_cdl_to_site_classes(val_test_pool_raw_labels, site)

    logger.log(
        f"year={year} balanced quality pool total_valid={total_valid} "
        f"train_s2_valid_count_threshold={train_threshold:.4f} "
        f"val_test_s2_valid_count_threshold={val_test_threshold:.4f} "
        f"eligible_train_top25={train_pool_coords.shape[0]} "
        f"eligible_val_test_top75={val_test_pool_coords.shape[0]} "
        f"class_names={class_names} allowed_labels={list(allowed_labels)} min_confidence={min_confidence:.2f}"
    )

    rng = np.random.default_rng(seed + year)
    train_pos = _sample_balanced_positions(
        pool_y=train_pool_y,
        class_ids=class_ids,
        per_class_count=train_per_class,
        rng=rng,
        pool_name="train top-25% S2 quality-pool",
    )
    train_coords = train_pool_coords[train_pos]
    train_y = train_pool_y[train_pos]
    train_quality = s2_counts[train_coords[:, 0], train_coords[:, 1]].astype(np.float32)
    taken = set(_coords_to_flat(train_coords, raster_width).tolist())

    remaining_coords, remaining_y = _filter_out_flat(val_test_pool_coords, val_test_pool_y, taken, raster_width)
    val_pos = _sample_balanced_positions(
        pool_y=remaining_y,
        class_ids=class_ids,
        per_class_count=val_per_class,
        rng=rng,
        pool_name="val top-75% S2 quality-pool excluding train",
    )
    val_coords = remaining_coords[val_pos]
    val_y = remaining_y[val_pos]
    taken.update(_coords_to_flat(val_coords, raster_width).tolist())

    remaining_coords, remaining_y = _filter_out_flat(val_test_pool_coords, val_test_pool_y, taken, raster_width)
    test_pos = _sample_balanced_positions(
        pool_y=remaining_y,
        class_ids=class_ids,
        per_class_count=test_per_class,
        rng=rng,
        pool_name="test top-75% S2 quality-pool excluding train/val",
    )
    test_coords = remaining_coords[test_pos]
    test_y = remaining_y[test_pos]

    selected_coords = np.concatenate([train_coords, val_coords, test_coords], axis=0)
    selected_y = np.concatenate([train_y, val_y, test_y], axis=0)
    train_n = int(train_y.shape[0])
    val_n = int(val_y.shape[0])
    test_n = int(test_y.shape[0])

    timeline = [date_token for date_token, _ in build_timeline(s1_scenes, s2_scenes)]
    logger.log(
        f"building shared feature arrays for balanced selected samples count={selected_coords.shape[0]} "
        f"train={train_n} val={val_n} test={test_n}"
    )
    all_x, all_obs_mask, all_doy, _ = _build_feature_arrays(
        coords_rc=selected_coords,
        s1_scenes=s1_scenes,
        s2_scenes=s2_scenes,
        s2_scale=10000.0,
        progress_hook=logger.log,
    )

    split_train = _slice_pixel_batch(
        x=all_x,
        obs_mask=all_obs_mask,
        doy=all_doy,
        y=selected_y,
        coords=selected_coords,
        class_names=class_names,
        timeline=timeline,
        start=0,
        end=train_n,
    )
    split_val = _slice_pixel_batch(
        x=all_x,
        obs_mask=all_obs_mask,
        doy=all_doy,
        y=selected_y,
        coords=selected_coords,
        class_names=class_names,
        timeline=timeline,
        start=train_n,
        end=train_n + val_n,
    )
    split_test = _slice_pixel_batch(
        x=all_x,
        obs_mask=all_obs_mask,
        doy=all_doy,
        y=selected_y,
        coords=selected_coords,
        class_names=class_names,
        timeline=timeline,
        start=train_n + val_n,
        end=train_n + val_n + test_n,
    )

    if h5_path.exists():
        h5_path.unlink()
    save_split_batch_to_h5(h5_path, "train", split_train.x, split_train.obs_mask, split_train.doy, split_train.y, split_train.coords)
    save_split_batch_to_h5(h5_path, "val", split_val.x, split_val.obs_mask, split_val.doy, split_val.y, split_val.coords)
    save_split_batch_to_h5(h5_path, "test", split_test.x, split_test.obs_mask, split_test.doy, split_test.y, split_test.coords)

    stats = QualitySelectionStats(
        candidate_count=total_valid,
        selected_count=int(train_n),
        train_count=train_n,
        val_count=val_n,
        test_count=test_n,
        quality_mean_candidates=float(valid_s2_counts.mean()) if valid_s2_counts.size else 0.0,
        quality_mean_selected=float(train_quality.mean()) if train_quality.size else 0.0,
        quality_min_selected=float(train_quality.min()) if train_quality.size else 0.0,
        quality_max_selected=float(train_quality.max()) if train_quality.size else 0.0,
    )
    save_metadata_to_h5(
        h5_path,
        {
            "site": site,
            "year": year,
            "sample_mode": "balanced_quality",
            "allowed_labels_json": json.dumps(list(allowed_labels)),
            "min_confidence": float(min_confidence),
            "top_fraction": float(top_fraction),
            "time_steps": int(all_x.shape[1]),
            "channels": int(all_x.shape[2]),
            "candidate_count": stats.candidate_count,
            "selected_count": stats.selected_count,
            "quality_mean_candidates": stats.quality_mean_candidates,
            "quality_mean_selected": stats.quality_mean_selected,
            "quality_min_selected": stats.quality_min_selected,
            "quality_max_selected": stats.quality_max_selected,
            "num_classes": int(len(class_names)),
            "class_names_json": json.dumps(class_names),
            "train_per_class": int(train_per_class),
            "val_per_class": int(val_per_class),
            "test_per_class": int(test_per_class),
        },
    )

    manifest = {
        "year": year,
        "site": site,
        "h5_path": str(h5_path),
        "sample_mode": "balanced_quality",
        "allowed_labels": list(allowed_labels),
        "min_confidence": min_confidence,
        "top_fraction": top_fraction,
        "class_names": class_names,
        "train_samples": train_n,
        "val_samples": val_n,
        "test_samples": test_n,
        "train_distribution": class_distribution(split_train.y),
        "val_distribution": class_distribution(split_val.y),
        "test_distribution": class_distribution(split_test.y),
        "quality_selection": {
            "applies_to": "train_top25_val_test_top75",
            "train_candidate_count": total_valid,
            "train_eligible_count": int(train_pool_coords.shape[0]),
            "val_test_eligible_count": int(val_test_pool_coords.shape[0]),
            "train_s2_valid_count_threshold": float(train_threshold),
            "val_test_s2_valid_count_threshold": float(val_test_threshold),
            "train_selected_count": train_n,
            "train_quality_mean_candidates": stats.quality_mean_candidates,
            "train_quality_mean_selected": stats.quality_mean_selected,
            "train_quality_min_selected": stats.quality_min_selected,
            "train_quality_max_selected": stats.quality_max_selected,
            "train_selection_rule": "class-balanced within pixels in the top 25% by S2 valid-count",
            "val_test_selection_rule": "class-balanced within pixels in the top 75% by S2 valid-count, excluding train/val",
        },
        "quality_stats_legacy": {
            "candidate_count": stats.candidate_count,
            "selected_count": stats.selected_count,
            "train_count": stats.train_count,
            "val_count": stats.val_count,
            "test_count": stats.test_count,
            "quality_mean_candidates": stats.quality_mean_candidates,
            "quality_mean_selected": stats.quality_mean_selected,
            "quality_min_selected": stats.quality_min_selected,
            "quality_max_selected": stats.quality_max_selected,
        },
    }
    with open(cache_dir / f"{site.split('_', 1)[-1].lower()}_{year}_quality_randomsplit_timeseries_manifest.json", "w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    args = parse_args()
    if min(args.per_class_train, args.per_class_val, args.per_class_test) < 0:
        raise ValueError("per-class sample counts must be non-negative")

    paths = project_paths()
    data_root = paths.raw_data_dir
    site_root = args.out_root or (paths.processed_data_dir / "from_tif" / args.site)
    site_root.mkdir(parents=True, exist_ok=True)
    root_logger = Logger(site_root / "build_feature_cache_from_tif_balance.log")
    root_logger.log(f"building balanced cache config={vars(args)}")

    summary: list[dict[str, object]] = []
    for year in args.years:
        year_dir = site_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        h5_path = year_dir / f"{args.site.split('_', 1)[-1].lower()}_{year}_quality_randomsplit_timeseries.h5"
        logger = Logger(year_dir / f"build_{year}_balanced_quality_feature_cache.log")
        manifest = _build_and_save_year(
            data_root=data_root,
            site=args.site,
            year=year,
            cache_dir=year_dir,
            h5_path=h5_path,
            allowed_labels=tuple(args.allowed_labels),
            min_confidence=args.min_confidence,
            top_fraction=args.top_fraction,
            train_per_class=args.per_class_train,
            val_per_class=args.per_class_val,
            test_per_class=args.per_class_test,
            seed=args.seed,
            logger=logger,
        )
        summary.append(manifest)
        root_logger.log(
            f"year={year} done train={manifest['train_samples']} val={manifest['val_samples']} test={manifest['test_samples']} "
            f"train_dist={manifest['train_distribution']} val_dist={manifest['val_distribution']} test_dist={manifest['test_distribution']}"
        )

    config = vars(args).copy()
    if config.get("out_root") is not None:
        config["out_root"] = str(config["out_root"])
    with open(site_root / "multiyear_quality_randomsplit_manifest.json", "w", encoding="ascii") as f:
        json.dump({"config": config, "years": summary}, f, indent=2)
    root_logger.log("balanced site feature cache build complete")


if __name__ == "__main__":
    main()
