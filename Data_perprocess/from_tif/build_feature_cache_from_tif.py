from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "config").is_dir() and (parent / "utils").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CacheBuildConfig, project_paths
from utils.h5_timeseries import save_metadata_to_h5, save_split_batch_to_h5
from utils.timeseries_dataset import (
    QualitySelectionStats,
    SplitPixelBatch,
    build_random_fraction_year_split,
    build_quality_filtered_random_fraction_year_split,
)
from utils.training_common import Logger


def class_distribution(y) -> list[int]:
    values = y.numpy()
    max_class = int(values.max()) if values.size else -1
    return [int((values == idx).sum()) for idx in range(max_class + 1)]


def write_year_cache(
    output_dir,
    h5_path,
    split: SplitPixelBatch,
    cfg: CacheBuildConfig,
    year: int,
    stats: QualitySelectionStats,
) -> None:
    quality_selection = {
        "applies_to": "train_top25_val_test_top75" if cfg.sample_mode == "quality" else "none",
        "train_candidate_count": int(stats.candidate_count),
        "train_selected_count": int(stats.selected_count),
        "train_quality_mean_candidates": float(stats.quality_mean_candidates),
        "train_quality_mean_selected": float(stats.quality_mean_selected),
        "train_quality_min_selected": float(stats.quality_min_selected),
        "train_quality_max_selected": float(stats.quality_max_selected),
        "train_selection_rule": (
            "prioritize pixels in the top quartile by S2 valid-count"
            if cfg.sample_mode == "quality"
            else "uniform random over all valid pixels"
        ),
        "val_test_selection_rule": (
            "uniform random within pixels in the top 75% by S2 valid-count, excluding train"
            if cfg.sample_mode == "quality"
            else "uniform random over all valid pixels"
        ),
    }
    save_split_batch_to_h5(h5_path, "train", split.train.x, split.train.obs_mask, split.train.doy, split.train.y, split.train.coords)
    save_split_batch_to_h5(h5_path, "val", split.val.x, split.val.obs_mask, split.val.doy, split.val.y, split.val.coords)
    save_split_batch_to_h5(h5_path, "test", split.test.x, split.test.obs_mask, split.test.doy, split.test.y, split.test.coords)
    save_metadata_to_h5(
        h5_path,
        {
            "site": cfg.site,
            "year": year,
            "train_fraction": cfg.train_fraction,
            "val_fraction": cfg.val_fraction,
            "test_fraction": cfg.test_fraction,
            "seed": cfg.seed,
            "sample_mode": cfg.sample_mode,
            "time_steps": int(split.train.x.shape[1]),
            "channels": int(split.train.x.shape[2]),
            "quality_applies_to": quality_selection["applies_to"],
            "train_quality_candidate_count": quality_selection["train_candidate_count"],
            "train_quality_selected_count": quality_selection["train_selected_count"],
            "train_quality_mean_candidates": quality_selection["train_quality_mean_candidates"],
            "train_quality_mean_selected": quality_selection["train_quality_mean_selected"],
            "train_quality_min_selected": quality_selection["train_quality_min_selected"],
            "train_quality_max_selected": quality_selection["train_quality_max_selected"],
            "num_classes": int(split.train.class_ids.numel()),
        },
    )

    manifest = {
        "config": asdict(cfg),
        "year": year,
        "h5_path": str(h5_path),
        "train_samples": int(split.train.y.numel()),
        "val_samples": int(split.val.y.numel()),
        "test_samples": int(split.test.y.numel()),
        "train_distribution": class_distribution(split.train.y),
        "val_distribution": class_distribution(split.val.y),
        "test_distribution": class_distribution(split.test.y),
        "num_classes": int(split.train.class_ids.numel()),
        "class_ids": split.train.class_ids.tolist(),
        "time_steps": int(split.train.x.shape[1]),
        "channels": int(split.train.x.shape[2]),
        "timeline": split.train.timeline,
        "quality_selection": quality_selection,
        "quality_stats_legacy": asdict(stats),
    }
    with open(output_dir / f"{cfg.site_token}_{year}_quality_randomsplit_timeseries_manifest.json", "w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2)


def parse_args() -> CacheBuildConfig:
    defaults = CacheBuildConfig()
    parser = argparse.ArgumentParser(description="Build site/year feature caches.")
    parser.add_argument("--site", default=defaults.site)
    parser.add_argument("--years", nargs="+", type=int, default=list(defaults.years))
    parser.add_argument("--train-fraction", type=float, default=defaults.train_fraction)
    parser.add_argument("--val-fraction", type=float, default=defaults.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=defaults.test_fraction)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--sample-mode", choices=["quality", "random"], default=defaults.sample_mode)
    parser.add_argument("--min-confidence", type=float, default=defaults.min_confidence)
    parser.add_argument("--allowed-labels", nargs="*", type=int, default=[])
    parser.add_argument("--fractions-from-eligible-pool", action="store_true")
    args = parser.parse_args()
    return CacheBuildConfig(
        site=args.site,
        years=tuple(args.years),
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        sample_mode=args.sample_mode,
        min_confidence=args.min_confidence,
        allowed_labels=tuple(args.allowed_labels),
        fractions_from_eligible_pool=args.fractions_from_eligible_pool,
    )


def main() -> None:
    cfg = parse_args()
    paths = project_paths()
    output_root = paths.processed_data_dir / "from_tif" / cfg.site
    output_root.mkdir(parents=True, exist_ok=True)

    root_logger = Logger(output_root / "build_feature_cache_from_tif.log")
    root_logger.log(f"building site feature cache config={asdict(cfg)}")

    summary: list[dict[str, object]] = []
    for year in cfg.years:
        year_dir = output_root / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        h5_path = year_dir / f"{cfg.site_token}_{year}_quality_randomsplit_timeseries.h5"
        logger = Logger(year_dir / f"build_{year}_quality_feature_cache.log")
        logger.log(f"building year={year} target={h5_path}")
        if h5_path.exists():
            h5_path.unlink()
            logger.log(f"removed existing cache {h5_path}")

        if cfg.sample_mode == "quality":
            split, stats = build_quality_filtered_random_fraction_year_split(
                data_root=paths.raw_data_dir,
                site=cfg.site,
                year=year,
                train_fraction=cfg.train_fraction,
                val_fraction=cfg.val_fraction,
                test_fraction=cfg.test_fraction,
                seed=cfg.seed,
                cache_dir=year_dir,
                min_confidence=cfg.min_confidence,
                allowed_labels=cfg.allowed_labels,
                fractions_from_eligible_pool=cfg.fractions_from_eligible_pool,
                progress_hook=logger.log,
            )
            logger.log(
                "quality-filtered split built "
                f"train={split.train.y.numel()} val={split.val.y.numel()} test={split.test.y.numel()} "
                f"time_steps={split.train.x.shape[1]} channels={split.train.x.shape[2]} "
                f"train_quality_mean_candidates={stats.quality_mean_candidates:.4f} "
                f"train_quality_mean_selected={stats.quality_mean_selected:.4f} "
                "val_test_sampling=random_top75_s2_valid_count_excluding_train"
            )
        else:
            split = build_random_fraction_year_split(
                data_root=paths.raw_data_dir,
                site=cfg.site,
                year=year,
                train_fraction=cfg.train_fraction,
                val_fraction=cfg.val_fraction,
                test_fraction=cfg.test_fraction,
                seed=cfg.seed,
                cache_dir=year_dir,
                progress_hook=logger.log,
            )
            selected_count = int(split.train.y.numel() + split.val.y.numel() + split.test.y.numel())
            stats = QualitySelectionStats(
                candidate_count=selected_count,
                selected_count=selected_count,
                train_count=int(split.train.y.numel()),
                val_count=int(split.val.y.numel()),
                test_count=int(split.test.y.numel()),
                quality_mean_candidates=0.0,
                quality_mean_selected=0.0,
                quality_min_selected=0.0,
                quality_max_selected=0.0,
            )
            logger.log(
                "random split built "
                f"train={split.train.y.numel()} val={split.val.y.numel()} test={split.test.y.numel()} "
                f"time_steps={split.train.x.shape[1]} channels={split.train.x.shape[2]}"
            )
        write_year_cache(year_dir, h5_path, split, cfg, year, stats)
        logger.log("year cache build complete")
        root_logger.log(
            f"year={year} done h5={h5_path.name} train={split.train.y.numel()} val={split.val.y.numel()} "
            f"test={split.test.y.numel()} sample_mode={cfg.sample_mode} "
            f"quality_applies_to={'train_top25_val_test_top75' if cfg.sample_mode == 'quality' else 'none'} "
            f"train_quality_mean_selected={stats.quality_mean_selected:.4f}"
        )
        summary.append(
            {
                "year": year,
                "h5_path": str(h5_path),
                "train_samples": int(split.train.y.numel()),
                "val_samples": int(split.val.y.numel()),
                "test_samples": int(split.test.y.numel()),
                "time_steps": int(split.train.x.shape[1]),
                "channels": int(split.train.x.shape[2]),
                "quality_selection": {
                    "applies_to": "train_top25_val_test_top75" if cfg.sample_mode == "quality" else "none",
                    "train_candidate_count": int(stats.candidate_count),
                    "train_selected_count": int(stats.selected_count),
                    "train_quality_mean_candidates": float(stats.quality_mean_candidates),
                    "train_quality_mean_selected": float(stats.quality_mean_selected),
                },
                "quality_stats_legacy": asdict(stats),
            }
        )

    with open(output_root / "multiyear_quality_randomsplit_manifest.json", "w", encoding="ascii") as f:
        json.dump({"config": asdict(cfg), "years": summary}, f, indent=2)
    root_logger.log("site feature cache build complete")


if __name__ == "__main__":
    main()
