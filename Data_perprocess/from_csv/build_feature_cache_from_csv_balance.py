from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch

REPO_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "config").is_dir() and (parent / "utils").is_dir()
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import CacheBuildConfig, project_paths
from utils.h5_timeseries import save_metadata_to_h5, save_split_batch_to_h5
from utils.training_common import Logger


CSV_META_COLUMNS = {
    "system:index",
    "doy",
    "label",
    "lat",
    "lon",
    "name",
    "sample_id",
    "sensor",
    "year",
    "ymd",
    ".geo",
}

DEFAULT_S2_FEATURE_COLUMNS = [
    "B2",
    "B3",
    "B4",
    "B5",
    "B6",
    "B7",
    "B8",
    "B8A",
    "B11",
    "B12",
]

DEFAULT_FJ_LABEL_MAP = {
    1: 0,
    2: 1,
    3: 2,
}


def class_distribution(y: torch.Tensor, num_classes: int) -> list[int]:
    values = y.numpy()
    return [int((values == idx).sum()) for idx in range(num_classes)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build balanced year feature caches from sample-feature CSV files.")
    parser.add_argument("--site", default="site_FJ")
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    parser.add_argument("--sensor", default="S2", help="Sensor subset to use, for example S2 or S1.")
    parser.add_argument("--per-class-train", type=int, required=True)
    parser.add_argument("--per-class-val", type=int, required=True)
    parser.add_argument("--per-class-test", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--feature-columns",
        nargs="*",
        default=None,
        help="Optional explicit feature column order. Defaults to the first non-empty CSV header order.",
    )
    return parser.parse_args()


def resolve_year_dir(data_root: Path, site: str, year: int) -> Path:
    candidates = [
        data_root / site / f"{site}_{year}",
        data_root / site / str(year),
        data_root / f"{site}_{year}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve year directory for site={site} year={year} under {data_root}")


def list_csv_files(year_dir: Path, sensor: str) -> list[Path]:
    pattern = f"*{sensor}*.csv"
    return sorted(path for path in year_dir.glob(pattern) if path.is_file() and path.stat().st_size > 2)


def infer_feature_columns(csv_files: Iterable[Path], explicit_columns: list[str] | None) -> list[str]:
    if explicit_columns:
        return explicit_columns

    for path in csv_files:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                if all(name in reader.fieldnames for name in DEFAULT_S2_FEATURE_COLUMNS):
                    return list(DEFAULT_S2_FEATURE_COLUMNS)
                return [name for name in reader.fieldnames if name not in CSV_META_COLUMNS]
    raise ValueError("Unable to infer feature columns because no non-empty CSV file was found.")


def safe_float(value: str | None) -> float:
    if value is None:
        return float("nan")
    text = value.strip()
    if not text:
        return float("nan")
    return float(text)


def safe_int(value: str | None, field_name: str) -> int:
    if value is None:
        raise ValueError(f"Missing required integer field {field_name}")
    text = value.strip()
    if not text:
        raise ValueError(f"Empty required integer field {field_name}")
    return int(float(text))


def compute_split_counts(total: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    if total <= 0:
        return 0, 0, 0

    raw = np.array(fractions, dtype=np.float64) * float(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[idx] += 1

    nonzero_targets = [idx for idx, frac in enumerate(fractions) if frac > 0]
    if total >= len(nonzero_targets):
        for idx in nonzero_targets:
            if counts[idx] == 0:
                donor = int(np.argmax(counts))
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1

    return int(counts[0]), int(counts[1]), int(counts[2])


def stratified_split_indices(
    labels: np.ndarray,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []

    for class_id in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == class_id)
        rng.shuffle(class_idx)
        train_n, val_n, test_n = compute_split_counts(
            total=class_idx.size,
            fractions=(train_fraction, val_fraction, test_fraction),
        )
        offset_train = train_n
        offset_val = train_n + val_n
        train_parts.append(class_idx[:offset_train])
        val_parts.append(class_idx[offset_train:offset_val])
        test_parts.append(class_idx[offset_val : offset_val + test_n])

    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def balanced_split_indices(
    labels: np.ndarray,
    per_class_train: int,
    per_class_val: int,
    per_class_test: int,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    per_class_total = per_class_train + per_class_val + per_class_test

    if per_class_total <= 0:
        raise ValueError("At least one per-class split count must be positive.")
    if min(per_class_train, per_class_val, per_class_test) < 0:
        raise ValueError("Per-class split counts must be non-negative.")

    for class_id in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == class_id)
        if class_idx.size < per_class_total:
            raise ValueError(
                f"Not enough samples for class={class_id}: required={per_class_total}, found={class_idx.size}"
            )
        picked = rng.choice(class_idx, size=per_class_total, replace=False)
        train_end = per_class_train
        val_end = per_class_train + per_class_val
        train_parts.append(picked[:train_end])
        val_parts.append(picked[train_end:val_end])
        test_parts.append(picked[val_end:])

    train_idx = np.concatenate(train_parts) if train_parts else np.empty(0, dtype=np.int64)
    val_idx = np.concatenate(val_parts) if val_parts else np.empty(0, dtype=np.int64)
    test_idx = np.concatenate(test_parts) if test_parts else np.empty(0, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_year_arrays(
    csv_files: list[Path],
    feature_columns: list[str],
    logger: Logger,
) -> tuple[dict[str, np.ndarray], dict[int, str], dict[int, str]]:
    sample_rows: dict[int, dict[str, object]] = {}
    timeline_doys: dict[int, int] = {}
    timeline_tokens: dict[int, str] = {}
    label_names: dict[int, str] = {}

    for csv_path in csv_files:
        logger.log(f"reading {csv_path.name}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                logger.log(f"skipping empty file {csv_path.name}")
                continue
            missing = [col for col in feature_columns if col not in reader.fieldnames]
            if missing:
                raise ValueError(f"{csv_path} missing required feature columns {missing}")
            row_count = 0
            for row in reader:
                row_count += 1
                sample_id = safe_int(row.get("sample_id"), "sample_id")
                doy = safe_int(row.get("doy"), "doy")
                ymd = (row.get("ymd") or "").strip() or f"{doy:03d}"
                label_raw = safe_int(row.get("label"), "label")
                if label_raw not in DEFAULT_FJ_LABEL_MAP:
                    continue

                timeline_doys[doy] = doy
                timeline_tokens[doy] = ymd
                label_names.setdefault(label_raw, (row.get("name") or "").strip())

                item = sample_rows.setdefault(
                    sample_id,
                    {
                        "label_raw": label_raw,
                        "lat": safe_float(row.get("lat")),
                        "lon": safe_float(row.get("lon")),
                        "values_by_doy": {},
                    },
                )
                if int(item["label_raw"]) != label_raw:
                    raise ValueError(f"sample_id={sample_id} has inconsistent labels across CSV files")

                feature_values = np.array([safe_float(row.get(col)) for col in feature_columns], dtype=np.float32)
                item["values_by_doy"][doy] = feature_values
            logger.log(f"loaded rows={row_count} from {csv_path.name}")

    if not sample_rows:
        raise ValueError("No usable rows were loaded from CSV files.")

    sorted_doys = sorted(timeline_doys)
    doy_to_index = {doy: idx for idx, doy in enumerate(sorted_doys)}
    sorted_sample_ids = sorted(sample_rows)
    raw_labels = sorted({int(item["label_raw"]) for item in sample_rows.values()})
    if not raw_labels:
        raise ValueError("No supported FJ labels found after filtering.")
    if not all(label in DEFAULT_FJ_LABEL_MAP for label in raw_labels):
        raise ValueError(f"Unsupported FJ raw labels found: {raw_labels}")
    label_map = {label_raw: DEFAULT_FJ_LABEL_MAP[label_raw] for label_raw in raw_labels}

    num_samples = len(sorted_sample_ids)
    time_steps = len(sorted_doys)
    channels = len(feature_columns)
    x = np.zeros((num_samples, time_steps, channels), dtype=np.float32)
    obs_mask = np.zeros((num_samples, time_steps, channels), dtype=np.bool_)
    doy = np.tile(np.asarray(sorted_doys, dtype=np.int64), (num_samples, 1))
    y = np.zeros((num_samples,), dtype=np.int64)
    coords = np.zeros((num_samples, 2), dtype=np.int64)

    for sample_index, sample_id in enumerate(sorted_sample_ids):
        item = sample_rows[sample_id]
        y[sample_index] = label_map[int(item["label_raw"])]
        coords[sample_index, 0] = int(sample_id)
        coords[sample_index, 1] = 0
        values_by_doy = item["values_by_doy"]
        for day, values in values_by_doy.items():
            t = doy_to_index[day]
            valid = np.isfinite(values) & (values != 0)
            x[sample_index, t, valid] = values[valid]
            obs_mask[sample_index, t, valid] = True

    class_name_map = {}
    for label_raw, mapped_label in DEFAULT_FJ_LABEL_MAP.items():
        if label_raw in label_names:
            class_name_map[mapped_label] = label_names.get(label_raw) or f"class_{mapped_label}"
    timeline = [timeline_tokens[d] for d in sorted_doys]
    arrays = {"x": x, "obs_mask": obs_mask, "doy": doy, "y": y, "coords": coords}
    return arrays, class_name_map, {idx: token for idx, token in enumerate(timeline)}


def subset_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, torch.Tensor]:
    return {
        "x": torch.from_numpy(arrays["x"][indices]),
        "obs_mask": torch.from_numpy(arrays["obs_mask"][indices]),
        "doy": torch.from_numpy(arrays["doy"][indices]),
        "y": torch.from_numpy(arrays["y"][indices]),
        "coords": torch.from_numpy(arrays["coords"][indices]),
    }


def main() -> None:
    args = parse_args()

    cfg = CacheBuildConfig(
        site=args.site,
        years=tuple(args.years),
        train_fraction=0.0,
        val_fraction=0.0,
        test_fraction=0.0,
        seed=args.seed,
        sample_mode=f"csv_{args.sensor.lower()}_balanced",
    )
    paths = project_paths()
    output_root = paths.processed_data_dir / "from_csv" / cfg.site
    output_root.mkdir(parents=True, exist_ok=True)
    root_logger = Logger(output_root / f"build_{cfg.site_token}_{args.sensor.lower()}_feature_cache_from_csv_balance.log")
    root_logger.log(f"building balanced csv feature cache config={vars(args)}")

    summary: list[dict[str, object]] = []
    multiyear_label_names: dict[str, str] = {}
    feature_columns_ref: list[str] | None = None

    for year in cfg.years:
        year_dir = resolve_year_dir(paths.raw_data_dir, cfg.site, year)
        csv_files = list_csv_files(year_dir, args.sensor)
        if not csv_files:
            root_logger.log(f"year={year} skipped because no non-empty {args.sensor} CSV files were found")
            continue

        feature_columns = infer_feature_columns(csv_files, args.feature_columns)
        if feature_columns_ref is None:
            feature_columns_ref = feature_columns
        elif feature_columns_ref != feature_columns:
            raise ValueError(
                f"Feature columns changed for year={year}. expected={feature_columns_ref} current={feature_columns}"
            )

        out_year_dir = output_root / str(year)
        out_year_dir.mkdir(parents=True, exist_ok=True)
        logger = Logger(out_year_dir / f"build_{year}_{args.sensor.lower()}_csv_feature_cache.log")
        logger.log(f"building year={year} from files={len(csv_files)} sensor={args.sensor}")

        arrays, class_name_map, timeline_map = build_year_arrays(csv_files, feature_columns, logger)
        split_idx = balanced_split_indices(
            arrays["y"],
            per_class_train=args.per_class_train,
            per_class_val=args.per_class_val,
            per_class_test=args.per_class_test,
            seed=args.seed + year,
        )

        h5_path = out_year_dir / f"{cfg.site_token}_{year}_quality_randomsplit_timeseries.h5"
        if h5_path.exists():
            h5_path.unlink()
            logger.log(f"removed existing cache {h5_path}")

        train = subset_arrays(arrays, split_idx["train"])
        val = subset_arrays(arrays, split_idx["val"])
        test = subset_arrays(arrays, split_idx["test"])

        save_split_batch_to_h5(h5_path, "train", train["x"], train["obs_mask"], train["doy"], train["y"], train["coords"])
        save_split_batch_to_h5(h5_path, "val", val["x"], val["obs_mask"], val["doy"], val["y"], val["coords"])
        save_split_batch_to_h5(h5_path, "test", test["x"], test["obs_mask"], test["doy"], test["y"], test["coords"])
        save_metadata_to_h5(
            h5_path,
            {
                "site": cfg.site,
                "year": year,
                "per_class_train": int(args.per_class_train),
                "per_class_val": int(args.per_class_val),
                "per_class_test": int(args.per_class_test),
                "seed": cfg.seed,
                "sample_mode": cfg.sample_mode,
                "sensor": args.sensor,
                "time_steps": int(arrays["x"].shape[1]),
                "channels": int(arrays["x"].shape[2]),
                "num_classes": int(arrays["y"].max()) + 1,
                "total_samples": int(arrays["y"].shape[0]),
            },
        )

        for class_idx, class_name in class_name_map.items():
            multiyear_label_names[str(class_idx)] = class_name

        timeline = [timeline_map[idx] for idx in range(len(timeline_map))]
        manifest = {
            "config": asdict(cfg),
            "year": year,
            "h5_path": str(h5_path),
            "sensor": args.sensor,
            "feature_columns": feature_columns,
            "per_class_train": int(args.per_class_train),
            "per_class_val": int(args.per_class_val),
            "per_class_test": int(args.per_class_test),
            "time_steps": int(arrays["x"].shape[1]),
            "channels": int(arrays["x"].shape[2]),
            "num_classes": int(arrays["y"].max()) + 1,
            "class_names": {str(key): value for key, value in class_name_map.items()},
            "timeline": timeline,
            "total_samples": int(arrays["y"].shape[0]),
            "train_samples": int(train["y"].numel()),
            "val_samples": int(val["y"].numel()),
            "test_samples": int(test["y"].numel()),
            "train_distribution": class_distribution(train["y"], int(arrays["y"].max()) + 1),
            "val_distribution": class_distribution(val["y"], int(arrays["y"].max()) + 1),
            "test_distribution": class_distribution(test["y"], int(arrays["y"].max()) + 1),
        }
        with (out_year_dir / f"{cfg.site_token}_{year}_quality_randomsplit_timeseries_manifest.json").open(
            "w",
            encoding="ascii",
        ) as f:
            json.dump(manifest, f, indent=2, ensure_ascii=True)

        logger.log(
            f"built h5={h5_path.name} samples={arrays['y'].shape[0]} train={train['y'].numel()} "
            f"val={val['y'].numel()} test={test['y'].numel()} time_steps={arrays['x'].shape[1]} "
            f"channels={arrays['x'].shape[2]}"
        )
        summary.append(
            {
                "year": year,
                "h5_path": str(h5_path),
                "sensor": args.sensor,
                "feature_columns": feature_columns,
                "time_steps": int(arrays["x"].shape[1]),
                "channels": int(arrays["x"].shape[2]),
                "num_classes": int(arrays["y"].max()) + 1,
                "total_samples": int(arrays["y"].shape[0]),
                "train_samples": int(train["y"].numel()),
                "val_samples": int(val["y"].numel()),
                "test_samples": int(test["y"].numel()),
            }
        )

    if not summary:
        raise ValueError(f"No yearly cache was produced for site={cfg.site} sensor={args.sensor}")

    with (output_root / "multiyear_quality_randomsplit_manifest.json").open("w", encoding="ascii") as f:
        json.dump(
            {
                "config": asdict(cfg),
                "sensor": args.sensor,
                "feature_columns": feature_columns_ref,
                "class_names": multiyear_label_names,
                "years": summary,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )
    root_logger.log("balanced csv feature cache build complete")


if __name__ == "__main__":
    main()
