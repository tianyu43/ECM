from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np


S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]


@dataclass
class YearArrays:
    x: np.ndarray
    obs_mask: np.ndarray
    doy: np.ndarray
    y: np.ndarray
    coords: np.ndarray
    channel_names: list[str]
    sample_splits: list[str | None]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert site_FJ CSV time-series data into ECM quality_randomsplit_timeseries H5 files."
    )
    parser.add_argument("--input-root", type=Path, default=Path("Data/Data_raw/site_FJ"))
    parser.add_argument("--out-root", type=Path, default=Path("Data_perprocess/from_csv/site_FJ"))
    parser.add_argument("--site-token", default="fj")
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--sample-id-col", default=None)
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--row-col", default=None)
    parser.add_argument("--col-col", default=None)
    parser.add_argument("--remap-labels", action="store_true")
    parser.add_argument(
        "--s2-12ch",
        action="store_true",
        help="When S2 band columns are present, write ECM 12-channel arrays with empty S1 channels and S2 in channels 2:12.",
    )
    parser.add_argument(
        "--sensor-filter",
        choices=("all", "S1", "S2"),
        default="all",
        help="Keep only CSV files/rows for this sensor when a sensor token or column is present.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def find_column(columns: list[str], candidates: list[str], explicit: str | None = None) -> str | None:
    if explicit:
        if explicit not in columns:
            raise ValueError(f"Requested column {explicit!r} is not present")
        return explicit
    normalized = {norm(col): col for col in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def year_from_path(path: Path) -> int | None:
    for part in [path.stem, *path.parts[::-1]]:
        match = re.search(r"(20\d{2})", part)
        if match:
            return int(match.group(1))
    return None


def csv_paths_by_year(input_root: Path, years: set[int] | None, sensor_filter: str = "all") -> dict[int, list[Path]]:
    paths = [input_root] if input_root.is_file() else sorted(input_root.rglob("*.csv"))
    out: dict[int, list[Path]] = {}
    for path in paths:
        if sensor_filter != "all" and f"_{sensor_filter}_" not in path.name:
            continue
        year = year_from_path(path)
        if year is None:
            continue
        if years is not None and year not in years:
            continue
        out.setdefault(year, []).append(path)
    return out


def read_csvs(paths: list[Path], sensor_filter: str = "all") -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    columns: list[str] = []
    seen_columns: set[str] = set()
    for path in paths:
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for col in reader.fieldnames:
                if col not in seen_columns:
                    columns.append(col)
                    seen_columns.add(col)
            for row in reader:
                if sensor_filter != "all" and row.get("sensor", sensor_filter) != sensor_filter:
                    continue
                row["_source_csv"] = str(path)
                rows.append(row)
    if not columns or not rows:
        raise FileNotFoundError("No CSV records found")
    return columns, rows


def parse_float(value: str | None) -> float:
    if value is None or value == "":
        return math.nan
    try:
        return float(value)
    except ValueError:
        return math.nan


def is_numeric_column(rows: list[dict[str, str]], col: str) -> bool:
    seen = False
    for row in rows[: min(len(rows), 1000)]:
        value = row.get(col, "")
        if value == "":
            continue
        try:
            float(value)
        except ValueError:
            return False
        seen = True
    return seen


def coerce_label(value: str) -> int:
    try:
        return int(float(value))
    except ValueError as exc:
        raise ValueError(
            f"Label {value!r} is not numeric. Provide numeric labels in CSV or pre-map crop names to ids."
        ) from exc


def coerce_doy(value: str) -> int:
    try:
        doy = int(float(value))
        if 1 <= doy <= 366:
            return doy
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return int(datetime.strptime(value, fmt).strftime("%j"))
        except ValueError:
            continue
    parsed = datetime.fromisoformat(value)
    return int(parsed.strftime("%j"))


def is_long_table(columns: list[str]) -> bool:
    normalized = {norm(col) for col in columns}
    return bool({"doy", "dayofyear", "date"} & normalized)


def feature_columns(rows: list[dict[str, str]], columns: list[str], exclude: set[str]) -> list[str]:
    if all(band in columns for band in S2_BANDS):
        return S2_BANDS
    features = [col for col in columns if col not in exclude and is_numeric_column(rows, col)]
    if not features:
        raise ValueError("Could not infer numeric feature columns")
    return features


def build_long_arrays(columns: list[str], rows: list[dict[str, str]], args: argparse.Namespace) -> YearArrays:
    sample_col = find_column(columns, ["sample_id", "id", "field_id", "pixel_id"], args.sample_id_col)
    label_col = find_column(columns, ["y", "label", "class", "crop_id", "crop"], args.label_col)
    row_col = find_column(columns, ["row", "r"], args.row_col)
    col_col = find_column(columns, ["col", "c"], args.col_col)
    split_col = find_column(columns, ["split"])
    time_col = find_column(columns, ["doy", "dayofyear"]) or find_column(columns, ["date"])
    if label_col is None:
        raise ValueError("Could not infer label column. Pass --label-col.")
    if time_col is None:
        raise ValueError("Could not infer doy/date column.")
    if sample_col is None and (row_col is None or col_col is None):
        raise ValueError("Need --sample-id-col or row/col columns for long CSV.")

    exclude = {
        c
        for c in [
            sample_col,
            label_col,
            row_col,
            col_col,
            split_col,
            time_col,
            "year",
            "ymd",
            "lat",
            "lon",
            "latitude",
            "longitude",
            "system:index",
        ]
        if c
    }
    channels = feature_columns(rows, columns, exclude)
    out_channels = ["VV", "VH", *S2_BANDS] if args.s2_12ch and all(band in columns for band in S2_BANDS) else channels
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = row[sample_col] if sample_col else f"{row[row_col]}:{row[col_col]}"
        grouped.setdefault(key, []).append(row)
    sample_keys = list(grouped.keys())
    timeline = sorted({coerce_doy(row[time_col]) for row in rows if row.get(time_col, "") != ""})
    doy_index = {doy: idx for idx, doy in enumerate(timeline)}

    x = np.zeros((len(sample_keys), len(timeline), len(out_channels)), dtype=np.float32)
    obs_mask = np.zeros_like(x, dtype=bool)
    doy = np.asarray(timeline, dtype=np.int16)[None, :].repeat(len(sample_keys), axis=0)
    y = np.zeros((len(sample_keys),), dtype=np.int64)
    coords = np.zeros((len(sample_keys), 2), dtype=np.int64)
    sample_splits: list[str | None] = []

    for sample_idx, key in enumerate(sample_keys):
        group = grouped[key]
        first = group[0]
        y[sample_idx] = coerce_label(first[label_col])
        if row_col and col_col:
            coords[sample_idx, 0] = int(float(first[row_col]))
            coords[sample_idx, 1] = int(float(first[col_col]))
        else:
            coords[sample_idx, 0] = sample_idx
        sample_splits.append(first.get(split_col, "").lower() if split_col else None)
        for row in group:
            if row.get(time_col, "") == "":
                continue
            time_idx = doy_index[coerce_doy(row[time_col])]
            if out_channels == ["VV", "VH", *S2_BANDS]:
                for channel, out_idx in (("VV", 0), ("VH", 1)):
                    value = parse_float(row.get(channel))
                    if math.isfinite(value):
                        x[sample_idx, time_idx, out_idx] = value
                        obs_mask[sample_idx, time_idx, out_idx] = True
                for channel_idx, channel in enumerate(S2_BANDS, start=2):
                    value = parse_float(row.get(channel))
                    if math.isfinite(value):
                        x[sample_idx, time_idx, channel_idx] = value
                        obs_mask[sample_idx, time_idx, channel_idx] = True
            else:
                for channel_idx, channel in enumerate(channels):
                    value = parse_float(row.get(channel))
                    if math.isfinite(value):
                        x[sample_idx, time_idx, channel_idx] = value
                        obs_mask[sample_idx, time_idx, channel_idx] = True
    return YearArrays(x, obs_mask, doy, y, coords, out_channels, sample_splits)


def parse_wide_feature(col: str) -> tuple[int, str] | None:
    normalized = norm(col)
    date_match = re.search(r"(20\d{2})(\d{2})(\d{2})", normalized)
    if date_match:
        doy = int(datetime.strptime(date_match.group(0), "%Y%m%d").strftime("%j"))
        channel = normalized.replace(date_match.group(0), "").strip("_")
        return doy, channel or col
    doy_match = re.search(r"(?:^|_)doy_?(\d{1,3})(?:_|$)", normalized)
    if doy_match:
        channel = re.sub(r"(?:^|_)doy_?\d{1,3}(?:_|$)", "_", normalized).strip("_")
        return int(doy_match.group(1)), channel or col
    trailing = re.search(r"_(\d{1,3})$", normalized)
    if trailing:
        doy = int(trailing.group(1))
        if 1 <= doy <= 366:
            return doy, normalized[: trailing.start()].strip("_") or col
    return None


def build_wide_arrays(columns: list[str], rows: list[dict[str, str]], args: argparse.Namespace) -> YearArrays:
    sample_col = find_column(columns, ["sample_id", "id", "field_id", "pixel_id"], args.sample_id_col)
    label_col = find_column(columns, ["y", "label", "class", "crop_id", "crop"], args.label_col)
    row_col = find_column(columns, ["row", "r"], args.row_col)
    col_col = find_column(columns, ["col", "c"], args.col_col)
    split_col = find_column(columns, ["split"])
    if label_col is None:
        raise ValueError("Could not infer label column. Pass --label-col.")
    exclude = {c for c in [sample_col, label_col, row_col, col_col, split_col, "year"] if c}
    specs = []
    for col in columns:
        if col in exclude or not is_numeric_column(rows, col):
            continue
        parsed = parse_wide_feature(col)
        if parsed is not None:
            specs.append((col, parsed[0], parsed[1]))
    if not specs:
        raise ValueError("Could not infer wide columns. Use names like B02_doy123, B02_123, or B02_20240502.")

    timeline = sorted({doy for _, doy, _ in specs})
    channels = sorted({channel for _, _, channel in specs})
    doy_index = {doy: idx for idx, doy in enumerate(timeline)}
    channel_index = {channel: idx for idx, channel in enumerate(channels)}
    x = np.zeros((len(rows), len(timeline), len(channels)), dtype=np.float32)
    obs_mask = np.zeros_like(x, dtype=bool)
    y = np.zeros((len(rows),), dtype=np.int64)
    coords = np.zeros((len(rows), 2), dtype=np.int64)
    sample_splits: list[str | None] = []

    for sample_idx, row in enumerate(rows):
        y[sample_idx] = coerce_label(row[label_col])
        if row_col and col_col:
            coords[sample_idx, 0] = int(float(row[row_col]))
            coords[sample_idx, 1] = int(float(row[col_col]))
        else:
            coords[sample_idx, 0] = sample_idx
        sample_splits.append(row.get(split_col, "").lower() if split_col else None)
        for col, doy_value, channel in specs:
            value = parse_float(row.get(col))
            if math.isfinite(value):
                x[sample_idx, doy_index[doy_value], channel_index[channel]] = value
                obs_mask[sample_idx, doy_index[doy_value], channel_index[channel]] = True
    doy = np.asarray(timeline, dtype=np.int16)[None, :].repeat(len(rows), axis=0)
    return YearArrays(x, obs_mask, doy, y, coords, channels, sample_splits)


def remap_labels(y: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    labels = sorted(int(value) for value in np.unique(y))
    mapping = {str(old): new for new, old in enumerate(labels)}
    return np.asarray([mapping[str(int(value))] for value in y], dtype=np.int64), mapping


def split_indices(arrays: YearArrays, args: argparse.Namespace) -> dict[str, np.ndarray]:
    explicit = {
        split: np.asarray([idx for idx, value in enumerate(arrays.sample_splits) if value == split], dtype=np.int64)
        for split in ("train", "val", "test")
    }
    if all(indices.size > 0 for indices in explicit.values()):
        return explicit

    total = args.train_fraction + args.val_fraction + args.test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    rng = np.random.default_rng(args.seed)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    for label in sorted(np.unique(arrays.y).tolist()):
        label_idx = np.flatnonzero(arrays.y == label)
        rng.shuffle(label_idx)
        train_end = int(round(label_idx.size * args.train_fraction))
        val_end = train_end + int(round(label_idx.size * args.val_fraction))
        train_parts.append(label_idx[:train_end])
        val_parts.append(label_idx[train_end:val_end])
        test_parts.append(label_idx[val_end:])
    return {
        "train": np.sort(np.concatenate(train_parts).astype(np.int64)),
        "val": np.sort(np.concatenate(val_parts).astype(np.int64)),
        "test": np.sort(np.concatenate(test_parts).astype(np.int64)),
    }


def write_h5(path: Path, arrays: YearArrays, splits: dict[str, np.ndarray], metadata: dict[str, object], overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        for split_name, indices in splits.items():
            group = h5.create_group(split_name)
            group.create_dataset("x", data=arrays.x[indices], compression="gzip", compression_opts=4)
            group.create_dataset("obs_mask", data=arrays.obs_mask[indices], compression="gzip", compression_opts=4)
            group.create_dataset("doy", data=arrays.doy[indices], compression="gzip", compression_opts=4)
            group.create_dataset("y", data=arrays.y[indices], compression="gzip", compression_opts=4)
            group.create_dataset("coords", data=arrays.coords[indices], compression="gzip", compression_opts=4)
        meta = h5.create_group("metadata")
        for key, value in metadata.items():
            meta.attrs[key] = json.dumps(value) if isinstance(value, (dict, list, tuple)) else value


def class_distribution(y: np.ndarray) -> list[int]:
    if y.size == 0:
        return []
    return [int((y == label).sum()) for label in range(int(y.max()) + 1)]


def build_year(year: int, paths: list[Path], args: argparse.Namespace) -> dict[str, object]:
    columns, rows = read_csvs(paths, args.sensor_filter)
    year_col = find_column(columns, ["year"])
    if year_col:
        rows = [row for row in rows if int(float(row[year_col])) == year]
    arrays = build_long_arrays(columns, rows, args) if is_long_table(columns) else build_wide_arrays(columns, rows, args)
    label_mapping: dict[str, int] = {}
    if args.remap_labels:
        arrays.y, label_mapping = remap_labels(arrays.y)
    splits = split_indices(arrays, args)
    out_path = args.out_root / str(year) / f"{args.site_token}_{year}_quality_randomsplit_timeseries.h5"
    metadata = {
        "site_token": args.site_token,
        "year": int(year),
        "format": "quality_randomsplit_timeseries",
        "source_csv": [str(path) for path in paths],
        "channel_names": arrays.channel_names,
        "label_mapping": label_mapping,
    }
    write_h5(out_path, arrays, splits, metadata, args.overwrite)
    return {
        "year": year,
        "h5_path": str(out_path),
        "source_csv": [str(path) for path in paths],
        "samples": int(arrays.y.size),
        "steps": int(arrays.x.shape[1]),
        "channels": int(arrays.x.shape[2]),
        "channel_names": arrays.channel_names,
        "splits": {
            split: {
                "samples": int(indices.size),
                "distribution": class_distribution(arrays.y[indices]),
            }
            for split, indices in splits.items()
        },
    }


def main() -> None:
    args = parse_args()
    by_year = csv_paths_by_year(args.input_root, set(args.years) if args.years else None, args.sensor_filter)
    if not by_year:
        raise FileNotFoundError(f"No CSV files with a year like 2024 found under {args.input_root}")
    outputs = [build_year(year, paths, args) for year, paths in sorted(by_year.items())]
    manifest_path = args.out_root / f"{args.site_token}_csv_to_h5_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"outputs": outputs}, indent=2), encoding="ascii")
    print(json.dumps({"manifest": str(manifest_path), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
