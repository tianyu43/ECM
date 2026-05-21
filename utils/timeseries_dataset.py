from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import rasterio
import torch
from rasterio.transform import xy
from rasterio.windows import Window
from torch import Tensor


@dataclass(frozen=True)
class SceneInfo:
    path: Path
    sensor: str
    date_token: str
    doy: int


@dataclass
class PixelBatch:
    x: Tensor
    obs_mask: Tensor
    doy: Tensor
    y: Tensor
    coords: Tensor
    class_ids: Tensor
    timeline: List[str]


@dataclass
class SplitPixelBatch:
    train: PixelBatch
    val: PixelBatch
    test: PixelBatch


@dataclass(frozen=True)
class QualitySelectionStats:
    candidate_count: int
    selected_count: int
    train_count: int
    val_count: int
    test_count: int
    quality_mean_candidates: float
    quality_mean_selected: float
    quality_min_selected: float
    quality_max_selected: float


@dataclass(frozen=True)
class SiteLabelConfig:
    kept_cdl_labels: Tuple[int, ...]
    class_names: Tuple[str, ...]
    include_others: bool = True


SITE_LABEL_CONFIGS: Dict[str, SiteLabelConfig] = {
    "site_IA": SiteLabelConfig(
        kept_cdl_labels=(1, 5),
        class_names=("others", "maize", "soybean"),
    ),
    "site_OH": SiteLabelConfig(
        kept_cdl_labels=(1, 5),
        class_names=("others", "maize", "soybean"),
    ),
    # site_MS fixed mapping requested by user:
    # others=0, maize(CDL 1)=1, soybean(CDL 5)=2, rice(CDL 3)=3, cotton(CDL 2)=4.
    "site_MS": SiteLabelConfig(
        kept_cdl_labels=(1, 5, 3, 2),
        class_names=("others", "maize", "soybean", "rice", "cotton"),
    ),
    "site_AK": SiteLabelConfig(
        kept_cdl_labels=(1, 5, 3, 2),
        class_names=("others", "maize", "soybean", "rice", "cotton"),
    ),
    "site_SB": SiteLabelConfig(
        kept_cdl_labels=(1110, 1130, 1410, 1430),
        class_names=("wheat", "maize", "sunflower", "rapeseed"),
        include_others=False,
    ),
}


def parse_scene_name(path: Path) -> SceneInfo:
    stem = path.stem
    if stem.endswith(")") and " (" in stem:
        stem = stem[: stem.rfind(" (")]
    parts = stem.split("_")
    if len(parts) == 5:
        _, _, sensor, date_token, doy = parts
    elif len(parts) == 4:
        _, _, date_token, doy = parts
        sensor = "S2"
    else:
        raise ValueError(f"Unexpected scene name: {path.name}")
    return SceneInfo(
        path=path,
        sensor=sensor,
        date_token=date_token,
        doy=int(doy),
    )


def resolve_year_dir(data_root: Path, site: str, year: int) -> Path:
    candidates = [
        data_root / site / str(year),
        data_root / site / f"{site}_{year}",
        data_root / f"{site}_{year}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve year directory for site={site} year={year} under {data_root}")


def resolve_label_path(data_root: Path, site: str, year: int) -> Path:
    year_dir = resolve_year_dir(data_root, site, year)
    candidates = [
        year_dir / "label" / f"CDL_{year}.tif",
        year_dir / f"CDL_{year}.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve CDL label for site={site} year={year} under {year_dir}")


def resolve_confidence_path(data_root: Path, site: str, year: int) -> Path:
    year_dir = resolve_year_dir(data_root, site, year)
    candidates = [
        year_dir / "label" / f"conf_{year}.tif",
        year_dir / f"conf_{year}.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Unable to resolve confidence raster for site={site} year={year} under {year_dir}")


def get_site_label_config(site: str) -> SiteLabelConfig:
    try:
        return SITE_LABEL_CONFIGS[site]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported site={site}. Expected one of {sorted(SITE_LABEL_CONFIGS.keys())}"
        ) from exc


def list_scenes(data_root: Path, site: str, year: int, sensor: str) -> List[SceneInfo]:
    year_dir = resolve_year_dir(data_root, site, year)
    nested_dir = year_dir / sensor
    if nested_dir.exists():
        scenes = [parse_scene_name(path) for path in sorted(nested_dir.glob("*.tif"))]
    else:
        scenes = []
        dedup: Dict[Tuple[str, str, int], SceneInfo] = {}
        for path in sorted(year_dir.glob("*.tif")):
            if path.name.startswith(("CDL_", "conf_")):
                continue
            try:
                scene = parse_scene_name(path)
            except ValueError:
                continue
            if scene.sensor != sensor:
                continue
            key = (scene.sensor, scene.date_token, scene.doy)
            prev = dedup.get(key)
            if prev is None or len(path.name) < len(prev.path.name):
                dedup[key] = scene
        scenes = list(dedup.values())
    return sorted(scenes, key=lambda x: x.date_token)


def build_timeline(s1_scenes: Sequence[SceneInfo], s2_scenes: Sequence[SceneInfo]) -> List[Tuple[str, int]]:
    merged = {(scene.date_token, scene.doy) for scene in s1_scenes}
    merged.update((scene.date_token, scene.doy) for scene in s2_scenes)
    return sorted(merged, key=lambda x: x[0])


def sample_labeled_pixels(
    label_path: Path,
    class_ids: Sequence[int],
    max_samples_per_class: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)

    coords: List[np.ndarray] = []
    out_labels: List[np.ndarray] = []
    for class_id in class_ids:
        rc = np.argwhere((labels == class_id) & cdl_same_class_8_neighbor_mask(labels))
        if rc.size == 0:
            continue
        take = min(max_samples_per_class, rc.shape[0])
        picked = rc[rng.choice(rc.shape[0], size=take, replace=False)]
        coords.append(picked)
        out_labels.append(np.full(take, class_id, dtype=np.int64))

    if not coords:
        raise ValueError(f"No labeled pixels found for classes {list(class_ids)} in {label_path}")

    return np.concatenate(coords, axis=0), np.concatenate(out_labels, axis=0)


def sample_three_class_pixels(
    label_path: Path,
    train_per_class: int,
    test_per_class: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)

    class_specs = [
        ("others", (labels > 0) & (labels != 1) & (labels != 5)),
        ("maize", labels == 1),
        ("soybean", labels == 5),
    ]

    train_coords: List[np.ndarray] = []
    test_coords: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    test_labels: List[np.ndarray] = []

    for class_idx, (_, class_mask) in enumerate(class_specs):
        rc = np.argwhere(class_mask)
        need = train_per_class + test_per_class
        if rc.shape[0] < need:
            raise ValueError(
                f"Not enough samples for class index {class_idx}: required {need}, found {rc.shape[0]}"
            )
        picked = rc[rng.choice(rc.shape[0], size=need, replace=False)]
        train_part = picked[:train_per_class]
        test_part = picked[train_per_class:]
        train_coords.append(train_part)
        test_coords.append(test_part)
        train_labels.append(np.full(train_per_class, class_idx, dtype=np.int64))
        test_labels.append(np.full(test_per_class, class_idx, dtype=np.int64))

    class_names = [name for name, _ in class_specs]
    return (
        np.concatenate(train_coords, axis=0),
        np.concatenate(train_labels, axis=0),
        np.concatenate(test_coords, axis=0),
        np.concatenate(test_labels, axis=0),
        class_names,
    )


def sample_random_three_class_split(
    label_path: Path,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)

    valid_mask = (labels > 0) & cdl_same_class_8_neighbor_mask(labels)
    coords = np.argwhere(valid_mask)
    if coords.size == 0:
        raise ValueError(f"No valid labeled pixels found in {label_path}")

    total = coords.shape[0]
    train_n = int(total * train_fraction)
    val_n = int(total * val_fraction)
    test_n = int(total * test_fraction)
    need = train_n + val_n + test_n
    if need <= 0:
        raise ValueError("Requested split sizes are all zero")
    if need > total:
        raise ValueError(f"Requested {need} samples but only {total} valid pixels exist")

    picked_idx = rng.choice(total, size=need, replace=False)
    picked = coords[picked_idx]
    picked_labels = labels[picked[:, 0], picked[:, 1]]

    mapped = np.full(need, 0, dtype=np.int64)
    mapped[picked_labels == 1] = 1
    mapped[picked_labels == 5] = 2

    train_coords = picked[:train_n]
    train_y = mapped[:train_n]
    val_coords = picked[train_n : train_n + val_n]
    val_y = mapped[train_n : train_n + val_n]
    test_coords = picked[train_n + val_n :]
    test_y = mapped[train_n + val_n :]
    class_names = ["others", "maize", "soybean"]
    return train_coords, train_y, val_coords, val_y, test_coords, test_y, class_names


def sample_random_three_class_candidates(
    label_path: Path,
    sample_count: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rng = np.random.default_rng(seed)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)

    valid_mask = (labels > 0) & cdl_same_class_8_neighbor_mask(labels)
    coords = np.argwhere(valid_mask)
    if coords.size == 0:
        raise ValueError(f"No valid labeled pixels found in {label_path}")

    total = coords.shape[0]
    take = min(sample_count, total)
    picked_idx = rng.choice(total, size=take, replace=False)
    picked = coords[picked_idx]
    picked_labels = labels[picked[:, 0], picked[:, 1]]

    mapped = np.full(take, 0, dtype=np.int64)
    mapped[picked_labels == 1] = 1
    mapped[picked_labels == 5] = 2
    class_names = ["others", "maize", "soybean"]
    return picked, mapped, class_names


def map_cdl_to_three_class(labels: np.ndarray) -> np.ndarray:
    mapped = np.full(labels.shape, 0, dtype=np.int64)
    mapped[labels == 1] = 1
    mapped[labels == 5] = 2
    return mapped


def map_cdl_to_site_classes(labels: np.ndarray, site: str) -> Tuple[np.ndarray, List[str]]:
    cfg = get_site_label_config(site)
    mapped = np.zeros(labels.shape, dtype=np.int64)
    start_idx = 1 if cfg.include_others else 0
    for class_idx, cdl_label in enumerate(cfg.kept_cdl_labels, start=start_idx):
        mapped[labels == cdl_label] = class_idx
    return mapped, list(cfg.class_names)


def cdl_same_class_8_neighbor_mask(labels: np.ndarray) -> np.ndarray:
    """Return pixels whose full 3x3 CDL neighborhood has the same raw label."""
    if labels.ndim != 2:
        raise ValueError(f"CDL labels must be 2-D, got shape={labels.shape}")
    labels_i64 = labels.astype(np.int64, copy=False)
    padded = np.pad(labels_i64, pad_width=1, mode="constant", constant_values=-1)
    same = labels_i64 > 0
    height, width = labels_i64.shape
    for row_offset in (-1, 0, 1):
        for col_offset in (-1, 0, 1):
            if row_offset == 0 and col_offset == 0:
                continue
            neighbor = padded[
                1 + row_offset : 1 + row_offset + height,
                1 + col_offset : 1 + col_offset + width,
            ]
            same &= neighbor == labels_i64
    return same


def _coords_to_xy(transform, coords_rc: np.ndarray) -> List[Tuple[float, float]]:
    rows = coords_rc[:, 0].tolist()
    cols = coords_rc[:, 1].tolist()
    xs, ys = xy(transform, rows, cols, offset="center")
    return list(zip(xs, ys))


VALID_OBSERVATION_RULE = "finite_nonzero"


def _default_read_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(8, cpu_count))


def _emit_progress(progress_hook: Optional[Callable[[str], None]], message: str) -> None:
    if progress_hook is not None:
        progress_hook(message)


def _read_scene_valid_mask(scene_path: Path, label_mask: np.ndarray) -> np.ndarray:
    valid = np.zeros(label_mask.shape, dtype=bool)
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(scene_path) as ds:
            for _, window in ds.block_windows(1):
                data = ds.read(window=window, out_dtype="float32")
                block_valid = np.any(np.isfinite(data) & (data != 0), axis=0)
                row_start = int(window.row_off)
                col_start = int(window.col_off)
                row_end = row_start + int(window.height)
                col_end = col_start + int(window.width)
                valid[row_start:row_end, col_start:col_end] = block_valid
    return valid & label_mask


def _load_sensor_stack(
    scenes: Sequence[SceneInfo],
    coords_rc: np.ndarray,
    time_index: Dict[str, int],
    x: np.ndarray,
    obs_mask: np.ndarray,
    channel_slice: slice,
    scale: float,
    progress_hook: Optional[Callable[[str], None]] = None,
    max_workers: Optional[int] = None,
) -> None:
    if not scenes:
        _emit_progress(progress_hook, "no scenes found for requested sensor stack")
        return
    worker_count = min(len(scenes), max_workers or _default_read_workers())
    _emit_progress(
        progress_hook,
        f"start loading sensor={scenes[0].sensor} scenes={len(scenes)} samples={coords_rc.shape[0]} "
        f"workers={worker_count} gdal_threads=ALL_CPUS scale={scale} read_mode=windowed",
    )
    if worker_count <= 1:
        for scene_idx, scene in enumerate(scenes, start=1):
            tidx = time_index[scene.date_token]
            _emit_progress(
                progress_hook,
                f"start reading {scene.sensor} scene {scene_idx}/{len(scenes)} "
                f"{scene.path.name} -> timeline_idx={tidx}",
            )
            values, valid = _read_scene_samples(scene.path, coords_rc)
            x[:, tidx, channel_slice] = values / scale
            obs_mask[:, tidx, channel_slice] = valid
            _emit_progress(
                progress_hook,
                f"finished reading {scene.sensor} scene {scene_idx}/{len(scenes)} "
                f"{scene.path.name} valid_ratio={float(valid.mean()):.4f}",
            )
        _emit_progress(progress_hook, f"finished loading sensor={scenes[0].sensor}")
        return

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        _emit_progress(
            progress_hook,
            f"submitted {len(scenes)} parallel read tasks for sensor={scenes[0].sensor}",
        )
        future_map = {
            executor.submit(_read_scene_samples, scene.path, coords_rc): (scene_idx, scene)
            for scene_idx, scene in enumerate(scenes, start=1)
        }
        for future in as_completed(future_map):
            scene_idx, scene = future_map[future]
            tidx = time_index[scene.date_token]
            values, valid = future.result()
            x[:, tidx, channel_slice] = values / scale
            obs_mask[:, tidx, channel_slice] = valid
            _emit_progress(
                progress_hook,
                f"finished reading {scene.sensor} scene {scene_idx}/{len(scenes)} "
                f"{scene.path.name} -> timeline_idx={tidx} valid_ratio={float(valid.mean()):.4f}",
            )
    _emit_progress(progress_hook, f"finished loading sensor={scenes[0].sensor}")


def _read_scene_samples(scene_path: Path, coords_rc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
        with rasterio.open(scene_path) as ds:
            band_count = ds.count
            sample_count = int(coords_rc.shape[0])
            values = np.zeros((sample_count, band_count), dtype=np.float32)
            valid = np.zeros((sample_count, band_count), dtype=bool)

            if ds.block_shapes:
                block_h, block_w = ds.block_shapes[0]
            else:
                block_h, block_w = 512, 512

            block_groups: Dict[Tuple[int, int], List[int]] = {}
            for idx, (row, col) in enumerate(coords_rc):
                key = (int(row) // block_h, int(col) // block_w)
                block_groups.setdefault(key, []).append(idx)

            for (block_row, block_col), idxs in block_groups.items():
                row_off = block_row * block_h
                col_off = block_col * block_w
                height = min(block_h, ds.height - row_off)
                width = min(block_w, ds.width - col_off)
                window = Window(col_off=col_off, row_off=row_off, width=width, height=height)
                data = ds.read(window=window, out_dtype="float32")

                block_coords = coords_rc[idxs]
                rel_rows = block_coords[:, 0] - row_off
                rel_cols = block_coords[:, 1] - col_off

                block_values = np.moveaxis(data[:, rel_rows, rel_cols], 0, 1)
                finite_nonzero = np.isfinite(block_values) & (block_values != 0)
                block_valid = finite_nonzero
                block_values = np.where(finite_nonzero, block_values, 0.0).astype(np.float32, copy=False)

                values[idxs] = block_values
                valid[idxs] = block_valid
    return values, valid


def _scene_signature(scenes: Sequence[SceneInfo]) -> List[Dict[str, object]]:
    return [
        {
            "name": scene.path.name,
            "size": int(scene.path.stat().st_size),
            "mtime_ns": int(scene.path.stat().st_mtime_ns),
        }
        for scene in scenes
    ]


def _cube_scene_key(scene: SceneInfo) -> str:
    return f"{scene.sensor}_{scene.date_token}_{scene.doy}"


def _year_scene_cube_path(cache_dir: Path, site: str, year: int) -> Path:
    site_token = site.split("_", 1)[-1].lower()
    return cache_dir / f"{site_token}_{year}_year_scene_cube.h5"


def _build_valid_pixel_year_cube(
    data_root: Path,
    cache_dir: Path,
    site: str,
    year: int,
    s2_scale: float = 10000.0,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cube_path = _year_scene_cube_path(cache_dir, site, year)
    label_path = resolve_label_path(data_root, site, year)
    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    timeline = build_timeline(s1_scenes, s2_scenes)
    time_index = {item[0]: idx for idx, item in enumerate(timeline)}

    if cube_path.exists():
        try:
            with h5py.File(cube_path, "r") as f:
                meta = f["metadata"].attrs
                expected_scene_count = len(s1_scenes) + len(s2_scenes)
                if (
                    int(meta["year"]) == year
                    and str(meta["site"]) == site
                    and int(meta["scene_count"]) == expected_scene_count
                    and int(meta["timeline_steps"]) == len(timeline)
                ):
                    _emit_progress(progress_hook, f"loaded existing year cube {cube_path}")
                    return cube_path
        except Exception:
            _emit_progress(progress_hook, f"year cube invalid, rebuilding {cube_path}")

    if cube_path.exists():
        cube_path.unlink()

    _emit_progress(progress_hook, f"building year scene cube target={cube_path}")
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)
    valid_mask = (labels > 0) & cdl_same_class_8_neighbor_mask(labels)
    coords = np.argwhere(valid_mask).astype(np.int32, copy=False)
    mapped_y, class_names = map_cdl_to_site_classes(labels[valid_mask], site)
    pixel_count = int(coords.shape[0])
    _emit_progress(
        progress_hook,
        f"year cube valid labeled pixels={pixel_count} timeline_steps={len(timeline)} "
        f"s1_scenes={len(s1_scenes)} s2_scenes={len(s2_scenes)}",
    )

    with h5py.File(cube_path, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["site"] = site
        meta.attrs["year"] = year
        meta.attrs["pixel_count"] = pixel_count
        meta.attrs["timeline_steps"] = len(timeline)
        meta.attrs["scene_count"] = len(s1_scenes) + len(s2_scenes)
        meta.attrs["s2_scale"] = float(s2_scale)
        meta.attrs["class_names_json"] = json.dumps(class_names)
        meta.attrs["timeline_json"] = json.dumps([date_token for date_token, _ in timeline])

        f.create_dataset("coords", data=coords, compression="gzip", compression_opts=4)
        f.create_dataset("y", data=mapped_y.astype(np.int64, copy=False), compression="gzip", compression_opts=4)
        f.create_dataset(
            "doy",
            data=np.asarray([doy for _, doy in timeline], dtype=np.int16),
            compression="gzip",
            compression_opts=4,
        )
        scenes_grp = f.create_group("scenes")

        all_scenes = [*s1_scenes, *s2_scenes]
        for scene_idx, scene in enumerate(all_scenes, start=1):
            timeline_idx = time_index[scene.date_token]
            _emit_progress(
                progress_hook,
                f"cube scene {scene_idx}/{len(all_scenes)} start sensor={scene.sensor} "
                f"name={scene.path.name} timeline_idx={timeline_idx}",
            )
            with rasterio.Env(GDAL_NUM_THREADS="ALL_CPUS"):
                with rasterio.open(scene.path) as ds:
                    values_full = ds.read(out_dtype="float32")
            values = np.moveaxis(values_full[:, coords[:, 0], coords[:, 1]], 0, 1)
            valid = np.isfinite(values) & (values != 0)
            values = np.where(valid, values, 0.0).astype(np.float32, copy=False)
            if scene.sensor == "S2":
                values /= s2_scale

            scene_grp = scenes_grp.create_group(_cube_scene_key(scene))
            scene_grp.attrs["sensor"] = scene.sensor
            scene_grp.attrs["timeline_idx"] = timeline_idx
            scene_grp.attrs["date_token"] = scene.date_token
            scene_grp.attrs["doy"] = scene.doy
            scene_grp.create_dataset("values", data=values, compression="gzip", compression_opts=4)
            scene_grp.create_dataset("valid", data=valid.astype(np.uint8, copy=False), compression="gzip", compression_opts=4)
            _emit_progress(
                progress_hook,
                f"cube scene {scene_idx}/{len(all_scenes)} done sensor={scene.sensor} "
                f"name={scene.path.name} valid_ratio={float(valid.mean()):.4f}",
            )

    _emit_progress(progress_hook, f"year cube build complete {cube_path}")
    return cube_path


def _read_h5_rows(dataset: h5py.Dataset, indices: np.ndarray) -> np.ndarray:
    if indices.size == 0:
        return dataset[:0]
    order = np.argsort(indices)
    sorted_idx = indices[order]
    rows = dataset[sorted_idx]
    restore = np.empty_like(order)
    restore[order] = np.arange(order.size)
    return rows[restore]


def _materialize_samples_from_year_cube(
    cube_path: Path,
    selected_idx: np.ndarray,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], List[str]]:
    with h5py.File(cube_path, "r") as f:
        timeline = json.loads(f["metadata"].attrs["timeline_json"])
        class_names = json.loads(f["metadata"].attrs["class_names_json"])
        doy_1d = f["doy"][:].astype(np.int64, copy=False)
        coords = _read_h5_rows(f["coords"], selected_idx).astype(np.int64, copy=False)
        y = _read_h5_rows(f["y"], selected_idx).astype(np.int64, copy=False)
        scene_items = sorted(
            f["scenes"].items(),
            key=lambda item: (int(item[1].attrs["timeline_idx"]), item[0]),
        )

        sample_count = int(selected_idx.size)
        time_steps = int(doy_1d.shape[0])
        x = np.zeros((sample_count, time_steps, 12), dtype=np.float32)
        obs_mask = np.zeros((sample_count, time_steps, 12), dtype=bool)
        doy = np.broadcast_to(doy_1d.reshape(1, -1), (sample_count, time_steps)).copy()

        _emit_progress(
            progress_hook,
            f"materializing selected samples from year cube count={sample_count} scenes={len(scene_items)}",
        )
        for scene_pos, (_, scene_grp) in enumerate(scene_items, start=1):
            sensor = str(scene_grp.attrs["sensor"])
            timeline_idx = int(scene_grp.attrs["timeline_idx"])
            channel_slice = slice(0, 2) if sensor == "S1" else slice(2, 12)
            values = _read_h5_rows(scene_grp["values"], selected_idx)
            valid = _read_h5_rows(scene_grp["valid"], selected_idx).astype(bool, copy=False)
            x[:, timeline_idx, channel_slice] = values
            obs_mask[:, timeline_idx, channel_slice] = valid
            if scene_pos == 1 or scene_pos == len(scene_items) or scene_pos % 10 == 0:
                _emit_progress(
                    progress_hook,
                    f"materialized cube scene {scene_pos}/{len(scene_items)} sensor={sensor} timeline_idx={timeline_idx}",
                )
    return x, obs_mask, doy, y, coords, timeline, class_names


def _load_cube_labels(cube_path: Path) -> np.ndarray:
    with h5py.File(cube_path, "r") as f:
        return f["y"][:].astype(np.int64, copy=False)


def _compute_cube_s2_quality(cube_path: Path, progress_hook: Optional[Callable[[str], None]] = None) -> np.ndarray:
    with h5py.File(cube_path, "r") as f:
        pixel_count = int(f["coords"].shape[0])
        counts = np.zeros(pixel_count, dtype=np.uint16)
        s2_scenes = [
            scene_grp
            for _, scene_grp in sorted(f["scenes"].items(), key=lambda item: item[0])
            if str(scene_grp.attrs["sensor"]) == "S2"
        ]
        _emit_progress(progress_hook, f"computing cube S2 quality scenes={len(s2_scenes)} pixels={pixel_count}")
        for idx, scene_grp in enumerate(s2_scenes, start=1):
            valid = scene_grp["valid"][:].astype(bool, copy=False)
            counts += valid.any(axis=1).astype(np.uint16)
            if idx == 1 or idx == len(s2_scenes) or idx % 10 == 0:
                _emit_progress(progress_hook, f"cube S2 quality scene {idx}/{len(s2_scenes)}")
    return counts


def _load_sensor_count_cache(
    cache_dir: Optional[Path],
    year: int,
    label_path: Path,
    scenes: Sequence[SceneInfo],
    sensor: str,
) -> Optional[np.ndarray]:
    if cache_dir is None:
        return None
    sensor_tag = sensor.lower()
    counts_path = cache_dir / f"{sensor_tag}_valid_counts_{year}.npy"
    meta_path = cache_dir / f"{sensor_tag}_valid_counts_{year}.json"
    if not counts_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        label_stat = label_path.stat()
        expected = {
            "label_name": label_path.name,
            "label_size": int(label_stat.st_size),
            "label_mtime_ns": int(label_stat.st_mtime_ns),
            "sensor": sensor,
            "valid_observation_rule": VALID_OBSERVATION_RULE,
            "scenes": _scene_signature(scenes),
        }
        if meta != expected:
            return None
        return np.load(counts_path)
    except Exception:
        return None


def _save_sensor_count_cache(
    cache_dir: Optional[Path],
    year: int,
    label_path: Path,
    scenes: Sequence[SceneInfo],
    sensor: str,
    counts: np.ndarray,
) -> None:
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    sensor_tag = sensor.lower()
    counts_path = cache_dir / f"{sensor_tag}_valid_counts_{year}.npy"
    meta_path = cache_dir / f"{sensor_tag}_valid_counts_{year}.json"
    label_stat = label_path.stat()
    meta = {
        "label_name": label_path.name,
        "label_size": int(label_stat.st_size),
        "label_mtime_ns": int(label_stat.st_mtime_ns),
        "sensor": sensor,
        "valid_observation_rule": VALID_OBSERVATION_RULE,
        "scenes": _scene_signature(scenes),
    }
    np.save(counts_path, counts)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def _compute_sensor_valid_observation_counts(
    scenes: Sequence[SceneInfo],
    sensor: str,
    label_mask: np.ndarray,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> np.ndarray:
    counts = np.zeros(label_mask.shape, dtype=np.uint16)
    valid_rows, valid_cols = np.where(label_mask)
    worker_count = min(len(scenes), _default_read_workers())
    if worker_count <= 1:
        for scene_idx, scene in enumerate(scenes, start=1):
            if progress_hook is not None:
                progress_hook(
                    f"counting {sensor} validity scene {scene_idx}/{len(scenes)} {scene.path.name}"
                )
            valid = _read_scene_valid_mask(scene.path, label_mask)
            counts[valid_rows, valid_cols] += valid[valid_rows, valid_cols].astype(np.uint16)
        return counts

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_read_scene_valid_mask, scene.path, label_mask): (scene_idx, scene)
            for scene_idx, scene in enumerate(scenes, start=1)
        }
        for future in as_completed(future_map):
            scene_idx, scene = future_map[future]
            valid = future.result()
            counts[valid_rows, valid_cols] += valid[valid_rows, valid_cols].astype(np.uint16)
            if progress_hook is not None:
                progress_hook(
                    f"counting {sensor} validity scene {scene_idx}/{len(scenes)} {scene.path.name}"
                )
    return counts


def _build_feature_arrays(
    coords_rc: np.ndarray,
    s1_scenes: Sequence[SceneInfo],
    s2_scenes: Sequence[SceneInfo],
    s2_scale: float,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    timeline = build_timeline(s1_scenes, s2_scenes)
    time_index = {item[0]: idx for idx, item in enumerate(timeline)}
    _emit_progress(
        progress_hook,
        f"building feature arrays samples={coords_rc.shape[0]} timeline_steps={len(timeline)} "
        f"s1_scenes={len(s1_scenes)} s2_scenes={len(s2_scenes)}",
    )

    num_samples = coords_rc.shape[0]
    num_steps = len(timeline)
    num_channels = 12
    x = np.zeros((num_samples, num_steps, num_channels), dtype=np.float32)
    obs_mask = np.zeros((num_samples, num_steps, num_channels), dtype=bool)
    doy = np.zeros((num_samples, num_steps), dtype=np.int64)
    for idx, (_, day) in enumerate(timeline):
        doy[:, idx] = day

    _load_sensor_stack(
        scenes=s1_scenes,
        coords_rc=coords_rc,
        time_index=time_index,
        x=x,
        obs_mask=obs_mask,
        channel_slice=slice(0, 2),
        scale=1.0,
        progress_hook=progress_hook,
    )
    _load_sensor_stack(
        scenes=s2_scenes,
        coords_rc=coords_rc,
        time_index=time_index,
        x=x,
        obs_mask=obs_mask,
        channel_slice=slice(2, 12),
        scale=s2_scale,
        progress_hook=progress_hook,
    )

    _emit_progress(
        progress_hook,
        f"feature arrays ready samples={num_samples} timeline_steps={num_steps} channels={num_channels}",
    )

    return x, obs_mask, doy, [date_token for date_token, _ in timeline]


def _quality_score_from_obs_mask(obs_mask: np.ndarray) -> np.ndarray:
    s2_valid_count = obs_mask[:, :, 2:].any(axis=-1).sum(axis=1).astype(np.float32)
    return s2_valid_count


def _make_pixel_batch(
    x: np.ndarray,
    obs_mask: np.ndarray,
    doy: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    class_names: Sequence[str],
    timeline: List[str],
) -> PixelBatch:
    return PixelBatch(
        x=torch.from_numpy(x),
        obs_mask=torch.from_numpy(obs_mask),
        doy=torch.from_numpy(doy),
        y=torch.from_numpy(y),
        coords=torch.from_numpy(coords.astype(np.int64)),
        class_ids=torch.arange(len(class_names), dtype=torch.int64),
        timeline=timeline,
    )


def _build_split_batch(
    coords_rc: np.ndarray,
    y: np.ndarray,
    class_names: Sequence[str],
    timeline: List[str],
    s1_scenes: Sequence[SceneInfo],
    s2_scenes: Sequence[SceneInfo],
    s2_scale: float,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> PixelBatch:
    x, obs_mask, doy, _ = _build_feature_arrays(
        coords_rc=coords_rc,
        s1_scenes=s1_scenes,
        s2_scenes=s2_scenes,
        s2_scale=s2_scale,
        progress_hook=progress_hook,
    )
    return _make_pixel_batch(
        x=x,
        obs_mask=obs_mask,
        doy=doy,
        y=y,
        coords=coords_rc,
        class_names=class_names,
        timeline=timeline,
    )


def _slice_pixel_batch(
    x: np.ndarray,
    obs_mask: np.ndarray,
    doy: np.ndarray,
    y: np.ndarray,
    coords: np.ndarray,
    class_names: Sequence[str],
    timeline: List[str],
    start: int,
    end: int,
) -> PixelBatch:
    return _make_pixel_batch(
        x=x[start:end],
        obs_mask=obs_mask[start:end],
        doy=doy[start:end],
        y=y[start:end],
        coords=coords[start:end],
        class_names=class_names,
        timeline=timeline,
    )


def build_year_batch(
    data_root: Path,
    site: str,
    year: int,
    class_ids: Sequence[int],
    max_samples_per_class: int = 256,
    seed: int = 42,
    s2_scale: float = 10000.0,
) -> PixelBatch:
    label_path = data_root / site / str(year) / "label" / f"CDL_{year}.tif"
    coords_rc, labels = sample_labeled_pixels(label_path, class_ids, max_samples_per_class, seed=seed)

    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    timeline = build_timeline(s1_scenes, s2_scenes)
    time_index = {item[0]: idx for idx, item in enumerate(timeline)}

    num_samples = coords_rc.shape[0]
    num_steps = len(timeline)
    num_channels = 12

    x = np.zeros((num_samples, num_steps, num_channels), dtype=np.float32)
    obs_mask = np.zeros((num_samples, num_steps, num_channels), dtype=bool)
    doy = np.zeros((num_samples, num_steps), dtype=np.int64)
    for idx, (_, day) in enumerate(timeline):
        doy[:, idx] = day

    for scene in s1_scenes:
        tidx = time_index[scene.date_token]
        values, valid = _read_scene_samples(scene.path, coords_rc)
        x[:, tidx, :2] = values
        obs_mask[:, tidx, :2] = valid

    for scene in s2_scenes:
        tidx = time_index[scene.date_token]
        values, valid = _read_scene_samples(scene.path, coords_rc)
        values = values / s2_scale
        x[:, tidx, 2:] = values
        obs_mask[:, tidx, 2:] = valid

    class_id_to_index = {class_id: idx for idx, class_id in enumerate(class_ids)}
    y = np.array([class_id_to_index[int(label)] for label in labels], dtype=np.int64)

    return PixelBatch(
        x=torch.from_numpy(x),
        obs_mask=torch.from_numpy(obs_mask),
        doy=torch.from_numpy(doy),
        y=torch.from_numpy(y),
        coords=torch.from_numpy(coords_rc.astype(np.int64)),
        class_ids=torch.tensor(class_ids, dtype=torch.int64),
        timeline=[date_token for date_token, _ in timeline],
    )


def build_sequence_arrays_for_coords(
    data_root: Path,
    site: str,
    year: int,
    coords_rc: np.ndarray,
    s2_scale: float = 10000.0,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Dict[str, np.ndarray | List[str]]:
    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    x, obs_mask, doy, timeline = _build_feature_arrays(
        coords_rc=coords_rc,
        s1_scenes=s1_scenes,
        s2_scenes=s2_scenes,
        s2_scale=s2_scale,
        progress_hook=progress_hook,
    )
    return {
        "x": x,
        "obs_mask": obs_mask,
        "doy": doy,
        "timeline": timeline,
    }


def build_three_class_year_split(
    data_root: Path,
    site: str,
    year: int,
    train_per_class: int = 10000,
    test_per_class: int = 2000,
    seed: int = 42,
    s2_scale: float = 10000.0,
) -> SplitPixelBatch:
    label_path = resolve_label_path(data_root, site, year)
    train_coords, train_y, test_coords, test_y, class_names = sample_three_class_pixels(
        label_path=label_path,
        train_per_class=train_per_class,
        test_per_class=test_per_class,
        seed=seed,
    )

    all_coords = np.concatenate([train_coords, test_coords], axis=0)
    all_y = np.concatenate([train_y, test_y], axis=0)

    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    timeline = build_timeline(s1_scenes, s2_scenes)
    time_index = {item[0]: idx for idx, item in enumerate(timeline)}

    num_samples = all_coords.shape[0]
    num_steps = len(timeline)
    num_channels = 12
    x = np.zeros((num_samples, num_steps, num_channels), dtype=np.float32)
    obs_mask = np.zeros((num_samples, num_steps, num_channels), dtype=bool)
    doy = np.zeros((num_samples, num_steps), dtype=np.int64)
    for idx, (_, day) in enumerate(timeline):
        doy[:, idx] = day

    for scene in s1_scenes:
        tidx = time_index[scene.date_token]
        values, valid = _read_scene_samples(scene.path, all_coords)
        x[:, tidx, :2] = values
        obs_mask[:, tidx, :2] = valid

    for scene in s2_scenes:
        tidx = time_index[scene.date_token]
        values, valid = _read_scene_samples(scene.path, all_coords)
        values = values / s2_scale
        x[:, tidx, 2:] = values
        obs_mask[:, tidx, 2:] = valid

    split_idx = train_coords.shape[0]
    train_batch = PixelBatch(
        x=torch.from_numpy(x[:split_idx]),
        obs_mask=torch.from_numpy(obs_mask[:split_idx]),
        doy=torch.from_numpy(doy[:split_idx]),
        y=torch.from_numpy(all_y[:split_idx]),
        coords=torch.from_numpy(all_coords[:split_idx].astype(np.int64)),
        class_ids=torch.arange(len(class_names), dtype=torch.int64),
        timeline=[date_token for date_token, _ in timeline],
    )
    val_batch = PixelBatch(
        x=torch.from_numpy(x[split_idx:split_idx]),
        obs_mask=torch.from_numpy(obs_mask[split_idx:split_idx]),
        doy=torch.from_numpy(doy[split_idx:split_idx]),
        y=torch.from_numpy(all_y[split_idx:split_idx]),
        coords=torch.from_numpy(all_coords[split_idx:split_idx].astype(np.int64)),
        class_ids=torch.arange(len(class_names), dtype=torch.int64),
        timeline=[date_token for date_token, _ in timeline],
    )
    test_batch = PixelBatch(
        x=torch.from_numpy(x[split_idx:]),
        obs_mask=torch.from_numpy(obs_mask[split_idx:]),
        doy=torch.from_numpy(doy[split_idx:]),
        y=torch.from_numpy(all_y[split_idx:]),
        coords=torch.from_numpy(all_coords[split_idx:].astype(np.int64)),
        class_ids=torch.arange(len(class_names), dtype=torch.int64),
        timeline=[date_token for date_token, _ in timeline],
    )
    return SplitPixelBatch(train=train_batch, val=val_batch, test=test_batch)


def build_random_fraction_year_split(
    data_root: Path,
    site: str,
    year: int,
    train_fraction: float = 0.01,
    val_fraction: float = 0.002,
    test_fraction: float = 0.005,
    seed: int = 42,
    s2_scale: float = 10000.0,
    cache_dir: Optional[Path] = None,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> SplitPixelBatch:
    _ = cache_dir
    label_path = resolve_label_path(data_root, site, year)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)
    valid_mask = (labels > 0) & cdl_same_class_8_neighbor_mask(labels)
    total_valid = int(valid_mask.sum())
    train_coords, train_y, val_coords, val_y, test_coords, test_y, class_names = sample_random_three_class_split(
        label_path=label_path,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    train_n = train_coords.shape[0]
    val_n = val_coords.shape[0]
    test_n = test_coords.shape[0]
    all_coords = np.concatenate([train_coords, val_coords, test_coords], axis=0)
    all_y = np.concatenate([train_y, val_y, test_y], axis=0)

    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    timeline = [date_token for date_token, _ in build_timeline(s1_scenes, s2_scenes)]
    _emit_progress(
        progress_hook,
        f"random selection total_valid={total_valid} selected={all_coords.shape[0]} "
        f"train={train_n} val={val_n} test={test_n}",
    )
    _emit_progress(
        progress_hook,
        "building shared feature arrays for random split count="
        f"{all_coords.shape[0]} (single pass for train/val/test)",
    )
    all_x, all_obs_mask, all_doy, _ = _build_feature_arrays(
        coords_rc=all_coords,
        s1_scenes=s1_scenes,
        s2_scenes=s2_scenes,
        s2_scale=s2_scale,
        progress_hook=progress_hook,
    )
    split = SplitPixelBatch(
        train=_slice_pixel_batch(
            x=all_x,
            obs_mask=all_obs_mask,
            doy=all_doy,
            y=all_y,
            coords=all_coords,
            class_names=class_names,
            timeline=timeline,
            start=0,
            end=train_n,
        ),
        val=_slice_pixel_batch(
            x=all_x,
            obs_mask=all_obs_mask,
            doy=all_doy,
            y=all_y,
            coords=all_coords,
            class_names=class_names,
            timeline=timeline,
            start=train_n,
            end=train_n + val_n,
        ),
        test=_slice_pixel_batch(
            x=all_x,
            obs_mask=all_obs_mask,
            doy=all_doy,
            y=all_y,
            coords=all_coords,
            class_names=class_names,
            timeline=timeline,
            start=train_n + val_n,
            end=train_n + val_n + test_n,
        ),
    )
    _emit_progress(
        progress_hook,
        f"random split ready train={train_n} val={val_n} test={test_n} time_steps={split.train.x.shape[1]}",
    )
    return split


def build_quality_filtered_random_fraction_year_split(
    data_root: Path,
    site: str,
    year: int,
    train_fraction: float = 0.01,
    val_fraction: float = 0.002,
    test_fraction: float = 0.005,
    seed: int = 42,
    s2_scale: float = 10000.0,
    cache_dir: Optional[Path] = None,
    min_confidence: float = 0.0,
    allowed_labels: Optional[Sequence[int]] = None,
    fractions_from_eligible_pool: bool = False,
    progress_hook: Optional[Callable[[str], None]] = None,
) -> Tuple[SplitPixelBatch, QualitySelectionStats]:
    label_path = resolve_label_path(data_root, site, year)
    with rasterio.open(label_path) as ds:
        labels = ds.read(1)
    valid_mask = (labels > 0) & cdl_same_class_8_neighbor_mask(labels)
    allowed_labels_set = tuple(int(item) for item in (allowed_labels or ()))
    if allowed_labels_set:
        valid_mask &= np.isin(labels, np.asarray(allowed_labels_set, dtype=labels.dtype))
    if min_confidence > 0:
        conf_path = resolve_confidence_path(data_root, site, year)
        with rasterio.open(conf_path) as ds:
            confidence = ds.read(1)
        valid_mask &= confidence > float(min_confidence)
    total_valid = int(valid_mask.sum())

    s1_scenes = list_scenes(data_root, site, year, "S1")
    s2_scenes = list_scenes(data_root, site, year, "S2")
    s2_counts = _load_sensor_count_cache(
        cache_dir=cache_dir,
        year=year,
        label_path=label_path,
        scenes=s2_scenes,
        sensor="S2",
    )
    if s2_counts is None:
        _emit_progress(progress_hook, f"counting S2 valid observations total_valid={total_valid}")
        s2_counts = _compute_sensor_valid_observation_counts(
            scenes=s2_scenes,
            sensor="S2",
            label_mask=valid_mask,
            progress_hook=progress_hook,
        )
        _save_sensor_count_cache(
            cache_dir=cache_dir,
            year=year,
            label_path=label_path,
            scenes=s2_scenes,
            sensor="S2",
            counts=s2_counts,
        )
    else:
        _emit_progress(progress_hook, f"loaded cached S2 valid observation counts for year={year}")

    valid_s2_counts = s2_counts[valid_mask].astype(np.float32)
    train_threshold = float(np.quantile(valid_s2_counts, 0.75))
    val_test_threshold = float(np.quantile(valid_s2_counts, 0.25))
    train_pool_mask = valid_mask & (s2_counts >= train_threshold)
    val_test_pool_mask = valid_mask & (s2_counts >= val_test_threshold)
    train_pool_coords = np.argwhere(train_pool_mask)
    val_test_pool_coords = np.argwhere(val_test_pool_mask)
    if fractions_from_eligible_pool:
        train_pool_count = int(train_pool_coords.shape[0])
        train_n = int(train_pool_count * train_fraction)
    else:
        train_pool_count = total_valid
        train_n = int(train_pool_count * train_fraction)
    val_n = int(total_valid * val_fraction)
    test_n = int(total_valid * test_fraction)
    need = train_n + val_n + test_n
    if need <= 0:
        raise ValueError("Requested split sizes are all zero")
    if train_pool_coords.shape[0] < train_n:
        raise ValueError(
            f"S2-prioritized train pool too small for year={year}: "
            f"train_need={train_n}, eligible={train_pool_coords.shape[0]}, "
            f"s2_threshold={train_threshold}"
        )

    rng = np.random.default_rng(seed + year)
    train_perm = rng.choice(train_pool_coords.shape[0], size=train_n, replace=False)
    train_coords = train_pool_coords[train_perm]
    train_quality = s2_counts[train_coords[:, 0], train_coords[:, 1]].astype(np.float32)

    taken = {tuple(item) for item in train_coords.tolist()}
    remaining_coords = np.asarray(
        [coord for coord in val_test_pool_coords.tolist() if tuple(coord) not in taken],
        dtype=np.int64,
    )
    if remaining_coords.shape[0] < val_n + test_n:
        raise ValueError(
            f"Not enough top-75% S2 pixels for val/test after train selection in year={year}: "
            f"remaining={remaining_coords.shape[0]}, val_need={val_n}, test_need={test_n}, "
            f"s2_threshold={val_test_threshold}"
        )
    _emit_progress(
        progress_hook,
        f"quality filtering train_s2_valid_count_threshold={train_threshold:.4f} "
        f"val_test_s2_valid_count_threshold={val_test_threshold:.4f} "
        f"eligible_train_top25={train_pool_coords.shape[0]} "
        f"eligible_val_test_top75_after_train={remaining_coords.shape[0]} "
        f"train_selected={train_n} val_selected={val_n} test_selected={test_n} "
        f"min_confidence={min_confidence:.2f} allowed_labels={list(allowed_labels_set)} "
        f"fractions_from_eligible_pool={fractions_from_eligible_pool}",
    )

    remaining_perm = rng.choice(remaining_coords.shape[0], size=val_n + test_n, replace=False)
    remainder_coords = remaining_coords[remaining_perm]
    val_coords = remainder_coords[:val_n]
    test_coords = remainder_coords[val_n:]

    selected_coords = np.concatenate([train_coords, val_coords, test_coords], axis=0)
    selected_labels = labels[selected_coords[:, 0], selected_coords[:, 1]]
    selected_y, class_names = map_cdl_to_site_classes(selected_labels, site)

    timeline = [date_token for date_token, _ in build_timeline(s1_scenes, s2_scenes)]
    _emit_progress(
        progress_hook,
        f"building shared feature arrays for selected samples count={need} "
        f"(single pass for train/val/test)",
    )
    all_x, all_obs_mask, all_doy, _ = _build_feature_arrays(
        coords_rc=selected_coords,
        s1_scenes=s1_scenes,
        s2_scenes=s2_scenes,
        s2_scale=s2_scale,
        progress_hook=progress_hook,
    )

    split = SplitPixelBatch(
        train=_slice_pixel_batch(
            x=all_x,
            obs_mask=all_obs_mask,
            doy=all_doy,
            y=selected_y,
            coords=selected_coords,
            class_names=class_names,
            timeline=timeline,
            start=0,
            end=train_n,
        ),
        val=_slice_pixel_batch(
            x=all_x,
            obs_mask=all_obs_mask,
            doy=all_doy,
            y=selected_y,
            coords=selected_coords,
            class_names=class_names,
            timeline=timeline,
            start=train_n,
            end=train_n + val_n,
        ),
        test=_slice_pixel_batch(
            x=all_x,
            obs_mask=all_obs_mask,
            doy=all_doy,
            y=selected_y,
            coords=selected_coords,
            class_names=class_names,
            timeline=timeline,
            start=train_n + val_n,
            end=need,
        ),
    )
    stats = QualitySelectionStats(
        candidate_count=int(total_valid),
        selected_count=int(train_n),
        train_count=int(train_n),
        val_count=int(val_n),
        test_count=int(test_n),
        quality_mean_candidates=float(valid_s2_counts.mean()),
        quality_mean_selected=float(train_quality.mean()) if train_quality.size else 0.0,
        quality_min_selected=float(train_quality.min()) if train_quality.size else 0.0,
        quality_max_selected=float(train_quality.max()) if train_quality.size else 0.0,
    )
    return split, stats
