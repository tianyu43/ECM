from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import torch


def save_split_batch_to_h5(
    h5_path: Path,
    split_name: str,
    x: torch.Tensor,
    obs_mask: torch.Tensor,
    doy: torch.Tensor,
    y: torch.Tensor,
    coords: torch.Tensor,
) -> None:
    h5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "a") as f:
        if split_name in f:
            del f[split_name]
        grp = f.create_group(split_name)
        grp.create_dataset("x", data=x.numpy(), compression="gzip", compression_opts=4)
        grp.create_dataset("obs_mask", data=obs_mask.numpy(), compression="gzip", compression_opts=4)
        grp.create_dataset("doy", data=doy.numpy(), compression="gzip", compression_opts=4)
        grp.create_dataset("y", data=y.numpy(), compression="gzip", compression_opts=4)
        grp.create_dataset("coords", data=coords.numpy(), compression="gzip", compression_opts=4)


def save_metadata_to_h5(h5_path: Path, metadata: Dict[str, str | int | float]) -> None:
    with h5py.File(h5_path, "a") as f:
        meta = f.require_group("metadata")
        for key, value in metadata.items():
            if key in meta.attrs:
                del meta.attrs[key]
            meta.attrs[key] = value


def load_split_from_h5(h5_path: Path, split_name: str) -> Dict[str, torch.Tensor]:
    with h5py.File(h5_path, "r") as f:
        grp = f[split_name]
        return {
            "x": torch.from_numpy(grp["x"][:]),
            "obs_mask": torch.from_numpy(grp["obs_mask"][:]),
            "doy": torch.from_numpy(grp["doy"][:]),
            "y": torch.from_numpy(grp["y"][:]),
            "coords": torch.from_numpy(grp["coords"][:]),
        }


def read_metadata_from_h5(h5_path: Path) -> Dict[str, str]:
    with h5py.File(h5_path, "r") as f:
        attrs = f["metadata"].attrs
        return {key: attrs[key] for key in attrs.keys()}
