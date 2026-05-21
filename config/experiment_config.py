from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    raw_data_dir: Path
    processed_data_dir: Path
    experiments_dir: Path


def project_paths(root: Path | None = None) -> ProjectPaths:
    repo_root = root or Path(__file__).resolve().parent.parent
    return ProjectPaths(
        root=repo_root,
        raw_data_dir=Path(os.environ.get("ECM_RAW_DATA_DIR", repo_root / "Data" / "Data_raw")),
        processed_data_dir=Path(os.environ.get("ECM_PROCESSED_DATA_DIR", repo_root / "Data_perprocess")),
        experiments_dir=Path(os.environ.get("ECM_EXPERIMENTS_DIR", repo_root / "Experiments")),
    )


@dataclass(frozen=True)
class CommonTrainConfig:
    site: str = "site_IA"
    sensor_mode: str = "s1s2"
    train_years: tuple[int, ...] = (2020, 2021, 2022, 2023)
    test_year: int = 2024
    batch_size: int = 512
    epochs: int = 30
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    model_scale: str = "base"
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    seed: int = 3407
    log_interval: int = 40
    num_workers: int = 8
    early_min_doy: int = 150
    early_split_doy: int = 210
    early_max_doy: int = 270
    early_sampling_ratio: float = 0.6
    num_classes: int = 3
    time_shift_max_days: int = 5
    random_step_drop_prob: float = 0.10
    train_subset_fraction: float = 0.1
    val_subset_fraction: float = 0.1
    test_subset_fraction: float = 0.1

    @property
    def site_token(self) -> str:
        return self.site.split("_", 1)[-1].lower()

    def processed_site_dir(self, paths: ProjectPaths | None = None) -> Path:
        return (paths or project_paths()).processed_data_dir / self.site

    def experiment_site_dir(self, paths: ProjectPaths | None = None) -> Path:
        return (paths or project_paths()).experiments_dir / self.site

    def yearly_cache_path(self, year: int, paths: ProjectPaths | None = None) -> Path:
        return self.processed_site_dir(paths) / str(year) / f"{self.site_token}_{year}_quality_randomsplit_timeseries.h5"


@dataclass(frozen=True)
class ECMBaselineTrainConfig(CommonTrainConfig):
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    out_subdir: str = "ECM-baseline"
    tag: str = "ECM-baseline"
    experiment_site_dirname: str | None = None
    init_checkpoint: str | None = None
    early_cutoff_weight: float = 1.0
    mid_cutoff_weight: float = 1.0
    late_cutoff_weight: float = 1.0
    train_all_fixed_cutoffs: bool = False
    fixed_cutoff_step: int = 10


@dataclass(frozen=True)
class ECMMultiTrainConfig(ECMBaselineTrainConfig):
    out_subdir: str = "ECM-multi"
    tag: str = "ECM-multi"
    future_days: int = 30
    future_kd_weight: float = 0.5
    future_kd_temperature: float = 2.0
    future_proto_ce_weight: float = 0.7
    future_proto_temperature: float = 0.1
    future_pred_weight: float = 0.5
    aux_warmup_epochs: int = 5


@dataclass(frozen=True)
class ECMProtoTrainConfig(CommonTrainConfig):
    epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    out_subdir: str = "ECM-proto"
    tag: str = "ECM-proto"
    experiment_site_dirname: str | None = None
    init_checkpoint: str | None = None
    early_cutoff_weight: float = 1.0
    mid_cutoff_weight: float = 1.0
    late_cutoff_weight: float = 1.0
    prototypes_per_class: int = 4
    proto_temperature: float = 0.1
    proto_loss_weight: float = 0.5
    attn_loss_weight: float = 0.1
    proto_warmup_epochs: int = 10
    prototype_ema_momentum: float = 0.99
    kmeans_iters: int = 20
    prototype_batch_size: int = 1024
    fusion_mode: str = "gated_residual"
    future_query_tokens: int = 8
    future_query_step_days: int = 5
    future_query_max_doy: int = 270


@dataclass(frozen=True)
class ECMProtoV2TrainConfig(ECMProtoTrainConfig):
    out_subdir: str = "ECM-proto-v2"
    tag: str = "ECM-proto-v2"


@dataclass(frozen=True)
class ECMProtoFeatureTrainConfig(CommonTrainConfig):
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 5e-5
    out_subdir: str = "ECM-proto-feature"
    tag: str = "ECM-proto-feature"
    experiment_site_dirname: str | None = None
    early_cutoff_weight: float = 1.0
    mid_cutoff_weight: float = 1.0
    late_cutoff_weight: float = 1.0
    prototypes_per_class: int = 4
    proto_temperature: float = 0.1
    proto_loss_weight: float = 0.5
    attn_loss_weight: float = 0.1
    proto_warmup_epochs: int = 5
    prototype_update_epochs: int = 5
    kmeans_iters: int = 20
    prototype_batch_size: int = 512


MODEL_SCALE_PRESETS: dict[str, dict[str, int | float]] = {
    "small": {
        "d_model": 64,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "base": {
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 512,
        "dropout": 0.1,
    },
    "large": {
        "d_model": 256,
        "nhead": 8,
        "num_layers": 6,
        "dim_feedforward": 1024,
        "dropout": 0.1,
    },
}


def apply_model_scale(cfg: CommonTrainConfig, model_scale: str) -> CommonTrainConfig:
    if model_scale not in MODEL_SCALE_PRESETS:
        raise ValueError(f"unsupported model_scale={model_scale}")
    supported_fields = {field.name for field in fields(cfg)}
    scale_values = {
        key: value
        for key, value in MODEL_SCALE_PRESETS[model_scale].items()
        if key in supported_fields
    }
    return replace(cfg, **scale_values)


@dataclass(frozen=True)
class CacheBuildConfig:
    site: str = "site_IA"
    years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
    train_fraction: float = 0.01
    val_fraction: float = 0.002
    test_fraction: float = 0.005
    seed: int = 42
    sample_mode: str = "quality"
    min_confidence: float = 0.0
    allowed_labels: tuple[int, ...] = ()
    fractions_from_eligible_pool: bool = False

    @property
    def site_token(self) -> str:
        return self.site.split("_", 1)[-1].lower()

    def processed_site_dir(self, paths: ProjectPaths | None = None) -> Path:
        return (paths or project_paths()).processed_data_dir / self.site
