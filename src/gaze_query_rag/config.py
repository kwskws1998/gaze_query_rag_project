from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    artifacts_dir: Path = Path("artifacts")
    onestop_ia_path: Path = Path(
        "OneStop-Eye-Movements/data/OneStop/osfstorage-archive (1)/ia_Paragraph.csv.zip"
    )
    onestop_qa_json_path: Path = Path(
        "OneStop-Eye-Movements/data_preprocessing/onestop_qa.json"
    )


@dataclass(frozen=True)
class DataConfig:
    qa_dataset_name: str = "malmaud/onestop_qa"
    qa_split: str | None = None
    max_examples: int | None = None
    max_readers_per_example: int | None = None


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "sentence"
    max_words: int = 80
    stride: int = 0


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str = "intfloat/e5-large-v2"
    generator_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    smoke_generator_fallback_name: str | None = None
    device: str = "auto"
    dtype: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 13
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    models: ModelConfig = field(default_factory=ModelConfig)


def _coerce_paths(values: dict[str, Any]) -> dict[str, Any]:
    return {key: Path(value) if value is not None else value for key, value in values.items()}


def load_config(path: Path) -> ExperimentConfig:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return ExperimentConfig(
        seed=int(raw.get("seed", 13)),
        paths=PathConfig(**_coerce_paths(raw.get("paths", {}))),
        data=DataConfig(**raw.get("data", {})),
        chunking=ChunkingConfig(**raw.get("chunking", {})),
        retrieval=RetrievalConfig(**raw.get("retrieval", {})),
        models=ModelConfig(**raw.get("models", {})),
    )


def resolve_project_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path
