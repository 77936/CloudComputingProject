from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import joblib


@dataclass
class ArtifactMetadata:
    model_name: str
    version: str
    created_at: str
    threshold: float
    feature_columns: list[str]
    dropped_columns: list[str]
    metrics: dict[str, Any]
    dataset_stats: dict[str, Any]
    training_args: dict[str, Any]


def timestamped_run_dir(output_dir: Path) -> Path:
    run_dir = output_dir / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_bundle(
    run_dir: Path,
    preprocessor,
    model,
    metadata: ArtifactMetadata,
) -> Path:
    bundle_path = run_dir / "bundle.joblib"
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model": model,
            "metadata": metadata,
        },
        bundle_path,
    )
    with (run_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(metadata), handle, indent=2)
    return bundle_path


def load_bundle(bundle_path: Path) -> dict[str, Any]:
    return joblib.load(bundle_path)
