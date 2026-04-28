from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, UTC


@dataclass
class PredictionRequest:
    event_id: str
    event_timestamp: str
    features: dict[str, float]
    label: str | None = None
    binary_label: int | None = None


@dataclass
class PredictionResponse:
    event_id: str
    event_timestamp: str
    inference_timestamp: str
    anomaly_score: float
    anomaly_flag: int
    model_version: str
    label: str | None = None
    binary_label: int | None = None
    pipeline_mode: str | None = None
    experiment_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()
