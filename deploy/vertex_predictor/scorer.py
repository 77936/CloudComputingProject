from __future__ import annotations

import tempfile
from pathlib import Path

import joblib
import pandas as pd


class BundleScorer:
    def __init__(self, bundle_path: Path):
        bundle = joblib.load(bundle_path)
        self.preprocessor = bundle["preprocessor"]
        self.model = bundle["model"]
        self.metadata = bundle["metadata"]

    def score(self, payload: dict) -> dict:
        feature_columns = self.metadata.feature_columns
        frame = pd.DataFrame([payload["features"]], columns=feature_columns)
        transformed = self.preprocessor.transform(frame)
        score = float(self.model.score(transformed)[0])
        anomaly_flag = int(score >= float(self.metadata.threshold))
        return {
            "event_id": payload["event_id"],
            "event_timestamp": payload["event_timestamp"],
            "inference_timestamp": pd.Timestamp.utcnow().isoformat(),
            "anomaly_score": score,
            "anomaly_flag": anomaly_flag,
            "model_version": self.metadata.version,
            "label": payload.get("label"),
            "binary_label": payload.get("binary_label"),
            "pipeline_mode": "vertex",
        }
