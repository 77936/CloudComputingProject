from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rtad.artifacts import load_bundle
from rtad.schemas import PredictionRequest, PredictionResponse, utc_now


class BundleScorer:
    def __init__(self, bundle_path: Path):
        bundle = load_bundle(bundle_path)
        self.preprocessor = bundle["preprocessor"]
        self.model = bundle["model"]
        self.metadata = bundle["metadata"]

    @property
    def feature_columns(self) -> list[str]:
        return self.metadata.feature_columns

    def score_request(
        self,
        request: PredictionRequest,
        pipeline_mode: str | None = None,
        experiment_id: str | None = None,
    ) -> PredictionResponse:
        frame = pd.DataFrame([request.features], columns=self.feature_columns)
        transformed = self.preprocessor.transform(frame)
        score = float(self.model.score(transformed)[0])
        threshold = float(self.metadata.threshold)
        anomaly_flag = int(score >= threshold)
        return PredictionResponse(
            event_id=request.event_id,
            event_timestamp=request.event_timestamp,
            inference_timestamp=utc_now(),
            anomaly_score=score,
            anomaly_flag=anomaly_flag,
            model_version=self.metadata.version,
            label=request.label,
            binary_label=request.binary_label,
            pipeline_mode=pipeline_mode,
            experiment_id=experiment_id,
        )

    def score_json_line(
        self,
        raw_line: str,
        pipeline_mode: str | None = None,
        experiment_id: str | None = None,
    ) -> dict:
        payload = json.loads(raw_line)
        request = PredictionRequest(**payload)
        return self.score_request(
            request,
            pipeline_mode=pipeline_mode,
            experiment_id=experiment_id,
        ).to_dict()
