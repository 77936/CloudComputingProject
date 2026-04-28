from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from flask import Flask, jsonify, request

from scorer import BundleScorer


def resolve_artifact_path() -> Path:
    artifact_uri = os.environ.get("ARTIFACT_URI")
    artifact_path = os.environ.get("ARTIFACT_PATH")

    if artifact_path:
        return Path(artifact_path)

    if artifact_uri and artifact_uri.startswith("gs://"):
        try:
            from google.cloud import storage
        except ModuleNotFoundError as exc:
            raise RuntimeError("google-cloud-storage is required to download the artifact bundle") from exc
        bucket_name, blob_name = artifact_uri[5:].split("/", 1)
        destination = Path(tempfile.gettempdir()) / "bundle.joblib"
        client = storage.Client()
        client.bucket(bucket_name).blob(blob_name).download_to_filename(destination)
        return destination

    raise RuntimeError("Set ARTIFACT_PATH or ARTIFACT_URI before starting the predictor")


scorer = BundleScorer(resolve_artifact_path())
app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_version": scorer.metadata.version})


@app.post("/predict")
def predict():
    payload = request.get_json(force=True)
    instances = payload.get("instances", payload if isinstance(payload, list) else [])
    predictions = []
    for item in instances:
        if "features" not in item:
            item = {
                "event_id": item.get("event_id", "vertex-event"),
                "event_timestamp": item.get("event_timestamp"),
                "features": item,
                "label": item.get("label"),
                "binary_label": item.get("binary_label"),
            }
        predictions.append(scorer.score(item))
    return jsonify({"predictions": predictions})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
