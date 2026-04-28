from __future__ import annotations

import json
import time
from pathlib import Path

from rtad.inference import BundleScorer
from rtad.schemas import PredictionRequest, utc_now


def simulate_events(
    bundle_path: Path,
    input_jsonl: Path,
    destination: str,
    output_jsonl: Path | None,
    send_rate: float,
    burst_multiplier: float,
    experiment_id: str,
) -> int:
    scorer = BundleScorer(bundle_path)
    output_handle = None
    sent = 0
    interval = 0.0 if send_rate <= 0 else 1.0 / send_rate
    adjusted_interval = interval / burst_multiplier if burst_multiplier > 0 else interval

    if destination in {"local", "file"}:
        if output_jsonl is None:
            raise ValueError("output_jsonl is required for local or file destinations")
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_jsonl.open("w", encoding="utf-8")

    try:
        with input_jsonl.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                if destination == "local":
                    payload = json.loads(line)
                    payload["event_timestamp"] = utc_now()
                    scored = scorer.score_request(
                        PredictionRequest(**payload),
                        pipeline_mode="stream",
                        experiment_id=experiment_id,
                    ).to_dict()
                    output_handle.write(json.dumps(scored))
                    output_handle.write("\n")
                elif destination == "file":
                    output_handle.write(line)
                elif destination == "pubsub":
                    raise NotImplementedError(
                        "Pub/Sub publishing requires google-cloud-pubsub and live credentials. "
                        "Use the deploy scripts once GCP access is configured."
                    )
                else:
                    raise ValueError(f"Unsupported destination: {destination}")
                sent += 1
                if adjusted_interval > 0:
                    time.sleep(adjusted_interval)
    finally:
        if output_handle is not None:
            output_handle.close()

    return sent
