from __future__ import annotations

import json
from pathlib import Path

from rtad.inference import BundleScorer
from rtad.schemas import PredictionRequest, utc_now


def score_batch_file(
    bundle_path: Path,
    input_jsonl: Path,
    output_jsonl: Path,
    batch_size: int,
    experiment_id: str,
) -> int:
    scorer = BundleScorer(bundle_path)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with input_jsonl.open("r", encoding="utf-8") as source, output_jsonl.open(
        "w",
        encoding="utf-8",
    ) as destination:
        batch: list[str] = []
        for line in source:
            if not line.strip():
                continue
            batch.append(line)
            if len(batch) >= batch_size:
                count += _flush_batch(scorer, batch, destination, experiment_id)
                batch = []
        if batch:
            count += _flush_batch(scorer, batch, destination, experiment_id)
    return count


def _flush_batch(scorer, batch: list[str], destination, experiment_id: str) -> int:
    for line in batch:
        payload = json.loads(line)
        payload["event_timestamp"] = utc_now()
        scored = scorer.score_request(
            PredictionRequest(**payload),
            pipeline_mode="batch",
            experiment_id=experiment_id,
        ).to_dict()
        destination.write(json.dumps(scored))
        destination.write("\n")
    return len(batch)
