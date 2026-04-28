from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rtad.artifacts import ArtifactMetadata, save_bundle
from rtad.batch import score_batch_file
from rtad.evaluation import evaluate_predictions, tune_threshold
from rtad.inference import BundleScorer
from rtad.models import IsolationForestModel
from rtad.preprocessing import fit_preprocessor
from rtad.simulation import simulate_events


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        rows = []
        for i in range(30):
            rows.append({"f1": 0.1 + i * 0.001, "f2": 0.2 + i * 0.001, "binary_label": 0, "Label": "BENIGN"})
        for i in range(10):
            rows.append({"f1": 10 + i, "f2": 11 + i, "binary_label": 1, "Label": "ATTACK"})
        self.frame = pd.DataFrame(rows)
        self.feature_columns = ["f1", "f2"]

    def _train_bundle(self) -> Path:
        benign = self.frame[self.frame["binary_label"] == 0]
        validation = self.frame.copy()
        preprocessor = fit_preprocessor(benign, self.feature_columns)
        model = IsolationForestModel(contamination=0.2, random_state=42, n_estimators=100)
        model.fit(preprocessor.transform(benign))
        scores = model.score(preprocessor.transform(validation))
        threshold, metrics = tune_threshold(scores, validation["binary_label"].to_numpy())
        metadata = ArtifactMetadata(
            model_name="IsolationForest",
            version="test-version",
            created_at="2026-01-01T00:00:00+00:00",
            threshold=threshold,
            feature_columns=self.feature_columns,
            dropped_columns=[],
            metrics={"validation": metrics, "test": evaluate_predictions(scores, validation["binary_label"].to_numpy(), threshold)},
            dataset_stats={},
            training_args={},
        )
        return save_bundle(self.temp_dir, preprocessor, model, metadata)

    def _write_events(self) -> Path:
        events_path = self.temp_dir / "events.jsonl"
        with events_path.open("w", encoding="utf-8") as handle:
            for index, row in self.frame.iterrows():
                handle.write(
                    json.dumps(
                        {
                            "event_id": f"event-{index}",
                            "event_timestamp": "2026-01-01T00:00:00+00:00",
                            "features": {column: float(row[column]) for column in self.feature_columns},
                            "label": row["Label"],
                            "binary_label": int(row["binary_label"]),
                        }
                    )
                )
                handle.write("\n")
        return events_path

    def test_bundle_persistence_and_parity(self) -> None:
        bundle_path = self._train_bundle()
        scorer = BundleScorer(bundle_path)
        self.assertEqual(scorer.feature_columns, self.feature_columns)

        events_path = self._write_events()
        stream_output = self.temp_dir / "stream.jsonl"
        batch_output = self.temp_dir / "batch.jsonl"

        sent = simulate_events(
            bundle_path=bundle_path,
            input_jsonl=events_path,
            destination="local",
            output_jsonl=stream_output,
            send_rate=0.0,
            burst_multiplier=1.0,
            experiment_id="stream-test",
        )
        scored = score_batch_file(
            bundle_path=bundle_path,
            input_jsonl=events_path,
            output_jsonl=batch_output,
            batch_size=8,
            experiment_id="batch-test",
        )

        self.assertEqual(sent, len(self.frame))
        self.assertEqual(scored, len(self.frame))

        stream_lines = stream_output.read_text(encoding="utf-8").strip().splitlines()
        batch_lines = batch_output.read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(stream_lines), len(batch_lines))


if __name__ == "__main__":
    unittest.main()
