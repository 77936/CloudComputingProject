from __future__ import annotations

import argparse
from datetime import datetime, UTC
from pathlib import Path

from rtad.artifacts import ArtifactMetadata, save_bundle, timestamped_run_dir
from rtad.data import clean_dataframe, export_events_jsonl, load_dataset, split_dataset
from rtad.evaluation import evaluate_predictions, tune_threshold, write_json
from rtad.models import IsolationForestModel
from rtad.preprocessing import fit_preprocessor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the anomaly detection model locally.")
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-fraction", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--max-benign-train", type=int, default=None)
    parser.add_argument("--export-sample-size", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_dataset(
        args.dataset_path,
        sample_fraction=args.sample_fraction,
        random_state=args.random_state,
    )
    cleaned, dropped_columns, cleaning_stats = clean_dataframe(dataset)
    prepared = split_dataset(
        cleaned,
        random_state=args.random_state,
        max_benign_train=args.max_benign_train,
    )
    prepared.dropped_columns = dropped_columns

    preprocessor = fit_preprocessor(prepared.train_benign, prepared.feature_columns)
    train_transformed = preprocessor.transform(prepared.train_benign)
    validation_transformed = preprocessor.transform(prepared.validation)
    test_transformed = preprocessor.transform(prepared.test)

    model = IsolationForestModel(
        contamination=args.contamination,
        random_state=args.random_state,
    )
    model.fit(train_transformed)

    validation_scores = model.score(validation_transformed)
    validation_labels = prepared.validation["binary_label"].to_numpy()
    threshold, validation_metrics = tune_threshold(validation_scores, validation_labels)

    test_scores = model.score(test_transformed)
    test_labels = prepared.test["binary_label"].to_numpy()
    test_metrics = evaluate_predictions(test_scores, test_labels, threshold)

    run_dir = timestamped_run_dir(args.output_dir)
    version = run_dir.name
    metadata = ArtifactMetadata(
        model_name="IsolationForest",
        version=version,
        created_at=datetime.now(UTC).isoformat(),
        threshold=threshold,
        feature_columns=prepared.feature_columns,
        dropped_columns=dropped_columns,
        metrics={
            "validation": validation_metrics,
            "test": test_metrics,
        },
        dataset_stats={
            "cleaning": cleaning_stats,
            "splits": prepared.stats,
        },
        training_args={
            "sample_fraction": args.sample_fraction,
            "random_state": args.random_state,
            "contamination": args.contamination,
            "max_benign_train": args.max_benign_train,
        },
    )
    bundle_path = save_bundle(run_dir, preprocessor, model, metadata)
    export_events_jsonl(
        prepared.test,
        prepared.feature_columns,
        run_dir / "test_events.jsonl",
        limit=args.export_sample_size,
    )
    export_events_jsonl(
        prepared.validation,
        prepared.feature_columns,
        run_dir / "validation_events.jsonl",
        limit=args.export_sample_size,
    )
    write_json(run_dir / "evaluation.json", metadata.metrics)

    print(f"Saved bundle: {bundle_path}")
    print(f"Run directory: {run_dir}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Validation F1: {validation_metrics['f1']:.4f}")
    print(f"Test F1: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
