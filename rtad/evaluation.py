from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    candidates = np.quantile(scores, np.linspace(0.7, 0.99, 30))
    best_threshold = float(candidates[0])
    best_metrics = {"f1": -1.0}
    for threshold in candidates:
        predictions = (scores >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        false_positive_rate = float(fp / (fp + tn)) if (fp + tn) else 0.0
        metrics = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "false_positive_rate": false_positive_rate,
            "threshold": float(threshold),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        if metrics["f1"] > best_metrics["f1"]:
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


def evaluate_predictions(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def summarize_result_file(path: Path) -> dict:
    frame = pd.read_json(path, lines=True)
    labels = frame["binary_label"].fillna(0).astype(int).to_numpy()
    predictions = frame["anomaly_flag"].astype(int).to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="binary",
        zero_division=0,
    )
    end_to_end_ms = (
        pd.to_datetime(frame["inference_timestamp"]) - pd.to_datetime(frame["event_timestamp"])
    ).dt.total_seconds() * 1000.0
    return {
        "rows": int(len(frame)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "avg_latency_ms": float(end_to_end_ms.mean()) if len(end_to_end_ms) else 0.0,
        "p95_latency_ms": float(end_to_end_ms.quantile(0.95)) if len(end_to_end_ms) else 0.0,
    }
