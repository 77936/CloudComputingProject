from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLUMN = "Label"
BENIGN_LABEL = "BENIGN"


@dataclass
class PreparedDataset:
    train_benign: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    feature_columns: list[str]
    dropped_columns: list[str]
    stats: dict


def csv_files(dataset_path: Path) -> list[Path]:
    files = sorted(dataset_path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found under {dataset_path}")
    return files


def load_dataset(
    dataset_path: Path,
    sample_fraction: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_path in csv_files(dataset_path):
        frame = pd.read_csv(file_path)
        frame.columns = frame.columns.str.strip()
        frame[LABEL_COLUMN] = frame[LABEL_COLUMN].astype(str).str.strip()
        if sample_fraction is not None and 0 < sample_fraction < 1:
            frame = frame.sample(frac=sample_fraction, random_state=random_state)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict]:
    frame = df.copy()
    feature_columns = [column for column in frame.columns if column != LABEL_COLUMN]

    for column in feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame.replace([np.inf, -np.inf], np.nan, inplace=True)
    before_rows = len(frame)
    frame.dropna(inplace=True)

    constant_columns = [
        column
        for column in feature_columns
        if frame[column].nunique(dropna=False) <= 1
    ]
    if constant_columns:
        frame.drop(columns=constant_columns, inplace=True)

    frame["binary_label"] = (frame[LABEL_COLUMN] != BENIGN_LABEL).astype(int)
    stats = {
        "rows_before_dropna": before_rows,
        "rows_after_dropna": len(frame),
        "constant_columns_dropped": constant_columns,
        "class_distribution": frame[LABEL_COLUMN].value_counts().to_dict(),
        "binary_distribution": frame["binary_label"].value_counts().to_dict(),
    }
    return frame, constant_columns, stats


def split_dataset(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
    max_benign_train: int | None = None,
) -> PreparedDataset:
    feature_columns = [
        column for column in df.columns if column not in {LABEL_COLUMN, "binary_label"}
    ]
    train_df, temp_df = train_test_split(
        df,
        test_size=validation_size + test_size,
        random_state=random_state,
        stratify=df["binary_label"],
    )
    validation_ratio = validation_size / (validation_size + test_size)
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=1 - validation_ratio,
        random_state=random_state,
        stratify=temp_df["binary_label"],
    )

    train_benign = train_df[train_df["binary_label"] == 0].copy()
    if max_benign_train and len(train_benign) > max_benign_train:
        train_benign = train_benign.sample(
            n=max_benign_train,
            random_state=random_state,
        )

    stats = {
        "train_benign_rows": len(train_benign),
        "validation_rows": len(validation_df),
        "test_rows": len(test_df),
    }
    return PreparedDataset(
        train_benign=train_benign,
        validation=validation_df.copy(),
        test=test_df.copy(),
        feature_columns=feature_columns,
        dropped_columns=[],
        stats=stats,
    )


def export_events_jsonl(
    df: pd.DataFrame,
    feature_columns: list[str],
    output_path: Path,
    limit: int | None = None,
) -> int:
    frame = df.copy()
    if limit is not None:
        frame = frame.head(limit)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, row in frame.iterrows():
            payload = {
                "event_id": f"event-{index}",
                "event_timestamp": pd.Timestamp.utcnow().isoformat(),
                "features": {column: float(row[column]) for column in feature_columns},
                "label": row[LABEL_COLUMN],
                "binary_label": int(row["binary_label"]),
            }
            handle.write(json.dumps(payload))
            handle.write("\n")
            exported += 1
    return exported
