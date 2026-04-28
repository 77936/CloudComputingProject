from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class FittedPreprocessor:
    feature_columns: list[str]
    scaler: StandardScaler

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        ordered = frame[self.feature_columns].copy()
        transformed = self.scaler.transform(ordered)
        return pd.DataFrame(transformed, columns=self.feature_columns, index=frame.index)


def fit_preprocessor(frame: pd.DataFrame, feature_columns: list[str]) -> FittedPreprocessor:
    scaler = StandardScaler()
    scaler.fit(frame[feature_columns])
    return FittedPreprocessor(feature_columns=feature_columns, scaler=scaler)
