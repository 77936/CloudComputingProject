from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class IsolationForestModel:
    contamination: float
    random_state: int
    n_estimators: int = 200
    model: IsolationForest | None = None

    def fit(self, frame: pd.DataFrame) -> None:
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=self.n_estimators,
            n_jobs=-1,
        )
        self.model.fit(frame)

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model has not been trained")
        return -self.model.score_samples(frame)
