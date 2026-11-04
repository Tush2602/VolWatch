from dataclasses import dataclass

import numpy as np
import pandas as pd

from models.metrics import Metrics


@dataclass
class TrialResult:
    challenger: object
    challenger_qlike: float
    champion_qlike: float
    fit_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    wins: bool


class ChallengerTrial:
    """Trains a challenger on older data, then scores it and the champion on the same recent window."""

    def __init__(self, model_factory, holdout_days: int = 60, min_improvement: float = 0.02,
                 min_train_rows: int = 500):
        self.model_factory = model_factory
        self.holdout_days = holdout_days
        self.min_improvement = min_improvement
        self.min_train_rows = min_train_rows

    def run(self, labeled: pd.DataFrame, champion_model) -> TrialResult:
        if len(labeled) < self.holdout_days + self.min_train_rows:
            raise ValueError("not enough labeled rows to run a challenger trial")

        # one extra row is dropped, like TimeSplitter's purge: the last training label is the holdout's first day
        train = labeled.iloc[: -(self.holdout_days + 1)]
        holdout = labeled.iloc[-self.holdout_days:]

        challenger = self.model_factory().fit(train)
        actual = holdout["target"].to_numpy()
        challenger_q = Metrics.qlike(actual, np.asarray(challenger.predict(holdout)).ravel())
        champion_q = Metrics.qlike(actual, np.asarray(champion_model.predict(holdout)).ravel())

        return TrialResult(
            challenger=challenger, challenger_qlike=challenger_q, champion_qlike=champion_q,
            fit_end=train.index.max(), holdout_start=holdout.index.min(), holdout_end=holdout.index.max(),
            wins=bool(challenger_q < champion_q * (1 - self.min_improvement)),
        )