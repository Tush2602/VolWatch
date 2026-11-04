import numpy as np
import pandas as pd

from models.baselines import PersistenceForecaster
from monitoring.drift import FeatureDriftResult, PerformanceDriftResult, RetrainDecision
from monitoring.replay import PolicyReplay


def make_table(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    vol = rng.uniform(5, 20, n)
    table = pd.DataFrame({"vol": vol, "vix": vol * 1.5}, index=idx)
    table["target"] = table["vol"].shift(-1)
    return table


class SpyMonitor:
    """Records the windows it was given, and optionally asks for one retrain on a chosen call."""

    def __init__(self, retrain_on_call=None):
        self.calls = []
        self.retrain_on_call = retrain_on_call

    def check(self, reference, recent, champion, benchmark, days_since_retrain):
        self.calls.append((reference.index.max(), recent.index.max()))
        return RetrainDecision(
            retrain=len(self.calls) == self.retrain_on_call, triggers=[], suppressed_by_cooldown=False,
            feature=FeatureDriftResult({}, {}, [], False),
            performance=PerformanceDriftResult(0, None, None, None, False, False),
        )


def make_replay(monitor, table):
    return PolicyReplay(
        table, monitor, PersistenceForecaster, PersistenceForecaster,
        fit_end=table.index[150], start=table.index[151], step=5, recent_days=30, reference_days=100,
    )


def test_replay_never_uses_rows_from_the_check_date_or_later():
    table = make_table()
    monitor = SpyMonitor()
    records = make_replay(monitor, table).run()

    assert len(records) == len(monitor.calls) > 0
    for (reference_end, recent_end), date in zip(monitor.calls, records["date"]):
        assert recent_end < date
        assert reference_end <= table.index[150]   # no retrain, so the reference stays in the fit window


def test_reference_window_moves_after_a_retrain():
    table = make_table()
    monitor = SpyMonitor(retrain_on_call=2)
    make_replay(monitor, table).run()

    assert monitor.calls[2][0] > monitor.calls[1][0]   # new rows had arrived by the second check


def test_forward_forecasts_start_at_the_first_check_and_are_aligned():
    table = make_table()
    replay = make_replay(SpyMonitor(), table)
    replay.run()

    assert replay.forecasts.index.min() == table.index[151]
    assert replay.forecasts.index.is_unique
    assert replay.forecasts.notna().all().all()