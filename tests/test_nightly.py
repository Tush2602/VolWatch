import numpy as np
import pandas as pd
import pytest

from models.baselines import ScaledVIXForecaster
from monitoring.drift import FeatureDriftResult, PerformanceDriftResult, RetrainDecision
from pipeline.champion import ChampionBundle
from pipeline.nightly import NightlyPipeline, PipelineError, safe_error
from pipeline.store import InMemoryStore


def make_table(n=800, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    vol = rng.uniform(5, 20, n)
    table = pd.DataFrame({"vol": vol, "vix": vol * 1.5}, index=idx)
    table["target"] = table["vol"].shift(-1)   # the newest row has no target, like the real table
    return table


class FakeIngest:
    def __init__(self, ok=True, error=None):
        self.ok, self.error = ok, error

    def run(self):
        if self.error:
            raise self.error
        return self.ok


class FakeModel:
    def predict(self, df):
        return df["vol"].to_numpy()


class FakeLoader:
    def __init__(self, fit_end):
        self.fit_end, self.calls = fit_end, 0

    def load(self):
        self.calls += 1
        return ChampionBundle(version=self.calls, fit_end=self.fit_end, model=FakeModel())


class FakeMonitor:
    def __init__(self, retrain=False):
        self.retrain, self.days_seen = retrain, None

    def check(self, reference, recent, champion, benchmark, days_since_retrain):
        self.days_seen = days_since_retrain
        return RetrainDecision(
            retrain=self.retrain, triggers=["fake trigger"] if self.retrain else [], suppressed_by_cooldown=False,
            feature=FeatureDriftResult({}, {}, [], False),
            performance=PerformanceDriftResult(0, None, None, None, False, False),
        )


class FakeRetrainer:
    def __init__(self, promoted=True):
        self.promoted, self.calls = promoted, 0

    def run(self, labeled, champion, data_end):
        self.calls += 1
        return {"promoted": self.promoted}


def make_pipeline(table, ingest=None, monitor=None, retrainer=None, store=None, clock=None):
    parts = {
        "store": store or InMemoryStore(),
        "monitor": monitor or FakeMonitor(),
        "retrainer": retrainer or FakeRetrainer(),
        "loader": FakeLoader(fit_end=table.index[600]),
    }
    parts["pipeline"] = NightlyPipeline(
        ingest=ingest or FakeIngest(), table_provider=lambda: table, champion_loader=parts["loader"],
        monitor=parts["monitor"], retrainer=parts["retrainer"], store=parts["store"],
        benchmark_factory=ScaledVIXForecaster, ticker="TEST", recent_days=60, reference_days=300,
        max_stale_days=5, clock=clock or (lambda: table.index[-1] + pd.Timedelta(days=1)),
    )
    return parts


def test_quiet_night_writes_forecast_decision_and_run_record():
    table = make_table()
    p = make_pipeline(table)
    result = p["pipeline"].run()

    assert ("TEST", table.index[-1]) in p["store"].forecasts
    assert p["store"].decisions[table.index[-1]][0] is False
    assert p["store"].runs[-1]["status"] == "SUCCESS"
    assert p["retrainer"].calls == 0
    assert result["forecast"]["champion_version"] == 1


def test_triggered_retrain_runs_the_trial_and_forecasts_with_the_new_champion():
    table = make_table()
    p = make_pipeline(table, monitor=FakeMonitor(retrain=True), retrainer=FakeRetrainer(promoted=True))
    p["pipeline"].run()

    assert p["retrainer"].calls == 1
    assert p["store"].decisions[table.index[-1]][0] is True
    assert p["store"].forecasts[("TEST", table.index[-1])][0] == 2   # champion was reloaded after promotion


def test_force_retrain_runs_the_trial_even_without_a_trigger():
    p = make_pipeline(make_table(), monitor=FakeMonitor(retrain=False))
    p["pipeline"].run(force_retrain=True)
    assert p["retrainer"].calls == 1


def test_failed_ingest_records_failure_and_writes_no_forecast():
    p = make_pipeline(make_table(), ingest=FakeIngest(ok=False))
    with pytest.raises(PipelineError):
        p["pipeline"].run()
    assert p["store"].runs[-1]["status"] == "FAILED"
    assert p["store"].forecasts == {}


def test_stale_data_fails_instead_of_publishing_an_old_forecast():
    table = make_table()
    p = make_pipeline(table, clock=lambda: table.index[-1] + pd.Timedelta(days=30))
    with pytest.raises(PipelineError):
        p["pipeline"].run()
    assert p["store"].forecasts == {}


def test_stored_error_text_never_contains_a_connection_url():
    boom = RuntimeError("could not connect to postgresql://user:secret@host/db right now")
    p = make_pipeline(make_table(), ingest=FakeIngest(error=boom))
    with pytest.raises(RuntimeError):
        p["pipeline"].run()
    assert "secret" not in str(p["store"].runs[-1]["details"])
    assert "secret" not in safe_error(boom)


def test_cooldown_counts_trading_days_since_the_last_retrain_decision():
    table = make_table()
    store = InMemoryStore()
    store.save_decision(table.index[700], True, {})
    p = make_pipeline(table, store=store)
    p["pipeline"].run()

    labeled = table.dropna(subset=["target"])
    assert p["monitor"].days_seen == int((labeled.index > table.index[700]).sum())