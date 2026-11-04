import re

import numpy as np
import pandas as pd

_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://\S+")


class PipelineError(RuntimeError):
    pass


def safe_error(error: Exception, limit: int = 300) -> str:
    """Error text for storage. Connection URLs can contain passwords, so they are removed."""
    text = f"{type(error).__name__}: {error}"
    return _URL.sub("<redacted-url>", text)[:limit]


class NightlyPipeline:
    """The nightly job as plain Python. Prefect (flows/nightly.py) only adds scheduling and retries."""

    def __init__(self, ingest, table_provider, champion_loader, monitor, retrainer, store, benchmark_factory,
                 ticker: str = "^NSEI", recent_days: int = 60, reference_days: int = 750,
                 max_stale_days: int = 5, clock=None):
        self.ingest = ingest
        self.table_provider = table_provider
        self.champion_loader = champion_loader
        self.monitor = monitor
        self.retrainer = retrainer
        self.store = store
        self.benchmark_factory = benchmark_factory
        self.ticker = ticker
        self.recent_days = recent_days
        self.reference_days = reference_days
        self.max_stale_days = max_stale_days
        self.clock = clock or (lambda: pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None).normalize())

    # ---- run bookkeeping ----
    def start(self) -> int:
        return self.store.start_run()

    def finish_ok(self, run_id: int, result: dict) -> None:
        self.store.finish_run(run_id, "SUCCESS", result)

    def finish_failed(self, run_id: int, error: Exception) -> None:
        self.store.finish_run(run_id, "FAILED", {"error": safe_error(error)})

    # ---- steps ----
    def ingest_step(self) -> None:
        if not self.ingest.run():
            raise PipelineError("ingest failed: data was not stored and verified")

    def analyze_step(self, force_retrain: bool = False) -> dict:
        table = self.table_provider()
        latest = table.index.max()
        stale_days = int((self.clock() - latest.normalize()).days)
        if stale_days > self.max_stale_days:
            raise PipelineError(f"data is stale: latest bar is {stale_days} days old")

        labeled = table.dropna(subset=["target"])
        champion = self.champion_loader.load()

        decision = self._check(labeled, champion)
        retrain = decision.retrain or force_retrain
        self.store.save_decision(latest, retrain, {**decision.to_dict(), "forced": force_retrain})

        outcome = None
        if retrain:
            outcome = self.retrainer.run(labeled, champion, data_end=str(latest.date()))
            if outcome["promoted"]:
                champion = self.champion_loader.load()

        forecast = self._forecast(table, champion, latest)
        return {
            "as_of": str(latest.date()),
            "stale_days": stale_days,
            "champion_version": champion.version,
            "retrain_triggered": bool(decision.retrain),
            "forced": force_retrain,
            "triggers": decision.triggers,
            "feature_drift_warning": bool(decision.feature.drifted),
            "retrain_outcome": outcome,
            "forecast": forecast,
        }

    def run(self, force_retrain: bool = False) -> dict:
        """Everything in sequence, with a run record. Used for local runs and tests."""
        run_id = self.start()
        try:
            self.ingest_step()
            result = self.analyze_step(force_retrain)
        except Exception as error:
            self.finish_failed(run_id, error)
            raise
        self.finish_ok(run_id, result)
        return result

    # ---- internals ----
    def _check(self, labeled: pd.DataFrame, champion):
        fit = labeled[labeled.index <= champion.fit_end]
        recent = labeled.iloc[-self.recent_days:]
        reference = fit.iloc[-self.reference_days:]
        benchmark = self.benchmark_factory().fit(fit)   # same training data as the champion
        return self.monitor.check(
            reference, recent,
            np.asarray(champion.model.predict(recent)).ravel(),
            np.asarray(benchmark.predict(recent)).ravel(),
            days_since_retrain=self._days_since_retrain(labeled),
        )

    def _days_since_retrain(self, labeled: pd.DataFrame) -> int:
        last = self.store.last_retrain_date()
        if last is None:
            return 10 ** 6   # never retrained, so the cooldown cannot block
        return int((labeled.index > last).sum())

    def _forecast(self, table: pd.DataFrame, champion, latest) -> dict:
        latest_row = table.iloc[[-1]]   # newest day: its features are known, its target is not
        value = float(np.asarray(champion.model.predict(latest_row)).ravel()[0])
        self.store.save_forecast(self.ticker, latest, champion.version, value)
        return {"as_of": str(latest.date()), "champion_version": champion.version, "forecast_vol": round(value, 3)}