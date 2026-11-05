import json

import pandas as pd


class InMemoryStore:
    """A fake database with the same methods as Database, so the pipeline can be tested offline."""

    def __init__(self):
        self.forecasts = {}
        self.decisions = {}
        self.runs = []

    def save_forecast(self, ticker, as_of_date, model_version, forecast_vol, source="live", overwrite=True) -> None:
        key = (ticker, pd.Timestamp(as_of_date))
        if key in self.forecasts and not overwrite:
            return
        self.forecasts[key] = (int(model_version), float(forecast_vol), source)

    def save_decision(self, as_of_date, retrain, details) -> None:
        self.decisions[pd.Timestamp(as_of_date)] = (bool(retrain), json.dumps(details, default=str))

    def start_run(self) -> int:
        self.runs.append({"status": "RUNNING", "details": None})
        return len(self.runs)

    def finish_run(self, run_id, status, details) -> None:
        self.runs[run_id - 1].update(status=status, details=details)

    def latest_run(self, status=None):
        runs = [r for r in self.runs if status is None or r["status"] == status]
        return runs[-1] if runs else None

    def last_retrain_date(self):
        dates = [d for d, (retrain, _) in self.decisions.items() if retrain]
        return max(dates) if dates else None