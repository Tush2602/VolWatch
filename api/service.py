import pandas as pd

from features.volatility import VolatilityEstimator


class ForecastService:
    """Reads what the nightly job stored and shapes it for the API. No web code here."""

    NOTE = ("Forecast of the next trading session's range-based volatility (annualized %). "
            "A technical forecasting demo, not investment advice.")

    def __init__(self, db, stale_after_days: int = 5, now=None):
        self.db = db
        self.stale_after_days = stale_after_days
        self.now = now or (lambda: pd.Timestamp.now(tz="UTC"))
        self.estimator = VolatilityEstimator()

    def _today(self) -> pd.Timestamp:
        return self.now().tz_convert("Asia/Kolkata").tz_localize(None).normalize()

    def latest(self, ticker: str) -> dict | None:
        df = self.db.read_forecasts(ticker, limit=1, source="live")
        if df.empty:
            return None
        row = df.iloc[0]
        as_of = pd.Timestamp(row["as_of_date"]).normalize()
        age = int((self._today() - as_of).days)
        return {
            "ticker": ticker,
            "as_of_date": as_of.date(),
            "forecast_vol": round(float(row["forecast_vol"]), 3),
            "model_version": int(row["model_version"]),
            "age_days": age,
            "stale": age > self.stale_after_days,
            "note": self.NOTE,
        }

    def history(self, ticker: str, limit: int = 30) -> list:
        forecasts = self.db.read_forecasts(ticker, limit=limit, source=str(row['source']))
        if forecasts.empty:
            return []
        forecasts = forecasts.copy()
        forecasts["as_of_date"] = pd.to_datetime(forecasts["as_of_date"])
        forecasts = forecasts.sort_values("as_of_date")

        since = forecasts["as_of_date"].min() - pd.Timedelta(days=10)
        prices = self.db.read_prices(ticker, since=since)
        realized = self.estimator.estimate(prices) if len(prices) else pd.Series(dtype=float)

        points = []
        for _, row in forecasts.iterrows():
            day = row["as_of_date"]
            actual = None
            if day in realized.index:
                position = realized.index.get_loc(day)
                if position + 1 < len(realized):   # the next session's bar must exist
                    actual = round(float(realized.iloc[position + 1]), 3)
            points.append({
                "as_of_date": day.date(),
                "forecast_vol": round(float(row["forecast_vol"]), 3),
                "realized_vol": actual,
                "model_version": int(row["model_version"]),
            })
        return points

    def status(self) -> dict:
        last = self.db.latest_run()
        success = self.db.latest_run(status="SUCCESS")
        details = (success or {}).get("details") or {}
        retrain = self.db.last_retrain_date()

        hours = None
        if success and success.get("finished_at") is not None:
            hours = round((self.now() - pd.Timestamp(success["finished_at"])).total_seconds() / 3600, 1)
        healthy = bool(last and last["status"] == "SUCCESS" and hours is not None
                       and hours <= self.stale_after_days * 24)

        finished = last.get("finished_at") if last else None
        return {
            "healthy": healthy,
            "pipeline": {
                "last_run_status": last["status"] if last else None,
                "last_run_finished_at": pd.Timestamp(finished).isoformat() if finished is not None else None,
                "hours_since_last_success": hours,
            },
            "model": {
                "champion_version": details.get("champion_version"),
                "last_retrain_decision": str(retrain.date()) if retrain is not None else None,
            },
            "drift": {
                "as_of": details.get("as_of"),
                "feature_drift_warning": details.get("feature_drift_warning"),
                "retrain_triggered": details.get("retrain_triggered"),
                "forced": details.get("forced"),
            },
        }