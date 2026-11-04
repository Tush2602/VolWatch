import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.service import ForecastService

NOW = pd.Timestamp("2026-10-31 12:00", tz="UTC")   # 17:30 IST on 31 Oct
EMPTY = pd.DataFrame(columns=["as_of_date", "model_version", "forecast_vol"])


class FakeDB:
    def __init__(self, forecasts=None, prices=None, last_run=None, success_run=None, retrain=None):
        self.forecasts, self.prices = forecasts, prices
        self.last_run, self.success_run, self.retrain = last_run, success_run, retrain

    def read_forecasts(self, ticker, limit=60):
        if ticker != "^NSEI" or self.forecasts is None:
            return EMPTY
        return self.forecasts.sort_values("as_of_date", ascending=False).head(limit).reset_index(drop=True)

    def read_prices(self, ticker, since=None):
        return self.prices

    def latest_run(self, status=None):
        return self.success_run if status == "SUCCESS" else self.last_run

    def last_retrain_date(self):
        return self.retrain


class BrokenDB:
    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise RuntimeError("could not connect to postgresql://user:secret@host/db")
        return fail


def make_client(db):
    service = ForecastService(db, now=lambda: NOW)
    return TestClient(create_app(service), raise_server_exceptions=False)


def make_forecasts(dates, values):
    return pd.DataFrame({
        "as_of_date": [d.date() for d in pd.to_datetime(dates)],
        "model_version": 1,
        "forecast_vol": values,
    })


def make_run(status, finished, details=None):
    when = pd.Timestamp(finished, tz="UTC").to_pydatetime()
    return {"run_id": 1, "started_at": when, "finished_at": when, "status": status, "details": details}


GOOD_DETAILS = {"champion_version": 1, "as_of": "2026-10-30", "feature_drift_warning": False,
                "retrain_triggered": False, "forced": False}


# ---------- /predict ----------

def test_predict_returns_the_latest_forecast_with_its_age():
    client = make_client(FakeDB(forecasts=make_forecasts(["2026-10-29", "2026-10-30"], [9.0, 10.5])))
    body = client.get("/predict").json()
    assert body["as_of_date"] == "2026-10-30"
    assert body["forecast_vol"] == 10.5
    assert body["age_days"] == 1
    assert body["stale"] is False


def test_predict_flags_an_old_forecast_as_stale():
    client = make_client(FakeDB(forecasts=make_forecasts(["2026-10-20"], [10.0])))
    body = client.get("/predict").json()
    assert body["age_days"] == 11
    assert body["stale"] is True


def test_predict_unknown_ticker_is_404():
    client = make_client(FakeDB(forecasts=make_forecasts(["2026-10-30"], [10.0])))
    assert client.get("/predict", params={"ticker": "^OTHER"}).status_code == 404


# ---------- /history ----------

def test_history_joins_the_next_sessions_realized_volatility():
    idx = pd.bdate_range("2026-10-26", "2026-10-30")   # Monday to Friday
    prices = pd.DataFrame({"High": 102.0, "Low": 100.0}, index=idx)   # every day: 18.88 by hand
    client = make_client(FakeDB(
        forecasts=make_forecasts(["2026-10-28", "2026-10-29", "2026-10-30"], [10.0, 11.0, 12.0]),
        prices=prices,
    ))
    points = client.get("/history").json()
    assert [p["as_of_date"] for p in points] == ["2026-10-28", "2026-10-29", "2026-10-30"]
    assert points[0]["realized_vol"] == pytest.approx(18.88, abs=0.01)
    assert points[1]["realized_vol"] == pytest.approx(18.88, abs=0.01)
    assert points[2]["realized_vol"] is None   # the next session has not happened yet


def test_history_rejects_a_limit_below_one():
    assert make_client(FakeDB()).get("/history", params={"limit": 0}).status_code == 422


# ---------- /status ----------

def test_status_is_healthy_after_a_recent_success():
    ok = make_run("SUCCESS", "2026-10-31 10:30", GOOD_DETAILS)
    body = make_client(FakeDB(last_run=ok, success_run=ok, retrain=pd.Timestamp("2026-10-30"))).get("/status").json()
    assert body["healthy"] is True
    assert body["pipeline"]["hours_since_last_success"] == 1.5
    assert body["model"]["champion_version"] == 1
    assert body["model"]["last_retrain_decision"] == "2026-10-30"


def test_status_is_unhealthy_when_the_latest_run_failed():
    good = make_run("SUCCESS", "2026-10-30 10:30", GOOD_DETAILS)
    failed = make_run("FAILED", "2026-10-31 10:30", {"error": "something"})
    body = make_client(FakeDB(last_run=failed, success_run=good)).get("/status").json()
    assert body["healthy"] is False
    assert body["pipeline"]["last_run_status"] == "FAILED"
    assert body["model"]["champion_version"] == 1   # still reported from the last success
    assert "error" not in str(body)


def test_status_is_unhealthy_when_the_last_success_is_old():
    old = make_run("SUCCESS", "2026-10-20 10:30", GOOD_DETAILS)
    body = make_client(FakeDB(last_run=old, success_run=old)).get("/status").json()
    assert body["healthy"] is False


def test_status_before_any_run_has_no_data_and_is_unhealthy():
    body = make_client(FakeDB()).get("/status").json()
    assert body["healthy"] is False
    assert body["pipeline"]["last_run_status"] is None


# ---------- failure handling ----------

def test_database_failure_returns_503_without_leaking_the_connection_string():
    response = make_client(BrokenDB()).get("/predict")
    assert response.status_code == 503
    assert "secret" not in response.text
    assert response.headers["retry-after"] == "30"


def test_health_does_not_touch_the_database():
    assert make_client(BrokenDB()).get("/health").json() == {"status": "ok"}



'''
postgresql://neondb_owner:npg_vLAohQ0TmcW3@ep-delicate-tree-b56ex6jk-pooler.c-7.us-east-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require
'''