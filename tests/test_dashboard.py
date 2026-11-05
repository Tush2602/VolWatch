import pytest
import requests

from dashboard.charts import ChartBuilder
from dashboard.client import ApiClient, ApiUnavailable


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code, self.payload = status_code, payload

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(outcomes, attempts=3):
    sleeps = []
    session = FakeSession(outcomes)
    client = ApiClient("http://api.test", session=session, attempts=attempts, wait_seconds=1, sleep=sleeps.append)
    return client, session, sleeps


def test_returns_the_payload_on_success():
    client, _, sleeps = make_client([FakeResponse(200, {"forecast_vol": 10.5})])
    assert client.predict() == {"forecast_vol": 10.5}
    assert sleeps == []


def test_retries_through_a_cold_start_then_succeeds():
    client, session, sleeps = make_client(
        [requests.Timeout(), requests.ConnectionError(), FakeResponse(200, {"ok": True})])
    assert client.status() == {"ok": True}
    assert session.calls == 3
    assert len(sleeps) == 2


def test_a_503_is_retried():
    client, session, _ = make_client([FakeResponse(503), FakeResponse(200, {"ok": True})])
    assert client.status() == {"ok": True}
    assert session.calls == 2


def test_a_404_means_nothing_there_and_is_not_retried():
    client, session, _ = make_client([FakeResponse(404)])
    assert client.predict() is None
    assert session.calls == 1


def test_gives_up_with_a_clear_error_after_all_attempts():
    client, session, _ = make_client([requests.Timeout()] * 3, attempts=3)
    with pytest.raises(ApiUnavailable) as error:
        client.predict()
    assert session.calls == 3
    assert "api.test" not in str(error.value)


HISTORY = [
    {"as_of_date": "2026-10-28", "forecast_vol": 10.0, "realized_vol": 11.0, "model_version": 1, "source": "backfill"},
    {"as_of_date": "2026-10-29", "forecast_vol": 11.0, "realized_vol": 12.0, "model_version": 1, "source": "live"},
    {"as_of_date": "2026-10-30", "forecast_vol": 12.0, "realized_vol": None, "model_version": 1, "source": "live"},
]


def test_chart_separates_live_and_backfilled_forecasts():
    names = [trace.name for trace in ChartBuilder.forecast_vs_realized(HISTORY).data]
    assert names == [ChartBuilder.REALIZED, ChartBuilder.BACKFILL, ChartBuilder.LIVE]


def test_chart_omits_a_series_that_has_no_points():
    live_only = [row for row in HISTORY if row["source"] == "live"]
    names = [trace.name for trace in ChartBuilder.forecast_vs_realized(live_only).data]
    assert ChartBuilder.BACKFILL not in names


def test_model_comparison_is_sorted_best_first():
    models = [{"model": "b", "kind": "naive", "qlike": 0.8}, {"model": "a", "kind": "deployed", "qlike": 0.4}]
    figure = ChartBuilder.model_comparison(models)
    assert list(figure.data[0].y) == ["a", "b"]