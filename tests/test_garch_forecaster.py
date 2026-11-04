import numpy as np
import pandas as pd
import pytest

from models.garch_model import GarchForecaster


def manual_model():
    m = GarchForecaster()
    m.omega, m.alpha, m.beta = 0.1, 0.1, 0.8
    m.init_var = 1.0
    m.scale = 1.0
    return m


def test_predict_requires_fit_first():
    table = pd.DataFrame({"abs_return": [1.0]}, index=pd.bdate_range("2024-01-01", periods=1))
    with pytest.raises(RuntimeError):
        GarchForecaster().predict(table)


def test_recursion_matches_hand_calculation():
    # day 1: h = 0.1 + 0.1*2^2 + 0.8*1 = 1.3   -> sqrt(1.3)*sqrt(252) = 18.10
    # day 2: h = 0.1 + 0.1*0^2 + 0.8*1.3 = 1.14 -> sqrt(1.14)*sqrt(252) = 16.95
    table = pd.DataFrame({"abs_return": [2.0, 0.0]}, index=pd.bdate_range("2024-01-01", periods=2))
    out = manual_model().predict(table)
    assert out.iloc[0] == pytest.approx(18.10, abs=0.01)
    assert out.iloc[1] == pytest.approx(16.95, abs=0.01)


def test_forecast_does_not_use_future_rows():
    idx = pd.bdate_range("2024-01-01", periods=30)
    rng = np.random.default_rng(0)
    table = pd.DataFrame({"abs_return": rng.uniform(0, 2, 30)}, index=idx)
    base = manual_model().predict(table)

    changed = table.copy()
    changed.iloc[-3:, 0] = 50.0
    after = manual_model().predict(changed)

    pd.testing.assert_series_equal(base.iloc[:-3], after.iloc[:-3])


def test_fit_gives_finite_positive_forecasts():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2020-01-01", periods=400)
    table = pd.DataFrame({"abs_return": np.abs(rng.normal(0, 1, 400)),
                          "target": rng.uniform(5, 15, 400)}, index=idx)
    out = GarchForecaster().fit(table).predict(table)
    assert np.isfinite(out).all()
    assert (out > 0).all()