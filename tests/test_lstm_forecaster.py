import numpy as np
import pandas as pd
import pytest

from models.lstm_model import LSTMForecaster


def make_table(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    vol = rng.uniform(5, 20, n)
    df = pd.DataFrame({
        "vol": vol,
        "abs_return": rng.uniform(0, 2, n),
        "vix": vol * 1.5 + rng.normal(0, 1, n),
        "vix_change": rng.normal(0, 0.5, n),
    }, index=idx)
    df["target"] = vol * 1.1
    return df


def small_model(**kwargs):
    return LSTMForecaster(window=5, hidden=8, epochs=3, **kwargs)


def test_predict_requires_fit_first():
    with pytest.raises(RuntimeError):
        small_model().predict(make_table())


def test_first_rows_without_a_full_window_are_nan():
    table = make_table()
    out = small_model().fit(table).predict(table)
    assert len(out) == len(table)
    assert out.iloc[:4].isna().all()
    assert out.iloc[4:].notna().all()


def test_predict_ignores_the_target_column():
    table = make_table()
    model = small_model().fit(table)
    pd.testing.assert_series_equal(model.predict(table), model.predict(table.assign(target=0.0)))


def test_forecast_does_not_use_future_rows():
    table = make_table()
    model = small_model().fit(table)
    base = model.predict(table)

    changed = table.copy()
    changed.iloc[-3:, changed.columns.get_loc("vol")] *= 5   # alter only the last 3 days
    after = model.predict(changed)

    pd.testing.assert_series_equal(base.iloc[:-3], after.iloc[:-3])


def test_same_seed_gives_same_forecast():
    table = make_table()
    a = small_model(seed=1).fit(table).predict(table)
    b = small_model(seed=1).fit(table).predict(table)
    pd.testing.assert_series_equal(a, b)