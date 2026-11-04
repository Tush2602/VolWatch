from models.xgb_model import XGBForecaster
import pandas as pd 
import numpy as np 
import pytest
from models.splitter import Timesplitter

def make_table(n = 200, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    vol = rng.uniform(5,20,n)
    df = pd.DataFrame({"vol":vol, 'vix': vol*1.5 + rng.normal(0, 1, n)}, index=idx)
    df['target'] = vol *1.1
    return df

def test_predict_requires_fit_first():
    with pytest.raises(RuntimeError):
        XGBForecaster().predict(make_table())

def test_forecast_is_aligned_with_input():
    table = make_table()
    out= XGBForecaster(n_estimators = 20).fit(table).predict(table)
    assert len(out) == len(table)
    assert out.index.equals(table.index)

def test_predict_ignores_the_target_column():
    table = make_table()
    model = XGBForecaster(n_estimators= 20).fit(table)
    pd.testing.assert_series_equal(model.predict(table), model.predict(table.assign(target=0.0)))


def test_log_version_gives_positive_forecasts():
    table = make_table()
    out = XGBForecaster(log_target=True, n_estimators = 20).fit(table).predict(table)
    assert (out > 0).all()
