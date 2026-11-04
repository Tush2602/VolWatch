import pandas as pd
import pytest

from features.volatility import VolatilityEstimator


def make_df(high, low):
    return pd.DataFrame({"High": [high], "Low": [low]})


def test_zero_range_gives_zero_volatility():
    vol = VolatilityEstimator().estimate(make_df(100, 100))
    assert vol.iloc[0] == pytest.approx(0.0)


def test_known_value():
    # High=102, Low=100, worked out by hand: about 0.0118926 daily
    est = VolatilityEstimator(annualize=False, as_percent=False)
    assert est.estimate(make_df(102, 100)).iloc[0] == pytest.approx(0.0118926, abs=1e-5)


def test_annualized_percent():
    # same day, annualized and as percent: about 18.88
    assert VolatilityEstimator().estimate(make_df(102, 100)).iloc[0] == pytest.approx(18.88, abs=0.01)


def test_rejects_low_above_high():
    with pytest.raises(ValueError):
        VolatilityEstimator().estimate(make_df(100, 102))