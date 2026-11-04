import numpy as np
import pandas as pd
import pytest

from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator


def make_market(n=80, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    nifty = pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close}, index=idx)
    vix = pd.DataFrame({"Close": 15 + rng.normal(0, 1, n)}, index=idx)
    return nifty, vix


def test_target_is_next_days_volatility():
    nifty, vix = make_market()
    feats = FeatureBuilder(VolatilityEstimator()).build(nifty, vix)
    vol = VolatilityEstimator().estimate(nifty)

    t = feats.index[10]
    next_day = vol.index[vol.index.get_loc(t) + 1]
    assert feats.loc[t, "target"] == pytest.approx(vol.loc[next_day])


def test_only_the_last_row_has_no_target():
    nifty, vix = make_market()
    feats = FeatureBuilder(VolatilityEstimator()).build(nifty, vix)
    assert feats["target"].isna().sum() == 1
    assert pd.isna(feats["target"].iloc[-1])


def test_features_do_not_use_future_rows():
    nifty, vix = make_market()
    builder = FeatureBuilder(VolatilityEstimator())
    base = builder.build(nifty, vix)

    # wildly change the LAST day's prices; every earlier row's features must be unaffected
    changed = nifty.copy()
    changed.iloc[-1, changed.columns.get_loc("High")] *= 1.5
    after = builder.build(changed, vix)

    cols = builder.feature_names
    pd.testing.assert_frame_equal(base.iloc[:-1][cols], after.iloc[:-1][cols])