from datetime import datetime

import pandas as pd

from ingest.validate import BarFilter


def make_df(dates, high=102.0, low=100.0):
    n = len(dates)
    return pd.DataFrame(
        {"Open": [101.0] * n, "High": [high] * n, "Low": [low] * n, "Close": [101.0] * n},
        index=pd.to_datetime(dates),
    )


def test_drops_today_before_market_settles():
    df = make_df(["2026-10-14", "2026-10-15"])
    out, dropped = BarFilter().clean(df, now=datetime(2026, 10, 15, 11, 0))
    assert list(out.index.date.astype(str)) == ["2026-10-14"]
    assert dropped["unfinished today bar"] == 1


def test_keeps_today_after_market_settles():
    df = make_df(["2026-10-14", "2026-10-15"])
    out, dropped = BarFilter().clean(df, now=datetime(2026, 10, 15, 16, 30))
    assert len(out) == 2
    assert dropped["unfinished today bar"] == 0


def test_drops_low_above_high():
    out, dropped = BarFilter().clean(make_df(["2026-10-01"], high=100.0, low=102.0),
                                     now=datetime(2026, 10, 15, 18, 0))
    assert out.empty
    assert dropped["low above high"] == 1


def test_drops_duplicate_dates():
    df = make_df(["2026-10-01", "2026-10-01"])
    out, _ = BarFilter().clean(df, now=datetime(2026, 10, 15, 18, 0))
    assert len(out) == 1