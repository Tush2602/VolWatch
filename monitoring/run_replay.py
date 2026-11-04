import numpy as np
import pandas as pd

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import ScaledVIXForecaster
from models.run_final_test import XGB_PARAMS
from models.xgb_model import XGBForecaster
from monitoring.drift import DriftMonitor, FeatureDriftDetector, PerformanceDriftDetector, RetrainPolicy
from monitoring.replay import PolicyReplay

# Starting values. Calibrated on the 2012-2017 scenario only, then evaluated on 2018 onward.
MIN_EFFECT = 0.40
MIN_FEATURES = 3
MAX_RATIO = 1.35
COOLDOWN_DAYS = 20
RECENT_DAYS = 60
REFERENCE_DAYS = 750
STEP = 5

SCENARIOS = {
    "calibration (2012-2017)": {"fit_end": "2011-12-31", "start": "2012-01-01", "end": "2017-12-31"},
    "evaluation (2018 onward)": {"fit_end": "2017-12-31", "start": "2018-01-01", "end": None},
}


def build_monitor(features: list) -> DriftMonitor:
    return DriftMonitor(
        FeatureDriftDetector(features, min_effect=MIN_EFFECT, min_features=MIN_FEATURES),
        PerformanceDriftDetector(max_ratio=MAX_RATIO),
        RetrainPolicy(cooldown_days=COOLDOWN_DAYS),
    )


def summarize(name: str, records: pd.DataFrame) -> None:
    records = records.copy()
    records["year"] = pd.to_datetime(records["date"]).dt.year
    print(f"=== {name}: {len(records)} weekly checks ===")

    by_year = records.groupby("year").agg(
        checks=("retrain", "size"),
        feature_drift=("feature_drift", "sum"),
        perf_drift=("performance_drift", "sum"),
        retrains=("retrain", "sum"),
        suppressed=("suppressed", "sum"),
    )
    print(by_year.to_string())

    print("\nretrain events:")
    events = records[records["retrain"]]
    if events.empty:
        print("  none")
    for _, row in events.iterrows():
        print(f"{row['date'].date()}: {'; '.join(row['triggers'])}")

    print("share of checks that would flag feature drift (KS statistic gate only):")
    grid = []
    for effect in (0.3, 0.4, 0.5):
        for min_features in (2, 3, 4):
            share = np.mean([sum(v >= effect for v in stats.values()) >= min_features for stats in records["stats"]])
            grid.append({"min_effect": effect, "min_features": min_features, "share": round(float(share), 3)})
    print(pd.DataFrame(grid).to_string(index=False))

    ratios = records["perf_ratio"].dropna()
    print("\nshare of checks with champion/benchmark QLIKE ratio above:")
    for limit in (1.2, 1.35, 1.5):
        print(f"  {limit}: {round(float((ratios > limit).mean()), 3)}")
        

def build_monitor(features: list, mode: str = "either",
                  min_effect: float = MIN_EFFECT, min_features: int = MIN_FEATURES) -> DriftMonitor:
    return DriftMonitor(
        FeatureDriftDetector(features, min_effect=min_effect, min_features=min_features),
        PerformanceDriftDetector(max_ratio=MAX_RATIO),
        RetrainPolicy(cooldown_days=COOLDOWN_DAYS, mode=mode),
    )


def main() -> None:
    db = Database()
    builder = FeatureBuilder(VolatilityEstimator())
    table = builder.build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))

    for name, config in SCENARIOS.items():
        replay = PolicyReplay(
            table, build_monitor(builder.feature_names),
            model_factory=lambda: XGBForecaster(log_target=False, **XGB_PARAMS),
            benchmark_factory=ScaledVIXForecaster,
            step=STEP, recent_days=RECENT_DAYS, reference_days=REFERENCE_DAYS,
            **config,
        )
        summarize(name, replay.run())


if __name__ == "__main__":
    main()