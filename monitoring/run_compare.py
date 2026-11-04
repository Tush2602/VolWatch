import numpy as np
import pandas as pd

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import ScaledVIXForecaster
from models.metrics import Metrics
from models.run_final_test import XGB_PARAMS
from models.xgb_model import XGBForecaster
from monitoring.compare import NeverMonitor, ScheduleMonitor
from monitoring.replay import PolicyReplay
from monitoring.run_replay import REFERENCE_DAYS, RECENT_DAYS, SCENARIOS, STEP, build_monitor

STRICT = {"min_effect": 0.50, "min_features": 4}   # chosen from the calibration grid only


def strategies(features: list) -> dict:
    return {
        "never retrain": NeverMonitor(),
        "every 63 days": ScheduleMonitor(63),
        "drift: either (0.4 / 3)": build_monitor(features, "either"),
        "drift: performance only": build_monitor(features, "performance"),
        "drift: both signals": build_monitor(features, "both"),
        "drift: either, strict (0.5 / 4)": build_monitor(features, "either", **STRICT),
    }


def main() -> None:
    db = Database()
    builder = FeatureBuilder(VolatilityEstimator())
    table = builder.build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))

    for scenario, config in SCENARIOS.items():
        rows = []
        for name, monitor in strategies(builder.feature_names).items():
            replay = PolicyReplay(
                table, monitor,
                model_factory=lambda: XGBForecaster(log_target=False, **XGB_PARAMS),
                benchmark_factory=ScaledVIXForecaster,
                step=STEP, recent_days=RECENT_DAYS, reference_days=REFERENCE_DAYS, **config,
            )
            records = replay.run()
            fc = replay.forecasts
            scores = Metrics.summarize(fc["actual"], fc["forecast"])

            stress = (fc.index >= "2020-03-01") & (fc.index <= "2020-06-30")
            qlike_stress = Metrics.qlike(fc["actual"][stress], fc["forecast"][stress]) if stress.any() else np.nan
            rows.append({"strategy": name, "retrains": int(records["retrain"].sum()), **scores,
                         "qlike_mar_jun_2020": qlike_stress})

        print(f"\n=== {scenario} ===")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()