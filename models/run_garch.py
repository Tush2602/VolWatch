import numpy as np
import pandas as pd

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import EWMAForecaster, ScaledVIXForecaster
from models.garch_model import GarchForecaster
from models.metrics import Metrics
from models.run_final_test import STRESS, XGB_PARAMS
from models.splitter import Timesplitter
from models.xgb_model import XGBForecaster


def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    labeled = table.dropna(subset=["target"])
    splits = Timesplitter().split(table)

    fit_data = pd.concat([splits.train, splits.valid])
    test = splits.test

    stress = (test.index >= STRESS[0]) & (test.index <= STRESS[1])
    periods = {
        "test (all)": np.ones(len(test), dtype=bool),
        "stress (Mar-Jun 2020)": stress,
        "test excl. stress": ~stress,
    }

    garch = GarchForecaster()
    models = [EWMAForecaster(0.94), ScaledVIXForecaster(), XGBForecaster(log_target=False, **XGB_PARAMS), garch]

    rows = []
    for model in models:
        model.fit(fit_data)
        forecast = model.predict(labeled).loc[test.index]
        for label, mask in periods.items():
            scores = Metrics.summarize(test["target"].to_numpy()[mask], forecast.to_numpy()[mask])
            rows.append({"model": model.name, "period": label, **scores})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nGARCH(1,1) fitted parameters:")
    print(f"  omega={garch.omega:.4f}  alpha={garch.alpha:.4f}  beta={garch.beta:.4f}")
    print(f"  alpha+beta={garch.alpha + garch.beta:.4f}  (should be below 1)")
    print(f"  scale to our target={garch.scale:.3f}")


if __name__ == "__main__":
    main()