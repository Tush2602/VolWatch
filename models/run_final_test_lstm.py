import numpy as np
import pandas as pd

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import EWMAForecaster, PersistenceForecaster, ScaledVIXForecaster
from models.lstm_model import LSTMForecaster
from models.metrics import Metrics
from models.run_final_test import STRESS, XGB_PARAMS
from models.splitter import Timesplitter
from models.xgb_model import XGBForecaster

# Frozen after the validation search. Replace with the winner from tune_lstm, then never change.
LSTM_PARAMS = {"window": 14, "hidden": 32}
SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    labeled = table.dropna(subset=["target"])
    splits = Timesplitter().split(table)

    fit_data = pd.concat([splits.train, splits.valid])
    test = splits.test
    print(f"fit on {len(fit_data)} rows | test on {len(test)} rows")

    stress = (test.index >= STRESS[0]) & (test.index <= STRESS[1])
    periods = {
        "test (all)": np.ones(len(test), dtype=bool),
        "stress (Mar-Jun 2020)": stress,
        "test excl. stress": ~stress,
    }

    forecasts = {}
    for model in [PersistenceForecaster(), EWMAForecaster(0.94), ScaledVIXForecaster(),
                  XGBForecaster(log_target=False, **XGB_PARAMS)]:
        model.fit(fit_data)
        forecasts[model.name] = model.predict(labeled).loc[test.index]

    seed_forecasts = []
    for seed in SEEDS:
        model = LSTMForecaster(seed=seed, **LSTM_PARAMS).fit(fit_data)
        seed_forecasts.append(model.predict(labeled).loc[test.index])
    forecasts["lstm (avg of 5 seeds)"] = pd.concat(seed_forecasts, axis=1).mean(axis=1)

    rows = []
    no_vix_cols = [c for c in fit_data.columns if c not in ("target", "vix", "vix_change")]
    xgb_no_vix = XGBForecaster(log_target=False, feature_cols=no_vix_cols, **XGB_PARAMS)
    xgb_no_vix.name = "xgb_no_vix (post-hoc)"
    for name, forecast in forecasts.items():
        for label, mask in periods.items():
            scores = Metrics.summarize(test["target"].to_numpy()[mask], forecast.to_numpy()[mask])
            rows.append({"model": name, "period": label, **scores})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    print("\nLSTM single seeds, QLIKE on full test period:")
    for seed, f in zip(SEEDS, seed_forecasts):
        print(f"  seed {seed}: {Metrics.qlike(test['target'], f):.3f}")


if __name__ == "__main__":
    main()