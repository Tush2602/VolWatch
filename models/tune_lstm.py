import itertools

import numpy as np 
import pandas as pd 

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import ScaledVIXForecaster
from models.lstm_model import LSTMForecaster
from models.metrics import Metrics
from models.run_final_test import XGB_PARAMS
from models.splitter import Timesplitter
from models.xgb_model import XGBForecaster

GRID = {"window" : [14, 22], "hidden": [16,32]}
SEEDS = [0,1 ,2]

def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    labeled = table.dropna(subset=["target"])
    splits = Timesplitter().split(table)
    valid = splits.valid

    print("references on validation:")
    refs = [ScaledVIXForecaster().fit(splits.train), XGBForecaster(log_target=False, **XGB_PARAMS).fit(splits.train)]
    for m in refs:
        print(f"  {m.name:12s}", {k: round(v, 3) for k, v in Metrics.summarize(valid["target"], m.predict(valid)).items()})

    rows = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        per_seed = []
        for seed in SEEDS:
            model = LSTMForecaster(seed=seed, **params).fit(splits.train)
            forecast = model.predict(labeled).loc[valid.index]
            per_seed.append(Metrics.summarize(valid["target"], forecast))
        frame = pd.DataFrame(per_seed)
        rows.append({**params,
                     "rmse": frame["rmse"].mean(), "mae": frame["mae"].mean(),
                     "qlike": frame["qlike"].mean(), "qlike_spread": frame["qlike"].std()})

    print(f"\nLSTM, mean over {len(SEEDS)} seeds, ranked by QLIKE:")
    print(pd.DataFrame(rows).sort_values("qlike").round(3).to_string(index=False))


if __name__ == "__main__":
    main()