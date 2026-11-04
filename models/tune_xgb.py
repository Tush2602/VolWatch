import itertools

import pandas as pd

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import ScaledVIXForecaster
from models.metrics import Metrics
from models.splitter import Timesplitter
from models.xgb_model import XGBForecaster

GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05],
    "n_estimators": [200, 400],
}


def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    splits = Timesplitter().split(table)
    valid = splits.valid

    reference = ScaledVIXForecaster().fit(splits.train)
    ref_scores = Metrics.summarize(valid["target"], reference.predict(valid))
    print("reference vix_scaled:", {k: round(v, 3) for k, v in ref_scores.items()})

    keys = list(GRID)
    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        for log_target in (False, True):
            model = XGBForecaster(log_target=log_target, **params).fit(splits.train)
            scores = Metrics.summarize(valid["target"], model.predict(valid))
            rows.append({"log": log_target, **params, **scores})

    results = pd.DataFrame(rows).sort_values("qlike")
    print(f"\ntried {len(results)} combinations, best 8 by QLIKE:")
    print(results.head(8).round(3).to_string(index=False))


if __name__ == "__main__":
    main()