import numpy as np 
import pandas as pd 
from models.xgb_model import XGBForecaster
from models.metrics import Metrics
from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import ScaledVIXForecaster
from models.splitter import Timesplitter

def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    splits= Timesplitter().split(table)
    valid = splits.valid

    print(f"train {len(splits.train)} | valid {len(valid)}")
    print(f"validation period: {valid.index.min().date()} to {valid.index.max().date()}")

    rows= []
    for model in [ScaledVIXForecaster(), XGBForecaster(log_target=False), XGBForecaster(log_target=True)]:
        model.fit(splits.train)
        forecast = model.predict(valid)
        rows.append({"Model" : model.name,
                **Metrics.summarize(valid['target'], forecast)})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__=="__main__":
    main()