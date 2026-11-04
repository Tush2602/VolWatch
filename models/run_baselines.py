import numpy as np 
import pandas as pd 

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import EWMAForecaster, PersistenceForecaster, ScaledVIXForecaster
from models.metrics import Metrics
from models.splitter import Timesplitter

STRESS = ('2020-03-01', '2020-06-30')    #COVID 19 situation market crash 

def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    labeled  = table.dropna(subset=['target'])
    splits = Timesplitter().split(table)
    test =splits.test

    print(f"train {len(splits.train)} | valid {len(splits.valid)} | test {len(splits.test)}")
    print(f"test period : {test.index.min().date()} to {test.index.max().date()}")

    stress  = (test.index >=STRESS[0])& (test.index <= STRESS[1])
    periods = {
        "test (all)" : np.ones(len(test), dtype=bool),
        "stress (Mar-Jun 2020)" : stress,
        "test excl. stress" : ~stress
    }
    rows = []
    for model in [PersistenceForecaster(), EWMAForecaster(0.94), ScaledVIXForecaster()]:
        model.fit(splits.train)
        forecast = model.predict(labeled).loc[test.index]
        for label, mask in periods.items():
            scores = Metrics.summarize(actual = test['target'].to_numpy()[mask], forecast = forecast.to_numpy()[mask])
            rows.append({"model" : model.name, "period": label, **scores})
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__ == "__main__":
    main()