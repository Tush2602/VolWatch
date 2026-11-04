from tracking import tracker
import itertools
from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.baselines import PersistenceForecaster, ScaledVIXForecaster, EWMAForecaster
from models.metrics import Metrics
from models.splitter import Timesplitter
from models.tune_xgb import GRID
from models.xgb_model import XGBForecaster
from tracking.tracker import ExperimentTracker

SOURCE = "history_backfill"

def validation_metrics(scores:dict) -> dict:
    return {f"valid_{name}" : value for name, value in scores.items()}

def main() -> None:
    tracker = ExperimentTracker()
    existing = tracker.client.search_runs(
        [tracker.experiment_id], filter_string=f"tags.source = '{SOURCE}'", max_results=1
    )
    if existing:
        print("History already logged, nothing to do")
        return 

    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    labeled = table.dropna(subset=["target"])
    splits = Timesplitter().split(table=table)
    train, valid = splits.train, splits.valid
    base_tags  ={
        "source" : SOURCE,
        "stage" : "validation",
        "train_rows" : str(len(train)),
        "valid_rows": str(len(valid)),
        "data_end":  str(table.index.max().date())
    }
    count = 0
    for model in [PersistenceForecaster(), EWMAForecaster(0.94), ScaledVIXForecaster()]:
        model.fit(train)
        forecast = model.predict(labeled).loc[valid.index]
        scores = Metrics.summarize(valid['target'],forecast)
        tracker.log_run(model.name, {"model":model.name}, validation_metrics(scores), tags={**base_tags, "model":model.name})
        count+=1
        print(f"logged {model.name}")

    keys= list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        for log_target in (False, True):
            model = XGBForecaster(log_target=log_target, **params).fit(train)
            scores = Metrics.summarize(valid['target'], model.predict(valid))
            name = f"{model.name}_d{params['max_depth']}_lr{params['learning_rate']}_n{params['n_estimators']}"
            tracker.log_run(name, {**model.params, "log_target": log_target}, validation_metrics(scores),
                            tags={**base_tags, "model" : model.name})
            count+=1
            print(f"logged_{name}")
    print(f"\n done: {count} runs logged")

if __name__ == "__main__":
    main()




