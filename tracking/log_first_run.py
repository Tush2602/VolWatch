from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.metrics import Metrics
from models.run_final_test import XGB_PARAMS
from models.splitter import Timesplitter
from models.xgb_model import XGBForecaster
from tracking.tracker import ExperimentTracker

def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    splits =Timesplitter().split(table)

    model =XGBForecaster(log_target=False, **XGB_PARAMS).fit(splits.train)
    scores = Metrics.summarize(splits.valid['target'], model.predict(splits.valid))

    tracker = ExperimentTracker()
    run_id = tracker.log_run(
        run_name="xgb_first_run",
        params= {**model.params, "log_target" : model.log_target, "n_features" : len(model.feature_cols)},
        metrics = {f"valid_{name}" : value for name, value in scores.items()},
        tags= {
            "model" : "xgb",
            "stage" : "validation",
            "train_rows" : str(len(splits.train)),
            "data_end" : str(table.index.max().date()) 
        }
    )
    print(f"logged_name {run_id}")
    print({k : round(v, 3) for k, v in scores.items()})

if __name__ =="__main__":
    main()