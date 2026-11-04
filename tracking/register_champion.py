import pandas as pd 

from db.client import Database
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.metrics import Metrics
from models.splitter import Timesplitter
from models.run_final_test import XGB_PARAMS
from models.xgb_model import XGBForecaster
from tracking.pyfunc_wrapper import ForecasterModel
from tracking.registry import ModelRegistry
from tracking.tracker import ExperimentTracker

MODEL_NAME = "volwatch-xgb"

def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    splits = Timesplitter().split(table)
    fit_data = pd.concat([splits.train, splits.valid])
    test =splits.test

    model = XGBForecaster(log_target=False, **XGB_PARAMS).fit(fit_data)
    scores = Metrics.summarize(test['target'], model.predict(test))

    tracker = ExperimentTracker()
    registry = ModelRegistry(tracker, MODEL_NAME)
    run_id = tracker.log_run(
        run_name= "xgb_frozer_final",
        params = {**model.params, "log_target" : model.log_target, "n_features" : len(model.feature_cols)},
        metrics= {f"test_{name}":value for name, value in scores.items()},
        tags = {"model" : "xgb", "stage" : "final_test", "fit_rows" : str(len(fit_data)),
                "data_end" : str(table.index.max().date())},
        model=ForecasterModel(model),
        registered_model_name=MODEL_NAME
    )
    version = registry.version_for_run(run_id)
    registry.set_version_tag(version, "fit_end", str(fit_data.index.max().date()))
    promoted = registry.promote_if_better(version, metric="test_qlike")
    print({k:round(v, 2) for k, v in scores.items()})
    print(f"registerd {MODEL_NAME} version {version}; promoted to champion : {promoted}")

if __name__ =="__main__":
    main()
