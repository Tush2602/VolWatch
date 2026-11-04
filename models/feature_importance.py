import numpy as np 
import pandas as pd 

from db.client import Database
from models.metrics import Metrics
from models.xgb_model import XGBForecaster
from features.build_features import FeatureBuilder
from features.volatility import VolatilityEstimator
from models.splitter import Timesplitter
from models.baselines import ScaledVIXForecaster
from models.run_final_test import XGB_PARAMS

def permutation_importance(model, valid: pd.DataFrame, n_repeats : int =10, seed: int =0) -> pd.DataFrame:
    """How much qlike is affected if column are well shuffled"""
    rng= np.random.default_rng(seed)
    base = Metrics.qlike(valid['target'], forecast=model.predict(valid))
    rows =[]
    for col in model.feature_cols:
        increases = []
        for _ in range(n_repeats):
            shuffled= valid.copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            increases.append(Metrics.qlike(valid['target'], model.predict(shuffled))-base)
        rows.append({
                "feature": col,
                "qlike_increases" : np.mean(increases),
                "spread": np.std(increases)
            })
    return pd.DataFrame(rows).sort_values("qlike_increases", ascending=False)

def main() -> None:
    db = Database()
    table = FeatureBuilder(VolatilityEstimator()).build(db.read_prices("^NSEI"), db.read_prices("^INDIAVIX"))
    splits =Timesplitter().split(table)
    valid = splits.valid

    model = XGBForecaster(log_target=False, **XGB_PARAMS).fit(splits.train)

    print("--------------Model built-in importance-------------------")
    gain = pd.Series(model.model.feature_importances_, index=model.feature_cols).sort_values(ascending=False)
    print(gain.round(3).to_string())

    print("--------------Permutation importance----------------------")
    print(permutation_importance(model, valid).round(4).to_string(index=False))

    print("--------------Retrain without vix------------------- -----")
    all_cols = [c for c in splits.train.columns if c!='target']
    no_vix_cols = [c for c in all_cols if c not in ['vix', 'vix_change']]
    candidates = {
        "xgb (all features)" : model, 
        "xgb (no VIX)":  XGBForecaster(log_target=False, feature_cols=no_vix_cols, **XGB_PARAMS).fit(splits.train),
        "vix_scaled" : ScaledVIXForecaster().fit(splits.train)
    }
    rows = [{
        "model" :name,
        **Metrics.summarize(valid['target'], m.predict(valid))
    } for name, m in candidates.items()]
    print(pd.DataFrame(rows).round(3).to_string(index=False))

if __name__== "__main__":
    main()


