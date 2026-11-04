# import numpy as np 
# import pandas as pandas
# from xgboost import XGBRegressor

# from models.base import Forecaster

# class XGBForecaster(Forecaster):
#     """Gradient Boosted Trees that learn next day volatility from the features table."""
#     FLOOR =1e-4
#     def __init__(self, log_target : bool = True):
#         pass


from typing_extensions import NoExtraItems
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from models.base import Forecaster

class XGBForecaster(Forecaster):
    FLOOR = 1e-4
    def __init__(self, log_target: bool = False, feature_cols:  list | None = None, **params):
        """
        log_target : does we wish to convert target to log scale 
        feature_cols : what input column to give to model 
        **params : Xgboost addditon setting 
        """
        self.log_target=  log_target
        self.feature_cols = feature_cols
        self.params = {
            "n_estimators" : 300,
            "max_depth" : 3,
            "learning_rate" : 0.05,
            "subsample" : 0.8,
            "colsample_bytree" :0.8,
            "random_state" : 0.42,
            **params,
        }
        self.name = "xgb_log" if log_target else "xgb"
        self.model = None
    
    def fit(self, train:pd.DataFrame) -> "XGBForecaster":
        if self.feature_cols is None:
            self.feature_cols = [c for c in train.columns if c != "target"]
        y = train['target']
        if self.log_target:
            y = np.log(np.maximum(y, self.FLOOR))

        self.model = XGBRegressor(**self.params)
        self.model.fit(train[self.feature_cols], y)
        return self
    
    def predict(self, features: pd.DataFrame)-> pd.Series:
        if self.model is None:
            raise RuntimeError("Model is not Trained Yet")
        raw = self.model.predict(features[self.feature_cols])
        if self.log_target:
            raw = np.exp(raw)
        return pd.Series(raw, index=features.index, name="forecast")
        
        






class XGBForecaster(Forecaster):
    """Gradient-boosted trees that learn next-day volatility from the feature table."""
    FLOOR = 1e-4  # same floor as Metrics, keeps log() safe

    def __init__(self, log_target: bool = False, feature_cols: list | None = None, **params):
        self.log_target = log_target
        self.feature_cols = feature_cols
        self.params = {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            **params,
        }
        self.name = "xgb_log" if log_target else "xgb"
        self.model = None

    def fit(self, train: pd.DataFrame) -> "XGBForecaster":
        if self.feature_cols is None:
            self.feature_cols = [c for c in train.columns if c != "target"]
        y = train["target"]
        if self.log_target:
            y = np.log(np.maximum(y, self.FLOOR))
        self.model = XGBRegressor(**self.params)
        self.model.fit(train[self.feature_cols], y)
        return self

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("call fit() before predict()")
        raw = self.model.predict(features[self.feature_cols])
        if self.log_target:
            raw = np.exp(raw)
        return pd.Series(raw, index=features.index, name="forecast")