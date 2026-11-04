from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from bs4 import FeatureNotFound
import pandas as pd 
import numpy as np 

from models.base import Forecaster

class PersistenceForecaster(Forecaster):
    name = 'Persistence'
    
    def predict(self, features:pd.DataFrame)-> pd.Series:
        return features['vol'].rename('forecast')

class EWMAForecaster(Forecaster):
    def __init__(self, decay:float = 0.94):
        self.decay =decay
        self.name = f"ewma_{decay}"

    def predict(self, features:pd.DataFrame)-> pd.Series:
        variance = (features['vol'] ** 2).ewm(alpha = 1- self.decay, adjust= False).mean()
        return np.sqrt(variance).rename('forecast')

class ScaledVIXForecaster(Forecaster):
    name = "vix_scaled"
    def __init__(self):
        self.scale =None

    def fit(self, train : pd.DataFrame)-> "ScaledVIXForecaster":
        self.scale = float(train['target'].mean() / train['vix'].mean())
        return self
    def predict(self, features : pd.DataFrame) -> pd.Series:
        if self.scale is None:
            raise RuntimeError("call fit() before predict()")
        return (features['vix'] * self.scale).rename('forecast')
