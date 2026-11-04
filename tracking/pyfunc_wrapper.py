"""Wraps XGBForecaster so mlflow can save and load it"""
import mlflow.pyfunc
import pandas as pd 

from models.xgb_model import XGBForecaster

class ForecasterModel(mlflow.pyfunc.PythonModel):
    """Packaged a fitted forecaster, with its feature list and log setting, as one Mlflow model."""
    def __init__(self, forecaster : XGBForecaster):
        self.forecaster = forecaster
    
    def predict(self, model_input:pd.DataFrame, params : None) -> pd.Series:
        return self.forecaster.predict(model_input)

