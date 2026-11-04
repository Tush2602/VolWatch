from abc import ABC, abstractmethod
import pandas as pd 
import numpy as np 

class Forecaster(ABC):
    name : str = 'forecaster'
    def fit(self, train :pd.DataFrame) -> 'Forecaster':
        return self
    
    @abstractmethod
    def predict(self, features:  pd.DataFrame) -> pd.Series:
        """Next-day volatility forecast for each row, using only information up to that row."""
    