import numpy as np 
import pandas as pd 

class VolatilityEstimator:
    '''Turns daily high low prices into a volatility estimate (Parkinson range estimate)'''
    TRADING_DAYS = 252

    def __init__(self, annualize: bool= True, as_percent : bool = True):
        self.annualize = annualize
        self.as_percent = as_percent

    def estimate(self, df:pd.DataFrame)-> pd.Series:
        self._validate(df)
        log_range= np.log(df['High']/df['Low'])
        variance = (log_range ** 2)/(4* np.log(2))
        vol= np.sqrt(variance)

        if self.annualize:
            vol = vol * np.sqrt(self.TRADING_DAYS)
        if self.as_percent:
            vol = vol * 100
        return vol.rename("parkinson_vol")

    @staticmethod
    def _validate(df:pd.DataFrame)-> None:
        missing = {'High', 'Low'} - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns : {missing}")

        if (df['Low'] <=0).any():
            raise ValueError("Low value can't be negative")
        if (df['High'] < df['Low']).any():
            raise ValueError("Found rows where High < low")


