
import numpy as np 
import pandas as pd 

from features.volatility import VolatilityEstimator


class FeatureBuilder:
    def __init__(self, estimator : VolatilityEstimator, lags : tuple = (1,2,5), windows :tuple = (5 , 22)):
        self.estimator= estimator
        self.lags =lags
        self.windows = windows

    @property
    def feature_names(self)-> list[str]:
        names = ['vol']
        names += [f'vol_lag{k}' for k in self.lags]
        names += [f'vol_mean{k}' for k in self.windows]
        names += ['abs_return', 'vix', 'vix_change']
        return names
    
    def build(self, nifty: pd.DataFrame, vix:pd.DataFrame) -> pd.DataFrame:
        vol = self.estimator.estimate(nifty)
        df =pd.DataFrame(index=nifty.index)
        df['vol'] = vol
        for k in self.lags:
            df[f'vol_lag{k}'] = vol.shift(k)
            for w in self.windows:
                df[f'vol_mean{w}'] = vol.rolling(w).mean()

        df['abs_return'] = np.log(nifty['Close']/nifty['Close'].shift(1)).abs() * 100
        
        vix_close = vix['Close'].reindex(df.index).ffill(limit=3)
        df['vix'] = vix_close
        df['vix_change'] = vix_close.diff()
        df['target'] = vol.shift(-1)

        return df.dropna(subset=self.feature_names)
