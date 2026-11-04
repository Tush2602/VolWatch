import numpy as np
import pandas as pd
from arch import arch_model

from models.base import Forecaster


class GarchForecaster(Forecaster):
    """GARCH(1,1) fitted once on training returns, rescaled to our volatility target."""

    name = "garch_scaled"
    TRADING_DAYS = 252

    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
        self.init_var = None
        self.scale = None

    def fit(self, train: pd.DataFrame) -> "GarchForecaster":
        r = train["abs_return"].to_numpy(dtype=float)  # % returns; sign does not matter for zero-mean GARCH
        result = arch_model(r, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(disp="off")
        params = result.params
        self.omega = float(params["omega"])
        self.alpha = float(params["alpha[1]"])
        self.beta = float(params["beta[1]"])
        self.init_var = float(np.mean(r ** 2))

        raw = self._raw_forecast(train)
        self.scale = float(train["target"].mean() / raw.mean())  # learned on training rows only
        return self

    def _raw_forecast(self, features: pd.DataFrame) -> pd.Series:
        h = self.init_var
        out = np.empty(len(features))
        for i, r in enumerate(features["abs_return"].to_numpy(dtype=float)):
            h = self.omega + self.alpha * r ** 2 + self.beta * h
            out[i] = h
        vol = np.sqrt(out) * np.sqrt(self.TRADING_DAYS)  # daily % -> annualized %
        return pd.Series(vol, index=features.index)

    def predict(self, features: pd.DataFrame) -> pd.Series:
        if self.omega is None:
            raise RuntimeError("call fit() before predict()")
        return (self._raw_forecast(features) * self.scale).rename("forecast")