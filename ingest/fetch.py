import pandas as pd 
import yfinance as yf

import time 
from dataclasses import dataclass, field

class MarketDataFetcher:
    """
    Download daily OHLC data from yahoo finance , retrying on failure
    """

    def __init__(self, start : str = '2008-01-01', retries :int = 3, backoff_second :int =2):
        self.start= start
        self.retries = retries
        self.backoff_second = backoff_second   #backoff second defines time to wait before retrying

    def fetch(self, ticker:str) -> pd.DataFrame:
        for attempt in range(self.retries):
            try:
                df = yf.download(ticker, start= self.start, progress=False, auto_adjust= True)
                if not df.empty:
                    return self._flatten_columns(df)
                print(f"attempt {attempt}, Empty DataFrame")
            except Exception as e:
                print(f"attempt : {attempt} error at {e}")
            time.sleep(self.backoff_second * attempt)
        return pd.DataFrame()

    @staticmethod
    def _flatten_columns(df : pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
