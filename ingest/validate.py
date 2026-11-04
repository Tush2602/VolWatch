from dataclasses import dataclass, field
import pandas as pd 
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

@dataclass
class QualityReport:
    name : str
    ticker:str
    rows:int =0
    first_date : object = None
    last_date : object = None
    days_stale : int = 0
    missing_close : int = 0
    longest_gap_days : int = 0
    gaps_over_limit: int = 0
    high_equals_low : int = 0
    low_above_high: int = 0
    non_positive_close: int = 0
    high_below_open_close: int = 0
    low_above_open_close :int = 0
    big_gap_ends: list = field(default_factory=list)
    problems: list = field(default_factory=list)

    def print(self) -> None:
        print(f"\n------{self.name} ({self.ticker}------)")
        print(f"rows : {self.rows}")
        print(f"first date : {self.first_date}    last date  :{self.last_date}")
        print(f"days signce last row : {self.days_stale}")
        print(f"missing close values : {self.missing_close}")
        print(f"longest gap: {self.longest_gap_days} days")
        print(f"gaps over limit: {self.gaps_over_limit}")
        print(f"rows with High == Low: {self.high_equals_low}")
        print(f"rows with Low > High: {self.low_above_high}")
        print(f"rows with Close <= 0: {self.non_positive_close}")
        print(f"rows where High < max(Open, Close): {self.high_below_open_close}")
        print(f"rows where Low > min(Open, Close): {self.low_above_open_close}")
        print(f"long gaps end on: {self.big_gap_ends}")
        if self.problems:
            print("PROBLEMS:")
            for p in self.problems:
                print(f"{p}")
        else:
            print("no problems flagged")
    def is_loadable(self, min_rows: int=2500)-> bool:
        return (
            self.rows >=min_rows
            and self.missing_close ==0 
            and self.low_above_high ==0
            and self.non_positive_close ==0
            and self.high_below_open_close ==0
            and self.low_above_open_close == 0
        )


class QualityChecker:
    def __init__(self, max_gap_days: int =5, max_stale_days :int = 5, min_rows : int= 2500):
        self.max_gap_days = max_gap_days
        self.max_stale_days =max_stale_days
        self.min_rows = min_rows

    def check(self, name: str, ticker: str, df:pd.DataFrame)-> QualityReport:
        report= QualityReport(name= name, ticker=ticker)
        if df.empty:
            report.problems.append('no data returned')
            return report 
        gaps = df.index.to_series().diff().dt.days
        report.rows = len(df)
        report.first_date= df.index.min().date()
        report.last_date = df.index.max().date()
        report.days_stale = (pd.Timestamp.today().normalize() - df.index.max()).days    
        report.missing_close = int(df['Close'].isna().sum())
        report.longest_gap_days = int(gaps.max())
        report.gaps_over_limit = int((gaps > self.max_gap_days).sum())
        report.high_equals_low = int((df['High'] == df['Low']).sum())
        report.low_above_high = int((df['Low'] > df['High']).sum())
        report.non_positive_close = int((df['Close'] <= 0).sum())
        tol = 1e-6
        oc_max = df[["Open", "Close"]].max(axis=1)
        oc_min = df[["Open", "Close"]].min(axis=1)
        report.high_below_open_close = int((df["High"] < oc_max - tol).sum())
        report.low_above_open_close = int((df["Low"] > oc_min + tol).sum())
        report.big_gap_ends = [d.date() for d in gaps[gaps > self.max_gap_days].index]


        self._flag_problems(report)
        return report

    def _flag_problems(self, r: QualityReport) -> None:
        if r.rows < self.min_rows:
            r.problems.append(f"There are only {r.rows} rows and we want at least {self.min_rows}.")
        if r.days_stale > self.max_stale_days:
            r.problems.append(f"data is stale : last row : {r.days_stale} ago")
        if r.gaps_over_limit > 0:
            r.problems.append(f"{r.gaps_over_limit} gaps longer than {self.max_gap_days} days")
        if r.missing_close > 0:
            r.problems.append(f"{r.missing_close} missing Close values")
        if r.low_above_high > 0:
            r.problems.append(f"{r.low_above_high} rows where Low > High")
        if r.non_positive_close > 0:
            r.problems.append(f"{r.non_positive_close} rows with Close <= 0")
        if r.high_equals_low > 0:
            r.problems.append(f"{r.high_equals_low} rows where High == Low (check before using range volatility)")
        if r.high_below_open_close > 0:
            r.problems.append(f"{r.high_below_open_close} rows where High < Open or Close (inconsistent)")
        if r.low_above_open_close > 0:
            r.problems.append(f"{r.low_above_open_close} rows where Low > Open or Close (inconsistent)")


class BarFilter:
    MARKET_CLOSE = time(15, 30)
    OHLC = ["Open", "High", "Low", "Close"]

    def __init__(self, settle_minutes : int =30, timezone :str = 'Asia/Kolkata'):
        self.settle_minutes = settle_minutes
        self.timezone = timezone
    
    def clean(self, df:pd.DataFrame, now: datetime| None = None):
        now = now or datetime.now(ZoneInfo(self.timezone))
        now = now.replace(tzinfo=None)

        out = df[~df.index.duplicated(keep="last")].sort_index()
        dropped = {}
        
        rules = {
            "missing values": lambda df: df[self.OHLC].isna().any(axis=1),
            "non positive price": lambda df: (df[self.OHLC] <= 0).any(axis=1),
            "low above high": lambda df: df["Low"] > df["High"],
            "unfinished today bar": lambda df: self._unfinished_mask(df, now),
        }

        for name, rule in rules.items():
            mask = rule(out)
            dropped[name] = int(mask.sum())
            out = out.loc[~mask]

        return out, dropped

    
    def _unfinished_mask(self, df: pd.DataFrame, now: datetime) -> pd.Series:
        settled_at = datetime.combine(now.date(), self.MARKET_CLOSE) + timedelta(minutes=self.settle_minutes)
        is_today= pd.Series(df.index.date == now.date(), index=df.index)
        return is_today & (now < settled_at)
