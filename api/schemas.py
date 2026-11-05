from datetime import date

from pydantic import BaseModel


class ForecastOut(BaseModel):
    ticker: str
    as_of_date: date
    forecast_vol: float
    model_version: int
    age_days: int
    stale: bool
    note: str
    source: str


class HistoryPoint(BaseModel):
    as_of_date: date
    forecast_vol: float
    realized_vol: float | None
    model_version: int
    source: str