import os 
import time 
from pathlib import Path

import pandas as pd 
import numpy as np 
import psycopg
from dotenv import load_dotenv
import json 

load_dotenv()

class Database:
    def __init__(self, url:str | None=None, connect_retries :int =3, backoff_seconds:int = 2):
        self.url = url or os.getenv("DATABASE_URL")
        if not self.url:
            raise RuntimeError("DATABASE URL  is not set")
        self.connect_retries = connect_retries
        self.backoff_seconds = backoff_seconds

    def _connect(self) ->  psycopg.Connection:
        last_error = None
        for attempt in range(self.connect_retries):
            try:
                return psycopg.connect(self.url, connect_timeout = 15)
            except psycopg.OperationalError as e:
                last_error = e
                time.sleep(self.backoff_seconds * (attempt+1))
        raise RuntimeError(f"Could not connect after {self.connect_retries} attempt") from last_error

    def run_script(self, script_path : str)-> None:
        sql = Path(script_path).read_text()
        with self._connect() as conn:
            conn.execute(sql)
    
    def upsert_prices(self, ticker:str, df:pd.DataFrame)-> int:
        rows= [
            (ticker, idx.date(), float(r['Open']), float(r['High']), float(r["Low"]), float(r['Close']))
            for idx, r in df.iterrows()
        ]
        sql = """
                INSERT INTO prices (ticker, trade_date, open, high, low, close)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, trade_date) DO UPDATE
            SET open = EXCLUDED.open, high = EXCLUDED.high,
                low = EXCLUDED.low, close = EXCLUDED.close,
                ingested_at = now()
        """

        with self._connect() as conn:
            with conn.cursor() as curr:
                curr.executemany(sql, rows)

            return len(rows)

    def read_prices(self, ticker: str, since=None) -> pd.DataFrame:
        sql = "SELECT trade_date, open, high, low, close FROM prices WHERE ticker = %s"
        params = [ticker]
        if since is not None:
            sql += " AND trade_date >= %s"
            params.append(pd.Timestamp(since).date())
        sql += " ORDER BY trade_date"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        df = pd.DataFrame(rows, columns=["trade_date", "Open", "High", "Low", "Close"])
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        return df.set_index("trade_date")

    def count_rows(self, ticker:str)-> int:
        with self._connect() as conn:
            return conn.execute(f"SELECT count(*) FROM prices WHERE ticker = %s", (ticker, )).fetchone()[0]

    def save_decision(self, as_of_date, retrain: bool, details: dict) -> None:
        sql = """
            INSERT INTO monitoring_decisions (as_of_date, retrain, details)
            VALUES (%s, %s, %s)
            ON CONFLICT (as_of_date) DO UPDATE
            SET retrain = EXCLUDED.retrain, details = EXCLUDED.details, created_at = now()
        """
        with self._connect() as conn:
            conn.execute(sql, (pd.Timestamp(as_of_date).date(), bool(retrain), json.dumps(details, default=str)))

    def start_run(self) -> int:
        with self._connect() as conn:
            return conn.execute("INSERT INTO pipeline_runs (status) VALUES ('RUNNING') RETURNING run_id").fetchone()[0]

    def finish_run(self, run_id: int, status: str, details: dict) -> None:
        sql = "UPDATE pipeline_runs SET finished_at = now(), status = %s, details = %s WHERE run_id = %s"
        with self._connect() as conn:
            conn.execute(sql, (status, json.dumps(details, default=str), run_id))

    def latest_run(self, status: str | None = None) -> dict | None:
        sql = "SELECT run_id, started_at, finished_at, status, details FROM pipeline_runs"
        params = []
        if status is not None:
            sql += " WHERE status = %s"
            params.append(status)
        sql += " ORDER BY run_id DESC LIMIT 1"
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        if row is None:
            return None
        return {"run_id": row[0], "started_at": row[1], "finished_at": row[2], "status": row[3],
                "details": json.loads(row[4]) if row[4] else None}

                
    def last_retrain_date(self):
        with self._connect() as conn:
            value = conn.execute("SELECT MAX(as_of_date) FROM monitoring_decisions WHERE retrain").fetchone()[0]
        return pd.Timestamp(value) if value else None

    def save_forecast(self, ticker: str, as_of_date, model_version: int, forecast_vol: float,
                      source: str = "live", overwrite: bool = True) -> None:
        conflict = (
            "DO UPDATE SET model_version = EXCLUDED.model_version, forecast_vol = EXCLUDED.forecast_vol, "
            "source = EXCLUDED.source, created_at = now()"
            if overwrite else "DO NOTHING"
        )
        sql = f"""
            INSERT INTO forecasts (ticker, as_of_date, model_version, forecast_vol, source)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (ticker, as_of_date) {conflict}
        """
        with self._connect() as conn:
            conn.execute(sql, (ticker, pd.Timestamp(as_of_date).date(), int(model_version),
                               float(forecast_vol), source))

    def read_forecasts(self, ticker: str, limit: int = 60, source: str | None = None) -> pd.DataFrame:
        sql = "SELECT as_of_date, model_version, forecast_vol, source FROM forecasts WHERE ticker = %s"
        params = [ticker]
        if source is not None:
            sql += " AND source = %s"
            params.append(source)
        sql += " ORDER BY as_of_date DESC LIMIT %s"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return pd.DataFrame(rows, columns=["as_of_date", "model_version", "forecast_vol", "source"])

    

