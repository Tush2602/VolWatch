from datetime import date
from ingest.fetch import MarketDataFetcher
from ingest.validate import QualityChecker, BarFilter
from db.client import Database

TICKERS = {
    "Nifty 50" : "^NSEI",
    "India VIX" : "^INDIAVIX" 
}

class BackFillJob:
    '''Fetch -> clean -> check -> store -> verify for each ticker'''
    def __init__(self, tickers :dict, fetcher :MarketDataFetcher, bar_filter : BarFilter, 
                checker: QualityChecker, db : Database):
        self.tickers =tickers
        self.fetcher =fetcher
        self.bar_filter = bar_filter
        self.checker = checker
        self.db = db
    def run(self,)-> bool:
        """Return true if all data are stored and verified succesfully."""
        all_ok = True
        for name, ticker in self.tickers.items():
            all_ok &=self._run_one(name, ticker)
        return all_ok
    
    def _run_one(self, name: str, ticker : str) -> bool:
        print(f"--------{name}----{ticker}-------------")
        raw = self.fetcher.fetch(ticker)
        if raw.empty:
            print("Nothing Fetched")
            return False

        df, dropped = self.bar_filter.clean(raw)
        removed = {k : v for k, v in dropped.items() if v > 0}
        print(f"Fetched {len(raw)} rows kept {len(df)}, removed : {removed or 'none'}")

        report= self.checker.check(name, ticker, df)
        if not report.is_loadable(self.checker.min_rows):
            print("Failed data not safe to store")
            report.print()
            return False
        if report.problems:
            print(f"Warnings : {report.problems} ")
        sent =self.db.upsert_prices(ticker, df)
        stored = self.db.count_rows(ticker)
        if stored != sent:
            print(f"Failed : sent {sent} rows but database holds {stored}")
            return False

        print(f"Stored and verified : {stored} rows  ({df.index.min().date()} to {df.index.max().date()})")
        return True
    
def main() -> None:
    job = BackFillJob(tickers=TICKERS,
                    fetcher=MarketDataFetcher(start='2008-01-01', retries=3),
                    bar_filter = BarFilter(),
                    checker=QualityChecker(max_gap_days=5, max_stale_days=5),
                    db = Database()
    )
    ok = job.run()
    print("\n backfill complete" if ok else "\n backfill finished with FAILURES")

if __name__ =="__main__":
    main()
