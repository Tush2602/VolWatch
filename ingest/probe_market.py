from ingest.fetch import MarketDataFetcher
from ingest.validate import QualityChecker

TICKERS = {
    'Nifty 50': '^NSEI',
    'INDIA VIX' : '^INDIAVIX',
}

def main() ->None:
    fetcher = MarketDataFetcher(start='2008-01-01', retries=3)
    checker = QualityChecker(5, 5)

    for name, ticker in TICKERS.items():
        print(f"------------Fetching{name}{ticker}--------------")
        df = fetcher.fetch(ticker)
        report = checker.check(name, ticker, df)
        report.print()
        if not df.empty:
            print("Last 3 rows")
            print(df[['Open', 'Close', 'Low']].tail(3).round(2).to_string())
        
if __name__ =="__main__":
    main()