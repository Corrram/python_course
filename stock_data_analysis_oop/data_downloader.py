import pathlib

import yfinance as yf

from stock_data_analysis_oop.ticker_fetcher import TickerFetcher


class DataDownloader:
    def __init__(self, start_date="2010-01-01", end_date=None):
        self.start_date = start_date
        self.end_date = end_date

    def download(self, tickers):
        match tickers:
            case str():
                stock_list = [tickers]
            case list():
                stock_list = tickers
            case _:
                raise ValueError("tickers must be a string or a list of strings")
        pathlib.Path("data").mkdir(exist_ok=True)
        for stock in stock_list:
            data = yf.download(stock, start=self.start_date, end=self.end_date)
            data.to_csv(f"data/{stock}.csv")


if __name__ == "__main__":
    sp500_tickers = TickerFetcher().fetch_tickers()
    downloader = DataDownloader()
    downloader.download(["SPYI", "^DJI"])
    downloader.download(sp500_tickers)
    print("Done!")
