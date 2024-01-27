import glob
import pathlib

import pandas as pd
import yfinance


class DataLoader:
    def __init__(self, start_date=None, end_date=None):
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        self.start_date = start_date
        self.end_date = end_date

    def load_stock_prices(self, ticker=None):
        if ticker is None:
            stock_files = glob.glob("data/*.csv")

            stock_data = {}
            for stock_file in stock_files:
                stock_name = pathlib.Path(stock_file).stem
                stock_data[stock_name] = self.load_stock_prices(stock_name)
            return {name: data for name, data in stock_data.items()}
        filepath = f"data/{ticker}.csv"
        stock_data = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        if self.start_date is not None:
            stock_data = stock_data.loc[self.start_date :]
        if self.end_date is not None:
            stock_data = stock_data.loc[: self.end_date]
        return stock_data

    def load_stock_returns(self, ticker=None):
        stock_data = self.load_stock_prices(ticker)
        if isinstance(stock_data, dict):
            for name, data in stock_data.items():
                if data.empty:
                    continue
                stock_data[name] = data["Close"].pct_change().dropna()
        else:
            stock_data = stock_data["Close"].pct_change().dropna()
        return stock_data

    @staticmethod
    def fetch_stock_name(ticker):
        return yfinance.Ticker(ticker).info["longName"]


if __name__ == "__main__":
    ticker = "AAPL"
    data_loader = DataLoader()
    stock_name = data_loader.fetch_stock_name(ticker)
    print(f"Stock name for {ticker}: {stock_name}")
    stock_returns = data_loader.load_stock_returns(ticker)
    print(f"Average daily return for {ticker}: {stock_returns.mean()}")
    print("Done!")
