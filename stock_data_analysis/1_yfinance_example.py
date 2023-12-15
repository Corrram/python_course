import pathlib

import yfinance as yf


def download_stock_data(tickers=None):
    match tickers:
        case None:
            with open("sp500-ticker-list.txt") as f:
                stock_list = f.read().splitlines()
        case str():
            stock_list = [tickers]
        case list():
            stock_list = tickers
        case _:
            raise ValueError("tickers must be None, a string or a list of strings")
    pathlib.Path("data").mkdir(exist_ok=True)
    for stock in ["^DJI"] + stock_list:
        data = yf.download(stock, start="2010-01-01")
        data.to_csv(f"data/{stock}.csv")


if __name__ == "__main__":
    # sp500_tickers = fetch_sp500_tickers()
    download_stock_data("SPYI")
    print("Done!")
