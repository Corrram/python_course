import pathlib

import yfinance as yf

with open("sp500-ticker-list.txt") as f:
    stock_list = f.read().splitlines()

pathlib.Path("data").mkdir(exist_ok=True)

for stock in ["^DJI"] + stock_list:
    data = yf.download(stock, start="2009-01-01", end="2022-12-31")
    data.to_csv(f"data/{stock}.csv")

print("Done!")
