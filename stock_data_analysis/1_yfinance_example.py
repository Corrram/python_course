import pathlib

import yfinance as yf

with open("sp500-ticker-list.txt") as f:
    stock_list = f.read().splitlines()

pathlib.Path("data").mkdir(exist_ok=True)

for stock in ["^DJI"] + stock_list:
    data = yf.download(stock, start="2010-01-01")
    data.to_csv(f"data/{stock}.csv")

print("Done!")
