import pathlib

import bs4 as bs
import requests
import yfinance as yf


def fetch_sp500_tickers():
    """
    Fetches the S&P 500 tickers from Wikipedia and saves them to a file.
    Code adapted from
    https://stackoverflow.com/questions/58890570/python-yahoo-finance-download-all-sp-500-stocks

    :return:
    """
    resp = requests.get("http://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"class": "wikitable sortable"})
    tickers = []
    for row in table.findAll("tr")[1:]:
        ticker = row.findAll("td")[0].text
        tickers.append(ticker)
    tickers = [s.replace("\n", "") for s in tickers]
    with open("sp500-ticker-list.txt", "w") as f:
        f.write("\n".join(tickers))

    return tickers


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
    sp500_tickers = fetch_sp500_tickers()
    download_stock_data(sp500_tickers)
    print("Done!")
