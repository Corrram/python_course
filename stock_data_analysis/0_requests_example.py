import requests
import bs4 as bs


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


if __name__ == '__main__':
    sp500_tickers = fetch_sp500_tickers()
    print("Done!")
