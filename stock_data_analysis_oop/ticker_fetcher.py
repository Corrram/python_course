import requests
import bs4 as bs


class TickerFetcher:
    """
    Class for fetching tickers from Wikipedia.
    """

    def __init__(self, url=None):
        """
        Constructor for TickerFetcher class.

        :param url: URL to fetch tickers from.
        """
        self.url = (
            url if url else "http://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )

    def fetch_tickers(self):
        """
        Fetches the tickers from the URL.

        :return: List of tickers.
        """
        resp = requests.get(self.url)
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


if __name__ == "__main__":
    sp500_tickers = TickerFetcher().fetch_tickers()
    print("Done!")
