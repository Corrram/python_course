import glob
import pathlib

import numpy as np
import yfinance

from stock_data_analysis_oop.data_loader import DataLoader


class HedgeFinder:
    def __init__(self):
        data = DataLoader().load_stock_returns()
        # keep only the stocks with full data
        max_len = max([len(s) for s in data.values()])
        self.data = {k: v for k, v in data.items() if len(v) == max_len}

    def find_hedge(self):
        # calculate correlation matrix
        stock_prices = list(self.data.values())
        correlation_matrix = np.corrcoef(stock_prices)

        # find the index with the lowest correlation
        min_correlation = np.inf
        min_correlation_index = None
        for i in range(len(correlation_matrix)):
            for j in range(i + 1, len(correlation_matrix)):
                if correlation_matrix[i, j] < min_correlation:
                    min_correlation = correlation_matrix[i, j]
                    min_correlation_index = (i, j)

        stock_names = list(self.data.keys())
        stock_1, stock_2 = (
            stock_names[min_correlation_index[0]],
            stock_names[min_correlation_index[1]],
        )
        return stock_1, stock_2


if __name__ == "__main__":
    hedge_finder = HedgeFinder()
    stock_1, stock_2 = hedge_finder.find_hedge()
    data_loader = DataLoader()
    stock_1_name = data_loader.fetch_stock_name(stock_1)
    stock_2_name = data_loader.fetch_stock_name(stock_2)
    print(
        f"Stocks with smallest correlation: "
        f"{stock_1} ({stock_1_name}) and {stock_2} ({stock_2_name})"
    )
    print("Done!")
