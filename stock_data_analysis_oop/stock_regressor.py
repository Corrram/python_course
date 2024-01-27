import matplotlib.pyplot as plt
import statsmodels.api as sm

from stock_data_analysis_oop.data_loader import DataLoader


class StockRegressor:
    def __init__(self, ticker):
        self.ticker = ticker
        self._index_ticker = "^DJI"

    def run_regression(self):
        data_loader = DataLoader()
        stock_returns = data_loader.load_stock_returns(self.ticker)
        index_returns = data_loader.load_stock_returns(self._index_ticker)

        # keep only the intersection of dates
        common_indices = stock_returns.index.intersection(index_returns.index)
        stock_returns = stock_returns.loc[common_indices]
        index_returns = index_returns.loc[common_indices]

        # Linear regression
        X = sm.add_constant(index_returns)
        model = sm.OLS(stock_returns, X)
        results = model.fit()

        # Output the results
        print(results.summary())

        # visualize the plot
        plt.scatter(index_returns, stock_returns)
        plt.plot(index_returns, results.fittedvalues)
        plt.xlabel("Market Returns")
        plt.ylabel("Stock Returns")
        plt.show()


if __name__ == "__main__":
    StockRegressor("AAPL").run_regression()
    print("Done!")
