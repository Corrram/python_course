import numpy as np
from matplotlib import pyplot as plt

from stock_data_analysis_oop.data_loader import DataLoader


class StockVisualizer:
    def plot_prices(self, company):
        values = DataLoader().load_stock_prices(company)
        self._plot_values(values)

    def plot_returns(self, company):
        values = DataLoader().load_stock_returns(company)
        self._plot_values(values)

    @staticmethod
    def _plot_values(values):
        plt.plot(values)
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.show()
        plt.close()


def print_mean_and_variance(**kwargs):
    for k, v in kwargs.items():
        variance = v.var()
        mean = v.mean()
        print(f"{k} variance: {variance}")
        print(f"{k} mean: {mean}")


# calculate variance and mean of stock 2
def plot_prices(**kwargs):
    for k, v in kwargs.items():
        plt.plot(v, label=k)
    plt.legend()
    plt.grid()
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()
    plt.close()


if __name__ == "__main__":
    stock1 = "APA"
    stock2 = "CLX"
    data_loader = DataLoader()
    values1 = data_loader.load_stock_prices(stock1)
    values2 = data_loader.load_stock_prices(stock2)
    returns1 = data_loader.load_stock_returns(stock1)
    returns2 = data_loader.load_stock_returns(stock2)

    # calculate correlation of daily returns
    correlation_matrix = np.corrcoef(returns1, returns2)
    correlation = correlation_matrix[0, 1]
    print(f"Correlation: {correlation}")

    print_mean_and_variance(**{stock1: values1, stock2: values2})
    # plot the returns
    plot_prices(**{stock1: values1, stock2: values2})

    # plot joint portfolio
    portfolio = (values1 + values2) / 2
    plot_prices(**{stock1: values1, stock2: values2, "Portfolio": portfolio})

    # calculate variance and mean of portfolio
    print_mean_and_variance(**{"Portfolio": portfolio})
    print("Done!")
