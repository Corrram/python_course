import glob
import pathlib

import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm


def compute_beta(stock_returns, index_returns):
    covariance = stock_returns.cov(index_returns)
    variance = index_returns.var()
    return covariance / variance


def load_stock_prices(company=None):
    if company is not None:
        stock_files = [f"../stock_data_analysis/data/{company}.csv"]
    else:
        stock_files = glob.glob("../stock_data_analysis/data/*.csv")
    return {
        pathlib.Path(stock_file).stem: pd.read_csv(
            stock_file, index_col="Date", parse_dates=True
        )
        for stock_file in stock_files
    }


def compute_average_return(prices):
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    return total_return / len(prices)


def regress_total_return_on_beta(total_returns, betas):
    # OLS regression
    X = sm.add_constant(betas)
    model = sm.OLS(total_returns, X)
    results = model.fit()
    # plot regression
    plt.scatter(betas, total_returns)
    plt.plot(betas, results.fittedvalues)
    plt.xlabel("Beta")
    plt.ylabel("Total Return")
    plt.show()
    plt.close()
    return results


if __name__ == "__main__":
    stock_prices = load_stock_prices()
    # drop empty stocks
    stock_prices = {name: data for name, data in stock_prices.items() if not data.empty}
    index_prices = load_stock_prices("SPYI")["SPYI"]
    index_returns = index_prices["Close"].pct_change()
    stock_returns = {
        name: data["Close"].pct_change() for name, data in stock_prices.items()
    }
    betas = {
        stock: compute_beta(stock_returns[stock], index_returns)
        for stock in stock_returns
    }
    betas = pd.Series(betas).dropna()
    average_return = {
        stock: compute_average_return(stock_prices[stock]["Close"])
        for stock in stock_prices
    }
    average_return = pd.Series(average_return)
    average_return = average_return[betas.index]
    results = regress_total_return_on_beta(average_return, betas)
    print("Done!")
