import glob
import pathlib

import numpy as np

# load stock data
stock_files = glob.glob("data/*.csv")

stock_data = {}
for stock_file in stock_files:
    stock_name = pathlib.Path(stock_file).stem
    stock_data[stock_name] = np.loadtxt(stock_file, delimiter=",", skiprows=1, usecols=4)

# keep only stocks with more all data points
max_data_len = max(len(data) for data in stock_data.values())
stock_data = {name: data for name, data in stock_data.items() if len(data) == max_data_len}

# calculate correlation matrix
stock_prices = list(stock_data.values())
correlation_matrix = np.corrcoef(list(stock_prices))

# find the index with the lowest correlation
min_correlation = np.inf
min_correlation_index = None
for i in range(len(correlation_matrix)):
    for j in range(i + 1, len(correlation_matrix)):
        if correlation_matrix[i, j] < min_correlation:
            min_correlation = correlation_matrix[i, j]
            min_correlation_index = (i, j)

stock_names = list(stock_data.keys())
stock_1, stock_2 = stock_names[min_correlation_index[0]], stock_names[min_correlation_index[1]]
print(f"Stocks with smallest correlation: {stock_1} and {stock_2}")
print("Done!")
