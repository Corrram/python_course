from scipy import stats

pandas_example = __import__("stock_data_analysis.4_pandas_example", fromlist=["load_stock_prices"])

stock_prices = pandas_example.load_stock_prices("AAPL")["Close"]
index_prices = pandas_example.load_stock_prices("^DJI")["Close"]

# Assuming you've loaded the stock and index data as pandas Series
# Calculate returns
stock_returns = stock_prices.pct_change().dropna()
index_returns = index_prices.pct_change().dropna()

# Two-sample T-Test to check if the difference in means is significant
t_stat, p_value = stats.ttest_ind(stock_returns, index_returns)

# Output results
print("T-Statistic:", t_stat)
print("P-Value:", p_value)
