import pandas as pd
import statsmodels.api as sm

pandas_example = __import__("stock_data_analysis.4_pandas_example", fromlist=["load_stock_prices"])

# Load your data (replace these lines with your actual data loading code)
stock_returns = pandas_example.load_stock_prices("AAPL")["Close"].pct_change().dropna()
market_returns = pandas_example.load_stock_prices("^DJI")["Close"].pct_change().dropna()

# Aligning the data
aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
aligned_data.columns = ["Stock_Returns", "Market_Returns"]

# Linear regression
X = sm.add_constant(aligned_data["Market_Returns"])  # Adds a constant term to the predictor
model = sm.OLS(aligned_data["Stock_Returns"], X)
results = model.fit()

# Output the results
print(results.summary())
