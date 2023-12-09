import numpy as np
from matplotlib import pyplot as plt

stock1 = "ATVI"
stock2 = "LUMN"
filepath1 = f"data/{stock1}.csv"
filepath2 = f"data/{stock2}.csv"
values1 = np.loadtxt(filepath1, delimiter=",", skiprows=1, usecols=4)
values2 = np.loadtxt(filepath2, delimiter=",", skiprows=1, usecols=4)

# calculate variance and mean of stock 1
stock1_variance = values1.var()
stock1_mean = values1.mean()
print(f"{stock1} variance: {stock1_variance}")
print(f"{stock1} mean: {stock1_mean}")

# calculate variance and mean of stock 2
stock2_variance = values2.var()
stock2_mean = values2.mean()
print(f"{stock2} variance: {stock2_variance}")
print(f"{stock2} mean: {stock2_mean}")

# calculate correlation
correlation_matrix = np.corrcoef(values1, values2)
correlation = correlation_matrix[0, 1]
print(f"Correlation: {correlation}")

# plot the returns
plt.plot(values1, label=stock1)
plt.plot(values2, label=stock2)
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# plot joint portfolio
portfolio = (values1 + values2) / 2
plt.plot(portfolio, label="Portfolio")
plt.legend()
plt.grid()
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()

# calculate variance and mean of portfolio
portfolio_variance = portfolio.var()
portfolio_mean = portfolio.mean()
print(f"Portfolio variance: {portfolio_variance}")
print(f"Portfolio mean: {portfolio_mean}")

print("Done!")
