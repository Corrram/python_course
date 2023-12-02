import pandas as pd
from matplotlib import pyplot as plt

stock = "^DJI"
data = pd.read_csv(f"data/{stock}.csv", index_col="Date", parse_dates=True)
data = data["Close"]
data.plot()
plt.show()
print("Done!")
