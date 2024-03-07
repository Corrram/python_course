from matplotlib import pyplot as plt

from stock_data_analysis_oop.data_loader import DataLoader


class InvestmentTester:
    def __init__(self, ticker, start_date=None):
        self.prices = DataLoader(start_date).load_stock_prices(ticker)

    def test_savings_plan(self, interval=30, amount=None):
        if amount is None:
            amount = interval
        # invest amount every interval days
        investment_times = self.prices.index[::interval]
        investment_values = self.prices.loc[investment_times]
        investment_values["Amount"] = amount
        investment_values["Shares"] = amount / investment_values["Close"]
        investment_values["Total Shares"] = investment_values["Shares"].cumsum()
        investment_values["Total Value"] = (
            investment_values["Total Shares"] * investment_values["Close"]
        )
        values = investment_values["Total Value"]
        values.name = f"{interval} days interval"
        values.plot()

    def refine_plot(self):
        plt.grid()
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.show()
        plt.close()


if __name__ == "__main__":
    matic = "AAPL"
    start_date = "2023"
    tester = InvestmentTester(matic, start_date)
    tester.test_savings_plan(50)
    tester.test_savings_plan(10)
    tester.test_savings_plan(1)
    tester.refine_plot()
    print("Done!")
