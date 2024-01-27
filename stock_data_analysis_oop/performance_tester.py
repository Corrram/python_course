from scipy import stats

from stock_data_analysis_oop.data_loader import DataLoader


class PerformanceTester:
    def __init__(self):
        self._data_loader = DataLoader()

    def run_t_test(self, ticker, index_ticker):
        stock_returns = self._data_loader.load_stock_returns(ticker)
        index_returns = self._data_loader.load_stock_returns(index_ticker)
        t_stat, p_value = stats.ttest_ind(stock_returns, index_returns)
        print("T-Statistic:", t_stat)
        print("P-Value:", p_value)


if __name__ == "__main__":
    PerformanceTester().run_t_test("AAPL", "^DJI")
