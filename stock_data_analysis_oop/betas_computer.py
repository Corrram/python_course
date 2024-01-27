from stock_data_analysis_oop.data_loader import DataLoader


class BetasComputer:
    def __init__(self, index_ticker="SPYI"):
        self._data_loader = DataLoader()
        self.index_returns = self._data_loader.load_stock_returns(index_ticker)

    def compute_betas(self):
        betas = {}
        data_loader = DataLoader()
        stock_returns = data_loader.load_stock_returns()
        for stock, returns in stock_returns.items():
            if returns.empty:
                continue
            betas[stock] = self._compute_beta(returns, self.index_returns)
        return betas

    @staticmethod
    def _compute_beta(stock_returns, index_returns):
        common_indices = stock_returns.index.intersection(index_returns.index)
        stock_returns = stock_returns.loc[common_indices]
        index_returns = index_returns.loc[common_indices]

        # Calculate covariance and variance
        covariance = stock_returns.cov(index_returns)
        variance = index_returns.var()

        # Calculate beta
        return covariance / variance


if __name__ == "__main__":
    betas_computer = BetasComputer()
    stock_betas = betas_computer.compute_betas()
    ticker = max(stock_betas, key=stock_betas.get)
    stock_name = DataLoader().fetch_stock_name(ticker)
    beta = round(stock_betas[ticker], 2)
    print(f"{stock_name} ({ticker}) has the highest beta, with a value of {beta}.")
