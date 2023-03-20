from typing import List, Dict
from portfolio import Portfolio

class PortfolioOptimizer:
    def __init__(self, stock_names: List[str], start_date: str, end_date: str):
        """
        Initializes a PortfolioOptimizer object.

        Parameters:
        stock_names (List[str]): A list of stock names in the portfolio.
        start_date (str): The start date for the historical stock data.
        end_date (str): The end date for the historical stock data.
        """
        self.stock_names = stock_names
        self.start_date = start_date
        self.end_date = end_date
        self.portfolios = self._optimize_portfolios()

    def _optimize_portfolios(self) -> Dict[str, Portfolio]:
        """
        Optimizes a portfolio for each stock in the portfolio.

        Returns:
        A dictionary of Portfolio objects, where the keys are stock names.
        """
        portfolios = {}
        for stock_name in self.stock_names:
            portfolios[stock_name] = Portfolio([stock_name], self.start_date, self.end_date)
        return portfolios

    def get_portfolio_weights(self, stock_name: str) -> List[float]:
        """
        Returns the optimal portfolio weights for a given stock.

        Parameters:
        stock_name (str): The name of the stock.

        Returns:
        A list of optimal portfolio weights for the given stock.
        """
        return self.portfolios[stock_name].get_portfolio_weights()

    def get_portfolio_metrics(self, stock_name: str) -> Dict[str, float]:
        """
        Returns the expected return, volatility, Sharpe ratio, and Sortino ratio for the optimal portfolio of a given stock.

        Parameters:
        stock_name (str): The name of the stock.

        Returns:
        A dictionary of the expected return, volatility, Sharpe ratio, and Sortino ratio for the optimal portfolio of the given stock.
        """
        expected_return, volatility, sharpe_ratio, sortino_ratio = self.portfolios[stock_name].get_portfolio_metrics()
        return {'expected_return': expected_return, 'volatility': volatility, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio}


if __name__ == '__main__':
    portfolio_optimizer = PortfolioOptimizer(['AAPL', 'GOOG', 'AMZN', 'FB'], '2018-01-01', '2022-01-01')
    stock_name = 'AAPL'
    print(f'Portfolio weights for {stock_name}: {portfolio_optimizer.get_portfolio_weights(stock_name)}')
    print(f'Portfolio metrics for {stock_name}: {portfolio_optimizer.get_portfolio_metrics(stock_name)}')
