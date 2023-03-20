from typing import List, Tuple
from pyportfolioopt import expected_returns, risk_models, EfficientFrontier
import pandas as pd
import numpy as np

class Portfolio:
    def __init__(self, stock_names: List[str], start_date: str, end_date: str):
        """
        Initializes a Portfolio object.

        Parameters:
        stock_names (List[str]): A list of stock names in the portfolio.
        start_date (str): The start date for the historical stock data.
        end_date (str): The end date for the historical stock data.
        """
        self.stock_names = stock_names
        self.start_date = start_date
        self.end_date = end_date
        self.prices = self._get_prices()
        self.weights = None
        self.expected_return = None
        self.volatility = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self._optimize_portfolio()

    def _get_prices(self) -> pd.DataFrame:
        """
        Returns a DataFrame of historical prices for the stocks in the portfolio.

        Returns:
        A DataFrame of historical prices for the stocks in the portfolio.
        """
        prices = pd.DataFrame()
        for stock_name in self.stock_names:
            data = pd.read_csv(f'data/{stock_name}/{stock_name}_data.csv', index_col=0, parse_dates=True)
            prices[stock_name] = data['Adj Close']
        prices = prices.loc[self.start_date:self.end_date]
        return prices

    def _optimize_portfolio(self) -> None:
        """
        Uses the mean-variance optimization method to find the optimal portfolio weights.
        """
        # Calculate expected returns and covariance matrix
        mu = expected_returns.mean_historical_return(self.prices)
        S = risk_models.sample_cov(self.prices)

        # Optimize the portfolio using the EfficientFrontier class from pyportfolioopt
        ef = EfficientFrontier(mu, S)
        self.weights = ef.max_sharpe()
        self.expected_return, self.volatility, self.sharpe_ratio, self.sortino_ratio = ef.portfolio_performance()

    def get_portfolio_weights(self) -> List[float]:
        """
        Returns the optimal portfolio weights.

        Returns:
        A list of optimal portfolio weights.
        """
        return [self.weights[stock_name] for stock_name in self.stock_names]

    def get_portfolio_metrics(self) -> Tuple[float, float, float, float]:
        """
        Returns the expected return, volatility, Sharpe ratio, and Sortino ratio for the optimal portfolio.

        Returns:
        A tuple of the expected return, volatility, Sharpe ratio, and Sortino ratio for the optimal portfolio.
        """
        return self.expected_return, self.volatility, self.sharpe_ratio, self.sortino_ratio
    

if __name__ == '__main__':
    portfolio = Portfolio(['AAPL', 'GOOG', 'AMZN', 'FB'], '2018-01-01', '2022-01-01')
    print(f'Portfolio weights: {portfolio.get_portfolio_weights()}')
    print(f'Portfolio metrics: {portfolio.get_portfolio_metrics()}')
