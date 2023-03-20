from typing import List
import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from riskfolio.RiskFunctions import Returns

class Portfolio:
    """
    A class for constructing and optimizing a portfolio of stocks.
    """

    def __init__(self, stock_symbols: List[str]):
        self.stock_symbols = stock_symbols
        self.stock_data = StockData(stock_symbols)
        self.portfolio_data = pd.DataFrame()
        self.optimal_portfolio_weights = None
    
    def update_data(self):
        """
        Update the stock data and portfolio data.
        """
        # Update the stock data
        self.stock_data.update_data()
        
        # Calculate the portfolio data
        self._calculate_portfolio_data()
    
    def _calculate_portfolio_data(self):
        """
        Calculate the portfolio data for the current stock data.
        """
        # Get the stock prices
        stock_prices = self.stock_data.get_stock_prices(start_date='2015-01-01', end_date='2021-12-31')
        
        # Calculate the daily returns
        daily_returns = stock_prices.pct_change()
        
        # Remove the first row (which contains NaNs)
        daily_returns = daily_returns.iloc[1:]
        
        # Calculate the portfolio returns
        portfolio_returns = daily_returns.mean(axis=1)
        portfolio_returns.name = 'Portfolio'
        
        # Calculate the portfolio volatility
        portfolio_volatility = daily_returns.std(axis=1)
        portfolio_volatility.name = 'Volatility'
        
        # Combine the portfolio returns and volatility into a DataFrame
        self.portfolio_data = pd.concat([portfolio_returns, portfolio_volatility], axis=1)
    
    def optimize_portfolio(self):
        """
        Optimize the portfolio using mean-variance optimization.
        """
        # Calculate the covariance matrix
        covariance_matrix = self.stock_data.get_covariance_matrix(start_date='2015-01-01', end_date='2021-12-31')
        
        # Calculate the expected returns using the geometric method
        expected_returns = self.stock_data.get_expected_returns(start_date='2015-01-01', end_date='2021-12-31', method='geometric')
        
        # Create the EfficientFrontier object
        ef = EfficientFrontier(expected_returns, covariance_matrix)
        
        # Set the constraints
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: sum(w) == 1)
        
        # Optimize the portfolio
        self.optimal_portfolio_weights = ef.max_sharpe()
    
    def get_portfolio_data(self) -> pd.DataFrame:
        """
        Get the portfolio data.
        """
        return self.portfolio_data
    
    def get_optimal_portfolio_index(self) -> int:
        """
        Get the index of the optimal portfolio in the portfolio data.
        """
        # Calculate the returns and volatility of the optimal portfolio
        returns = (self.portfolio_data['Portfolio'] * self.optimal_portfolio_weights).sum()
        volatility = np.sqrt((self.optimal_portfolio_weights.T @ self.stock_data.get_covariance_matrix(start_date='2015-01-01', end_date='2021-12-31') @ self.optimal_portfolio_weights))
        
        # Find the index of the optimal portfolio
        optimal_portfolio_index = self.portfolio_data[(self.portfolio_data['Return'] == returns) & (self.portfolio_data['Volatility'] == volatility)].index[0]
        return optimal_portfolio_index

def get_optimal_portfolio_weights(self) -> np.ndarray:
    """
    Get the weights of the optimal portfolio.
    """
    return np.array(list(self.optimal_portfolio_weights.values()))

def get_portfolio_returns(self) -> pd.DataFrame:
    """
    Get the daily returns of the portfolio.
    """
    # Get the stock prices
    stock_prices = self.stock_data.get_stock_prices(start_date='2015-01-01', end_date='2021-12-31')
    
    # Calculate the daily returns
    daily_returns = stock_prices.pct_change()
    
    # Remove the first row (which contains NaNs)
    daily_returns = daily_returns.iloc[1:]
    
    # Calculate the portfolio returns
    portfolio_returns = (daily_returns * self.optimal_portfolio_weights).sum(axis=1)
    portfolio_returns.name = 'Portfolio'
    
    return portfolio_returns

def get_portfolio_volatility(self) -> float:
    """
    Get the volatility of the portfolio.
    """
    # Calculate the covariance matrix
    covariance_matrix = self.stock_data.get_covariance_matrix(start_date='2015-01-01', end_date='2021-12-31')
    
    # Calculate the volatility
    volatility = np.sqrt(self.optimal_portfolio_weights.T @ covariance_matrix @ self.optimal_portfolio_weights)
    
    return volatility

def get_portfolio_sharpe_ratio(self) -> float:
    """
    Get the Sharpe ratio of the portfolio.
    """
    # Calculate the portfolio returns
    portfolio_returns = self.get_portfolio_returns()
    
    # Calculate the expected returns using the geometric method
    expected_returns = self.stock_data.get_expected_returns(start_date='2015-01-01', end_date='2021-12-31', method='geometric')
    
    # Calculate the risk-free rate
    risk_free_rate = 0.0
    
    # Calculate the Sharpe ratio
    sharpe_ratio = (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std()
    
    return sharpe_ratio

def get_portfolio_sortino_ratio(self) -> float:
    """
    Get the Sortino ratio of the portfolio.
    """
    # Calculate the portfolio returns
    portfolio_returns = self.get_portfolio_returns()
    
    # Calculate the expected returns using the geometric method
    expected_returns = self.stock_data.get_expected_returns(start_date='2015-01-01', end_date='2021-12-31', method='geometric')
    
    # Calculate the risk-free rate
    risk_free_rate = 0.0
    
    # Calculate the downside deviation
    returns = portfolio_returns - expected_returns
    downside_returns = returns[returns < 0]
    downside_deviation = np.sqrt((downside_returns ** 2).mean())
    
    # Calculate the Sortino ratio
    sortino_ratio = (portfolio_returns.mean() - risk_free_rate) / downside_deviation
    
    return sortino_ratio

def get_portfolio_metrics(self) -> pd.DataFrame:
    """
    Get a DataFrame of the portfolio metrics.
    """
    # Calculate the portfolio data
    portfolio_data = self.get_portfolio_data()
    
    # Calculate the portfolio metrics
    portfolio_returns = self.get_portfolio_returns()
    portfolio_volatility = self.get_portfolio_volatility()
    portfolio_sharpe_ratio = self.get_portfolio_sharpe_ratio()
    portfolio_sortino_ratio = self.get_portfolio_sortino_ratio()
    
    # Create the DataFrame
    portfolio_metrics = pd.DataFrame({'Return': portfolio_returns.mean(), 'Volatility': portfolio_volatility, 'Sharpe Ratio': portfolio_sharpe_ratio, 'Sortino Ratio': portfolio_sortino_ratio}, index=['Portfolio'])
    
    return portfolio_metrics
