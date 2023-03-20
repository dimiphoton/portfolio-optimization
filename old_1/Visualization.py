from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .stock_data import StockData
from .portfolio import Portfolio
from .forecasting import Forecasting

class Visualization:
    """
    A class for visualizing financial data.
    """

    def __init__(self, stock_symbols: List[str]):
        self.stock_data = StockData(stock_symbols)
        self.portfolio = Portfolio(stock_symbols)
        self.forecasting_models = {symbol: Forecasting(symbol) for symbol in stock_symbols}
    
    def update_data(self):
        """
        Update the stock data, portfolio data, and forecasting models.
        """
        self.stock_data.update_data()
        self.portfolio.update_data()
        for model in self.forecasting_models.values():
            model.update_data()
    
    def plot_stock_prices(self, start_date: str, end_date: str):
        """
        Plot the historical stock prices over a given time range.
        """
        # Get the stock prices
        stock_prices = self.stock_data.get_stock_prices(start_date, end_date)
        
        # Plot the stock prices
        plt.figure(figsize=(10, 6))
        for symbol in stock_prices.columns:
            plt.plot(stock_prices.index, stock_prices[symbol], label=symbol)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Historical Stock Prices')
        plt.legend()
        plt.show()
    
    def plot_stock_returns(self, start_date: str, end_date: str):
        """
        Plot the historical stock returns over a given time range.
        """
        # Get the daily returns
        daily_returns = self.stock_data.get_daily_returns(start_date, end_date)
        
        # Plot the daily returns
        plt.figure(figsize=(10, 6))
        for symbol in daily_returns.columns:
            plt.plot(daily_returns.index, daily_returns[symbol], label=symbol)
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title('Historical Stock Returns')
        plt.legend()
        plt.show()
    
    def plot_portfolio_returns(self):
        """
        Plot the historical portfolio returns.
        """
        # Get the portfolio returns
        portfolio_returns = self.portfolio.get_portfolio_returns()
        
        # Plot the portfolio returns
        plt.figure(figsize=(10, 6))
        plt.plot(portfolio_returns.index, portfolio_returns)
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.title('Historical Portfolio Returns')
        plt.show()
    
    def plot_portfolio_allocation(self):
        """
        Plot the current allocation of the portfolio.
        """
        # Get the portfolio data
        portfolio_data = self.portfolio.get_portfolio_data()
        
        # Plot the portfolio allocation
        plt.figure(figsize=(10, 6))
        plt.pie(portfolio_data['Weights'], labels=portfolio_data.index, autopct='%1.1f%%')
        plt.title('Current Portfolio Allocation')
        plt.show()
    
    def plot_forecast(self, stock_symbol: str, start_date: str, end_date: str):
        """
        Plot the predicted stock prices over a new time range.
        """
        # Get the forecasting model
        model = self.forecasting_models[stock_symbol]
        
        # Make the forecast and plot it
        model.plot_forecast(start_date, end_date)
