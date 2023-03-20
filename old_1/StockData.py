from typing import List
import pandas as pd
import numpy as np
import yfinance as yf

class StockData:
    """
    A class for gathering stock prices and calculating financial metrics.
    """

    def __init__(self, stock_symbols: List[str]):
        self.stock_symbols = stock_symbols
        self.stock_data = {}
    
    def update_data(self):
        """
        Update the stock data for each stock symbol in the list.
        """
        for stock_symbol in self.stock_symbols:
            # Load the historical stock data from Yahoo Finance
            stock = yf.Ticker(stock_symbol)
            historical_data = stock.history(period='max')
            
            # Calculate the daily returns and add them to the data
            daily_returns = historical_data['Close'].pct_change()
            historical_data['Return'] = daily_returns
            
            # Add the stock data to the dictionary
            self.stock_data[stock_symbol] = historical_data
    
    def get_stock_prices(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the stock prices for each stock symbol in the list.
        """
        # Create an empty DataFrame to store the stock prices
        stock_prices = pd.DataFrame()
        
        # Iterate over each stock symbol and append the stock prices to the DataFrame
        for stock_symbol in self.stock_symbols:
            stock_data = self.stock_data[stock_symbol]
            stock_prices[stock_symbol] = stock_data['Close'].loc[start_date:end_date]
        
        return stock_prices
    
    def get_stock_returns(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the daily returns for each stock symbol in the list.
        """
        # Create an empty DataFrame to store the daily returns
        stock_returns = pd.DataFrame()
        
        # Iterate over each stock symbol and append the daily returns to the DataFrame
        for stock_symbol in self.stock_symbols:
            stock_data = self.stock_data[stock_symbol]
            stock_returns[stock_symbol] = stock_data['Return'].loc[start_date:end_date]
        
        return stock_returns
    
    def get_covariance_matrix(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get the covariance matrix for the stock returns.
        """
        # Get the daily returns for each stock symbol
        stock_returns = self.get_stock_returns(start_date, end_date)
        
        # Calculate the covariance matrix
        covariance_matrix = stock_returns.cov()
        
        return covariance_matrix
    
    def get_expected_returns(self, start_date: str, end_date: str, method: str = 'mean') -> pd.DataFrame:
        """
        Get the expected returns for each stock symbol using the specified method.
        """
        # Get the daily returns for each stock symbol
        stock_returns = self.get_stock_returns(start_date, end_date)
        
        # Calculate the expected returns
        if method == 'mean':
            expected_returns = stock_returns.mean()
        elif method == 'median':
            expected_returns = stock_returns.median()
        elif method == 'geometric':
            expected_returns = np.power((1 + stock_returns).prod(axis=0), 1 / len(stock_returns)) - 1
        else:
            raise ValueError('Invalid method')
        
        return expected_returns
    
    def get_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Get the historical data for a single stock symbol.
        """
        return self.stock_data[stock_symbol]
    
    def get_stock_symbols(self) -> List[str]:
        """
        Get the list of stock symbol
            """
        return self.stock_symbols

def get_stock_data_for_folder(self, folder_path: str, start_date: str, end_date: str):
    """
    Get the historical data for each stock in the folder path.
    """
    for stock_symbol in self.stock_symbols:
        # Load the historical stock data from the file
        file_path = f"{folder_path}/{stock_symbol}.csv"
        historical_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Filter the data to the specified date range
        historical_data = historical_data.loc[start_date:end_date]
        
        # Calculate the daily returns and add them to the data
        daily_returns = historical_data['Close'].pct_change()
        historical_data['Return'] = daily_returns
        
        # Add the stock data to the dictionary
        self.stock_data[stock_symbol] = historical_data

def get_stock_names(self) -> List[str]:
    """
    Get the names of the stocks for each stock symbol in the list.
    """
    stock_names = []
    
    # Iterate over each stock symbol and append the stock name to the list
    for stock_symbol in self.stock_symbols:
        stock = yf.Ticker(stock_symbol)
        stock_info = stock.info
        stock_names.append(stock_info['shortName'])
    
    return stock_names

def add_stock_symbol(self, stock_symbol: str):
    """
    Add a new stock symbol to the list.
    """
    self.stock_symbols.append(stock_symbol)

def remove_stock_symbol(self, stock_symbol: str):
    """
    Remove a stock symbol from the list.
    """
    self.stock_symbols.remove(stock_symbol)
