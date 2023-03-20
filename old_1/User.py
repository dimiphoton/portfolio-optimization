from typing import List
import pandas as pd
import yfinance as yf
from StockData import StockData
from Portfolio import Portfolio

class User:
    """
    A class for constructing and managing a user's investment account.
    """

    def __init__(self, name: str, email: str, initial_balance: float):
        self.name = name
        self.email = email
        self.balance = initial_balance
        self.stock_symbols = []
        self.stock_data = StockData(self.stock_symbols)
        self.portfolio = Portfolio(self.stock_symbols)
        self.transactions = pd.DataFrame(columns=['Date', 'Action', 'Symbol', 'Quantity', 'Price', 'Commission'])
    
    def update_data(self):
        """
        Update the stock data and portfolio data.
        """
        self.stock_data.update_data()
        self.portfolio.update_data()
    
    def add_funds(self, amount: float):
        """
        Add funds to the account balance.
        """
        self.balance += amount
    
    def buy_stock(self, stock_symbol: str, quantity: float, commission: float):
        """
        Buy a stock and update the account balance and transactions.
        """
        # Get the current stock price
        stock = yf.Ticker(stock_symbol)
        current_price = stock.history(period='1d')['Close'][0]
        
        # Calculate the total cost of the stock purchase
        total_cost = current_price * quantity + commission
        
        # Check if the user has enough funds to make the purchase
        if total_cost > self.balance:
            raise ValueError('Insufficient funds')
        
        # Deduct the cost of the purchase from the user's balance
        self.balance -= total_cost
        
        # Add the stock symbol to the list if it's not already there
        if stock_symbol not in self.stock_symbols:
            self.stock_symbols.append(stock_symbol)
            self.stock_data.add_stock_symbol(stock_symbol)
        
        # Add the transaction to the transactions DataFrame
        transaction = pd.DataFrame([[pd.Timestamp.now(), 'Buy', stock_symbol, quantity, current_price, commission]], columns=['Date', 'Action', 'Symbol', 'Quantity', 'Price', 'Commission'])
        self.transactions = self.transactions.append(transaction, ignore_index=True)
    
    def sell_stock(self, stock_symbol: str, quantity: float, commission: float):
        """
        Sell a stock and update the account balance and transactions.
        """
        # Get the current stock price
        stock = yf.Ticker(stock_symbol)
        current_price = stock.history(period='1d')['Close'][0]
        
        # Check if the user owns enough of the stock to make the sale
        if stock_symbol not in self.stock_symbols:
            raise ValueError('Stock symbol not found')
        elif self.stock_data.get_stock_quantity(stock_symbol) < quantity:
            raise ValueError('Insufficient shares')
        
        # Calculate the total value of the stock sale
        total_value = current_price * quantity - commission
        
        # Add the value of the sale to the user's balance
        self.balance += total_value
        
        # Add the transaction to the transactions DataFrame
        transaction = pd.DataFrame([[pd.Timestamp.now(), 'Sell', stock_symbol, quantity, current_price, commission]], columns=['Date', 'Action', 'Symbol', 'Quantity', 'Price', 'Commission'])
        self.transactions = self.transactions.append(transaction, ignore_index=True)
    
    def get_balance(self) -> float:
        """
        Get the current account balance.
        """
        return self.balance
    
    def get_portfolio_data(self) -> pd.DataFrame:
        """
        Get the current portfolio data.
        """
        return self.portfolio.get_portfolio_data()
    
    
    def get_portfolio_metrics(self) -> pd.DataFrame:
        """
        Get the current portfolio metrics.
        """
        return self.portfolio.get_portfolio_metrics()

    def get_transactions(self) -> pd.DataFrame:
        """
        Get the transaction history.
        """
        return self.transactions

    def get_stock_data(self, stock_symbol: str) -> pd.DataFrame:
        """
        Get the historical stock data for a given stock symbol.
        """
        return self.stock_data.get_stock_data(stock_symbol)
