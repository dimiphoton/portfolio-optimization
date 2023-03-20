from typing import List
import pandas as pd
import yfinance as yf
from fbprophet import Prophet
from StockData import StockData

class Forecasting:
    """
    A class for performing time series forecasting on historical stock data using Facebook Prophet.
    """

    def __init__(self, stock_symbol: str):
        self.stock_symbol = stock_symbol
        self.stock_data = StockData([self.stock_symbol])
        self.model = None
    
    def update_data(self):
        """
        Update the stock data.
        """
        self.stock_data.update_data()
    
    def train_model(self):
        """
        Train the Prophet model on the historical stock data.
        """
        # Get the historical stock data
        stock_data = self.stock_data.get_stock_data(self.stock_symbol)
        
        # Prepare the data for training
        data = stock_data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        # Create and fit the model
        self.model = Prophet(daily_seasonality=True)
        self.model.fit(data)
    
    def predict(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Predict future stock prices over a new time range.
        """
        # Create the date range
        dates = pd.date_range(start=start_date, end=end_date)
        
        # Create the DataFrame with the dates
        future_data = pd.DataFrame({'ds': dates})
        
        # Make the predictions
        forecast = self.model.predict(future_data)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    def plot_forecast(self, start_date: str, end_date: str):
        """
        Plot the predicted stock prices over a new time range.
        """
        # Predict the stock prices
        forecast = self.predict(start_date, end_date)
        
        # Plot the forecast
        self.model.plot(forecast, xlabel='Date', ylabel='Price')
    
    def save_model(self):
        """
        Save the trained model in the same folder as the stock data.
        """
        self.model.save(f'data/{self.stock_symbol}/model')
    
    def load_model(self):
        """
        Load the trained model from the same folder as the stock data.
        """
        self.model = Prophet(daily_seasonality=True)
        self.model.load(f'data/{self.stock_symbol}/model')
