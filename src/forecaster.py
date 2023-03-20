from abc import ABC, abstractmethod
from typing import List, Tuple
from fbprophet import Prophet

class BaseForecaster(ABC):
    """
    Abstract base class for time-series forecasters.
    """
    @abstractmethod
    def train_model(self, dates: List[str], prices: List[float]):
        """
        Trains a time-series forecasting model on the given stock data.
        """
        pass
    
    @abstractmethod
    def make_prediction(self, dates: List[str], horizon: int) -> Tuple[List[str], List[float]]:
        """
        Makes a forecast for the given stock data and prediction horizon.
        """
        pass
    
    @abstractmethod
    def evaluate_model(self, dates: List[str], prices: List[float]) -> float:
        """
        Evaluates the accuracy of the forecast model on the given stock data.
        """
        pass
    
class ProphetForecaster(BaseForecaster):
    """
    Time-series forecaster using Facebook Prophet.
    """
    def __init__(self):
        self.model = None
    
    def train_model(self, dates: List[str], prices: List[float]):
        """
        Trains a time-series forecasting model on the given stock data using Facebook Prophet.
        """
        # Use the Prophet library to train a forecasting model on the given stock data
        # Save the trained model as an instance variable
        df = pd.DataFrame({'ds': dates, 'y': prices})
        model = Prophet()
        model.fit(df)
        self.model = model
    
    def make_prediction(self, dates: List[str], horizon: int) -> Tuple[List[str], List[float]]:
        """
        Makes a forecast for the given stock data and prediction horizon using Facebook Prophet.
        """
        # Use the trained Prophet model to make a forecast for the given stock data and prediction horizon
        future = self.model.make_future_dataframe(periods=horizon)
        forecast = self.model.predict(future)
        forecast_dates = forecast['ds'].dt.strftime('%Y-%m-%d').tolist()[-horizon:]
        forecast_prices = forecast['yhat'].tolist()[-horizon:]
        return forecast_dates, forecast_prices
    
    def evaluate_model(self, dates: List[str], prices: List[float]) -> float:
        """
        Evaluates the accuracy of the forecast model on the given stock data using Facebook Prophet.
        """
        # Use the trained Prophet model to make a forecast for the given stock data
        # Compare the forecast to the actual prices and return a measure of accuracy, such as RMSE
        df = pd.DataFrame({'ds': dates, 'y': prices})
        forecast = self.model.predict(df)
        rmse = np.sqrt(np.mean((forecast['yhat'] - df['y'])**2))
        return rmse
    
    def get_model(self):
        """
        Returns the trained Prophet model.
        """
        return self.model

if __name__ == '__main__':
    # Unit tests for ProphetForecaster
    import pandas as pd
    import numpy as np
    # Generate example stock data
    dates = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    prices = np.sin(np.arange(len(dates))*2*np.pi/365) + np.random.normal(0, 0.1, len(dates))

    # Test ProphetForecaster methods
    forecaster = ProphetForecaster()
    forecaster.train_model(dates=dates.strftime('%Y-%m-%d').tolist(), prices=prices.tolist())
    forecast_dates, forecast_prices = forecaster.make_prediction(dates=dates.strftime('%Y-%m-%d').tolist(), horizon=30)
    rmse = forecaster.evaluate_model(dates=dates.strftime('%Y-%m-%d').tolist(), prices=prices.tolist())
    model = forecaster.get_model()
    print(f'Forecast dates: {forecast_dates}')
    print(f'Forecast prices: {forecast_prices}')
    print(f'RMSE: {rmse}')
