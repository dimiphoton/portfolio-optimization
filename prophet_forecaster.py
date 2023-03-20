import pandas as pd
from fbprophet import Prophet
from typing import List, Tuple

class ProphetForecaster:
    def __init__(self, data: pd.DataFrame, n_periods: int = 365, daily_seasonality: bool = True):
        self.data = data
        self.n_periods = n_periods
        self.daily_seasonality = daily_seasonality

    def preprocess_data(self) -> pd.DataFrame:
        data = self.data.reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
        return data[['ds', 'y']]

    def train_model(self) -> Prophet:
        model = Prophet(daily_seasonality=self.daily_seasonality)
        model.fit(self.preprocess_data())
        return model

    def generate_forecast(self, model: Prophet) -> Tuple[pd.DataFrame, pd.DataFrame]:
        future_dates = model.make_future_dataframe(periods=self.n_periods, include_history=False)
        forecast = model.predict(future_dates)
        forecast_components = model.plot_components(forecast)
        return forecast, forecast_components
