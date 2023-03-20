import pandas as pd
from kats.models.prophet import ProphetModel, ProphetParams

class TimeSeriesModeler:
    def __init__(self, config):
        self.config = config

    def prepare_time_series_data(self, data):
        time_series = pd.DataFrame(data).reset_index()
        time_series.columns = ['time', 'value']
        return time_series

    def train_model(self, data):
        time_series = self.prepare_time_series_data(data)
        params = ProphetParams(seasonality_mode='multiplicative')
        model = ProphetModel(time_series, params)
        model.fit()
        return model

    def forecast_prices(self, model):
        forecast = model.predict(steps=self.config['forecast_period'], freq="D")
        return forecast['fcst']

    def train_and_validate(self, data, validation_size):
        time_series = self.prepare_time_series_data(data)
        train_size = len(time_series) - validation_size
        train_data = time_series.iloc[:train_size]
        validation_data = time_series.iloc[train_size:]
        
        params = ProphetParams(seasonality_mode='multiplicative')
        model = ProphetModel(train_data, params)
        model.fit()

        validation_forecast = model.predict(steps=validation_size, freq="D")

        return model, validation_forecast['fcst']
