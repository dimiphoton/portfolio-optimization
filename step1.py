import os
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

from constants import DATA_PATH, FIGURE_PATH, FORECAST_PATH

def load_csv(stock_name: str, forecaster: bool = False) -> pd.DataFrame:
    path = FORECAST_PATH if forecaster else DATA_PATH
    directory = os.path.join(path, stock_name)
    file_path = os.path.join(directory, "stock_data.csv")
    date_parser = lambda x: pd.to_datetime(x, utc=True)
    df = pd.read_csv(file_path, parse_dates=['Date'], date_parser=date_parser)
    df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
    df['ds'] = df['ds'].dt.tz_localize(None)
    return df

def fit_model(stock_name: str, save: bool = False, model=Prophet, forecaster: bool = False):
    df = load_csv(stock_name, forecaster)
    model = model()
    model.fit(df)
    if save:
        path = FORECAST_PATH if forecaster else DATA_PATH
        directory = os.path.join(path, stock_name)
        model.save('prophet_model.pkl')
    return model

def plot_training_accuracy(stock_name: str, model: Prophet, save: bool = True, show: bool = True, forecaster: bool = False) -> None:
    df = load_csv(stock_name, forecaster)
    forecast = model.predict(df)

    plt.plot(df['ds'], df['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.legend()
    plt.title('Training Set Accuracy')
    plt.xlabel('Date')
    plt.ylabel('Value')
    if save:
        directory = os.path.join(FIGURE_PATH, stock_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plot_path = os.path.join(directory, "training_set_accuracy.png")
        plt.savefig(plot_path)
    if show:
        plt.show()

def forecast(stock_name: str, model: Prophet, horizon: int, forecaster: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)

    path = FORECAST_PATH if forecaster else DATA_PATH
    directory = os.path.join(path, 'forecasted')
    if not os.path.exists(directory):
        os.makedirs(directory)
    forecast_path = os.path.join(directory, f"{stock_name}.csv")
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).to_csv(forecast_path, index=False)
    return forecast[['ds', 'yhat']], forecast[['ds', 'yhat_lower', 'yhat_upper']]

def plot_forecast(stock_name: str, forecast: pd.DataFrame, ci: pd.DataFrame, save: bool = True, show: bool = True) -> None:
    """
    Plot the forecasted value and confidence intervals.

    Args:
        stock_name (str): The name of the stock.
        forecast (pd.DataFrame): The forecasted data.
        ci (pd.DataFrame): The confidence intervals.
        save (bool): Whether to save the plot as a PNG file.
        show (bool): Whether to display the plot.
        forecaster (bool): Whether to use forecasted data.
    """
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.fill_between(ci['ds'], ci['yhat_lower'], ci['yhat_upper'], alpha=0.2)
    plt.legend()
    plt.title('Forecast')
    plt.xlabel('Date')
    plt.ylabel('Value')
    if save:
        path = FORECAST_PATH
        directory = os.path.join(path, stock_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plot_path = os.path.join(directory, "forecast.png")
        plt.savefig(plot_path)
    if show:
        plt.show()

