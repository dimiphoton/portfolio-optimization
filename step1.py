import pickle
import os
from typing import Type, Tuple
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from constants import DATA_PATH, FORECAST_PATH, FIGURE_PATH, company_dict

def load_csv(stock_name: str, forecaster: bool = False) -> pd.DataFrame:
    """
    Load historical stock data from a CSV file.

    Args:
        stock_name (str): Name of the stock to load data for.
        forecaster (bool, optional): If True, load data from the forecasted_data directory. Otherwise, load data from the data directory. Default is False.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the historical stock data.
    """
    if forecaster:
        path = FORECAST_PATH 

        # Construct the path to the CSV file.
        directory = os.path.join(path, stock_name)
        file_path = os.path.join(directory, "stock_data.csv")

        df = pd.read_csv(file_path)

        # Rename columns and localize the date column to None.
        df = df.rename(columns={'yhat': 'y'})
        df = df[['ds','y']]
    else:
        path = DATA_PATH

        # Construct the path to the CSV file.
        directory = os.path.join(path, stock_name)
        file_path = os.path.join(directory, "stock_data.csv")

        # Parse the date column and load the CSV file as a DataFrame.
        date_parser = lambda x: pd.to_datetime(x, utc=True)
        df = pd.read_csv(file_path, parse_dates=['Date'], date_parser=date_parser)

        # Rename columns and localize the date column to None.
        df = df.rename(columns={'Date': 'ds', 'Adj Close': 'y'})
        df['ds'] = df['ds'].dt.tz_localize(None)

    return df


def fit_model(stock_name: str, save: bool = False) -> Prophet:
    """
    Fit a Prophet model to the stock data for a given stock name.

    Args:
        stock_name (str): The name of the stock to fit the model to.
        save (bool): If True, save the model to a pickle file. Defaults to False.
        model (type): The Prophet model class to use. Defaults to Prophet.
        forecaster (bool): If True, use the forecasted data. If False (default), use historical data.

    Returns:
        Prophet: The fitted Prophet model.

    """
    df = load_csv(stock_name, forecaster=False)
    model = Prophet()
    model.fit(df)
    if save:
        path = DATA_PATH
        directory = os.path.join(path, stock_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, 'prophet_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    return model

def load_model(stock_name: str):
    """
    Load a saved Prophet model from a pickle file.

    Args:
        stock_name (str): The name of the stock to load the model for.
        model_name (str): The name of the saved model file.
        forecaster (bool, optional): If True, load the model from the forecasted_data directory. Otherwise, load the model from the data directory. Default is False.

    Returns:
        Prophet: The loaded  model (Prophet by defaut).
    """
    # Determine the path to the data directory.
    path = DATA_PATH

    # Construct the path to the model file.
    directory = os.path.join(path, stock_name)
    model_name='prophet'
    model_path = os.path.join(directory, f"{model_name}_model.pkl")

    # Load the model from the pickle file.
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    return model


def plot_training_accuracy(stock_name: str, save: bool = True, show: bool = True, forecaster: bool = False) -> None:
    """
    Plot and display the training set accuracy of the given stock's Prophet model.

    Args:
    - stock_name (str): Name of the stock for which data is to be loaded.
    - model_name (str): The name of the saved model file.
    - save (bool): Whether to save the plot or not. Default is True.
    - show (bool): Whether to display the plot or not. Default is True.
    - forecaster (bool): Whether the function is called by the forecaster or not.
                         Default is False.

    Returns:
    - None
    """
    # Load the model
    model = load_model(stock_name)

    # Load the stock data
    df = load_csv(stock_name, forecaster)

    # Generate forecast using the model
    forecast = model.predict(df)

    # Plot actual and predicted values
    plt.plot(df['ds'], df['y'], label='Actual')
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.legend()
    plt.title('Training Set Accuracy')
    plt.xlabel('Date')
    plt.ylabel('Value')

    # Save the plot if save=True
    if save:
        directory = os.path.join(FIGURE_PATH, stock_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plot_path = os.path.join(directory, "training_set_accuracy.png")
        plt.savefig(plot_path)

    # Display the plot if show=True
    if show:
        plt.show()


def forecast(stock_name: str, horizon: int,save=False):
    """
    Forecast the stock price using the Prophet model.
    
    Args:
        stock_name (str): The name of the stock to forecast.
        model (Prophet): The fitted Prophet model.
        horizon (int): The number of periods to forecast.
        forecaster (bool, optional): A flag to indicate if the function is used for forecasting or not.
                                     Defaults to False.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple of two dataframes:
                                           - The first dataframe contains the forecasted values for the next `horizon` periods.
                                           - The second dataframe contains the lower and upper bounds of the forecast.
    """
    model=load_model(stock_name)
    # Create future dates for forecasting
    future = model.make_future_dataframe(periods=horizon)
    # Forecast future stock prices using the model
    forecast = model.predict(future)

    # Save forecast to a file
    if save:
        path = FORECAST_PATH
        directory = os.path.join(path, stock_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        forecast_path = os.path.join(directory, "stock_data.csv")
        print('saved in: '+str(forecast_path))
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon).to_csv(forecast_path, index=False)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(horizon)





def plot_forecast(stock_name: str, forecast: pd.DataFrame, save: bool = True, show: bool = True) -> None:
    """
    Plots the forecasted values and the associated confidence intervals.

    Args:
        stock_name (str): The name of the stock.
        forecast (pd.DataFrame): The forecasted data, with columns ['ds', 'yhat'].
        ci (pd.DataFrame): The confidence interval data, with columns ['ds', 'yhat_lower', 'yhat_upper'].
        save (bool, optional): Whether to save the plot as a PNG file in a subdirectory named after the stock.
            Defaults to True.
        show (bool, optional): Whether to display the plot on screen. Defaults to True.

    Returns:
        None
    """
    # Check if the required columns are present

    # Plot the forecast and confidence intervals
    plt.plot(forecast['ds'], forecast['yhat'], label='Predicted')
    plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], alpha=0.2)
    plt.legend()
    plt.title(f"Forecast for {stock_name}")
    plt.xlabel('Date')
    plt.ylabel('Value')

    # Save the plot to a file if requested
    if save:
        path = FORECAST_PATH
        directory = os.path.join(path, stock_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        plot_path = os.path.join(directory, "forecast.png")
        plt.savefig(plot_path)

    # Show the plot on screen if requested
    if show:
        plt.show()


if __name__ == '__main__':
    for stock in company_dict.keys():
        print('Training model for '+str(stock))
        fit_model(stock,save=True)
        print('Forecasting 180 days of '+str(stock))
        forecast = forecast(stock, horizon=180)
        print('Plotting forecast for '+str(stock))
        plot_forecast(stock, forecast)
        print('Done for '+str(stock))
