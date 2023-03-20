import yfinance as yf
from typing import Dict, List, Tuple
import pandas as pd
import pickle

# Define the number of largest companies to include in the stock list
NUM_LARGEST_COMPANIES = 10

# Define the default time range for the stock data
DEFAULT_TIME_RANGE = ('2010-01-01', '2022-12-31')

# Define the default frequency for the stock data
DEFAULT_FREQUENCY = '1d'

# Define the file extension for stock data files
STOCK_DATA_EXTENSION = '.csv'

# Define the file extension for model files
MODEL_EXTENSION = '.pkl'

def get_stock_list(num_companies: int) -> Dict[str, str]:
    """
    Returns a dictionary of the top n largest companies by market capitalization
    and their respective ticker symbols.
    """
    companies = yf.Tickers('^GSPC').tickers
    companies.sort(key=lambda x: x.info['marketCap'], reverse=True)
    return {company.info['longName']: company.ticker for company in companies[:num_companies]}

def get_stock_data_path(stock_name: str) -> str:
    """
    Returns the file path for the stock data file for the given stock name.
    """
    return f'data/{stock_name}/{stock_name}{STOCK_DATA_EXTENSION}'

def get_model_path(stock_name: str) -> str:
    """
    Returns the file path for the model file for the given stock name.
    """
    return f'data/{stock_name}/{stock_name}{MODEL_EXTENSION}'

def get_stock_data(stock_name: str) -> Tuple[List[str], List[float]]:
    """
    Returns a tuple of two lists: a list of dates and a list of corresponding stock prices
    for the given stock name.
    """
    file_path = get_stock_data_path(stock_name)
    # Load the stock data from the CSV file using pandas
    df = pd.read_csv(file_path)
    # Convert the date column to a datetime object
    df['Date'] = pd.to_datetime(df['Date'])
    # Sort the data by date in ascending order
    df.sort_values('Date', inplace=True)
    # Return the data as two lists of dates and prices
    return df['Date'].tolist(), df['Close'].tolist()

def get_model(stock_name: str):
    """
    Returns the trained model for the given stock name.
    """
    file_path = get_model_path(stock_name)
    # Load the trained model from the pickle file
    with open(file_path, 'rb') as f:
        model = pickle.load(f)
    # Return the loaded model
    return model


if __name__ == '__main__':
    # Test the get_stock_list function
    stock_list = get_stock_list(5)
    print(stock_list)

    # Test the get_stock_data function
    stock_name = 'AAPL'
    dates, prices = get_stock_data(stock_name)
    print(f'{stock_name} stock data:')
    print(dates)
    print(prices)

    # Test the get_model function
    model = get_model(stock_name)
    print(f'{stock_name} model:')
    print(model)
