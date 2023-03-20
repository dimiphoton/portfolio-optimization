from typing import List, Tuple
import yfinance as yf
import pandas as pd
import os
from datetime import datetime
from .constants import DEFAULT_TIME_RANGE, DEFAULT_FREQUENCY, get_stock_data_path

def fetch_stock_data(stock_name: str, start_date: str, end_date: str, frequency: str) -> Tuple[List[str], List[float]]:
    """
    Fetches stock data for the given stock name and date range using yfinance.
    Returns a tuple of two lists: a list of dates and a list of corresponding stock prices.
    """
    ticker = yf.Ticker(stock_name)
    data = ticker.history(start=start_date, end=end_date, interval=frequency)
    dates = data.index.strftime('%Y-%m-%d').tolist()
    prices = data['Close'].tolist()
    return dates, prices

def save_stock_data(stock_name: str, data: Tuple[List[str], List[float]]):
    """
    Saves the given stock data to a CSV file with the given stock name.
    """
    file_path = get_stock_data_path(stock_name)
    df = pd.DataFrame({'Date': data[0], 'Close': data[1]})
    df.to_csv(file_path, index=False)

def load_stock_data(stock_name: str) -> Tuple[List[str], List[float]]:
    """
    Loads stock data for the given stock name from a CSV file.
    Returns a tuple of two lists: a list of dates and a list of corresponding stock prices.
    """
    file_path = get_stock_data_path(stock_name)
    df = pd.read_csv(file_path)
    dates = df['Date'].tolist()
    prices = df['Close'].tolist()
    return dates, prices

if __name__ == '__main__':
    stock_name = 'AAPL'
    start_date, end_date = DEFAULT_TIME_RANGE
    frequency = DEFAULT_FREQUENCY
    
    # Check if stock data file exists
    file_path = get_stock_data_path(stock_name)
    if os.path.exists(file_path):
        # Load stock data from file
        dates, prices = load_stock_data(stock_name)
    else:
        # Fetch stock data using yfinance and save to file
        dates, prices = fetch_stock_data(stock_name, start_date, end_date, frequency)
        save_stock_data(stock_name, (dates, prices))
