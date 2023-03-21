import os
import sys
import pandas as pd
from constants import DEFAULT_START_DATE, DEFAULT_END_DATE, company_dict, DATA_PATH
import yfinance as yf




def download_and_save_stock_data(stock_name: str, start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE) -> None:
    """
    Download stock data for a given stock symbol and save it as a CSV file in the 'data' folder.

    :param stock_name: Stock symbol to download data for.
    :param start_date: Start date for downloading stock data.
    :param end_date: End date for downloading stock data.
    """
    stock = yf.Ticker(stock_name)
    stock_data = stock.history(start=start_date, end=end_date)
    print(stock_data)
    stock_data = stock_data[['Date', 'Adj Close']]  # Select only the 'Date' and 'Adj Close' columns
    directory = os.path.join(DATA_PATH, stock_name)

    if not os.path.exists(directory):
        os.makedirs(directory)

    stock_data.to_csv(os.path.join(directory, "stock_data.csv"))
    print("done")



if __name__ == "__main__":
    for key,value in company_dict.items():
        print(key)