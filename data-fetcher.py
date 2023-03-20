import yfinance as yf
import os
import pandas as pd
from typing import List

from constants import Constants

class DataFetcher:
    def __init__(self, stock_list: List[str], data_dir: str = 'data'):
        self.stock_list = stock_list
        self.data_dir = data_dir

    def fetch_data(self, start_date: str, end_date: str, interval: str = '1d'):
        for stock in self.stock_list:
            stock_dir = os.path.join(self.data_dir, stock)
            if not os.path.exists(stock_dir):
                os.makedirs(stock_dir)

            data_file = os.path.join(stock_dir, 'stock_data.csv')
            if not os.path.exists(data_file):
                stock_data = yf.download(stock, start=start_date, end=end_date, interval=interval)
                stock_data.to_csv(data_file)

    def load_data(self, stock: str) -> pd.DataFrame:
        stock_dir = os.path.join(self.data_dir, stock)
        data_file = os.path.join(stock_dir, 'stock_data.csv')
        return pd.read_csv(data_file, index_col=0)
