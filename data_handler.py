import yfinance as yf
import os
import pandas as pd

class DataHandler:
    def __init__(self, config):
        self.config = config

    def download_stock_data(self, ticker):
        data = yf.download(
            ticker,
            start=self.config['start_date'],
            end=self.config['end_date'],
            interval=self.config['interval']
        )['Adj Close']
        return data

    def save_stock_data(self, ticker, data):
        if not os.path.exists(f"data/{ticker}"):
            os.makedirs(f"data/{ticker}")
        data.to_csv(f"data/{ticker}/stock_data.csv")

    def load_stock_data(self, ticker):
        return pd.read_csv(f"data/{ticker}/stock_data.csv", index_col=0, parse_dates=True)

    def save_trained_model(self, ticker, model):
        if not os.path.exists(f"data/{ticker}"):
            os.makedirs(f"data/{ticker}")
        model.save(f"data/{ticker}/trained_model.pkl")

    def load_trained_model(self, ticker):
        model = ProphetModel.load(f"data/{ticker}/trained_model.pkl")
        return model
