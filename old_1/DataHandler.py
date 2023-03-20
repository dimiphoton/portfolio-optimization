import os
import json
import yfinance as yf
import pandas as pd
from typing import List, Dict
from pathlib import Path

class DataHandler:
    def __init__(self, stocks_json: str = "stocks.json"):
        self.stocks_json = stocks_json
        self.stocks = self._load_stocks_from_json()
        self.data_folder = Path("data")

    def _load_stocks_from_json(self) -> Dict[str, str]:
        """Load stock tickers and their names from a JSON file."""
        with open(self.stocks_json, "r") as f:
            stocks = json.load(f)
        return stocks

    def _save_stock_data(self, stock_data: pd.DataFrame, stock_folder: Path) -> None:
        """Save stock data to a CSV file in the specified folder."""
        stock_data.to_csv(stock_folder / "stock_data.csv")

    def fetch_and_save_stock_data(self, start: str, end: str) -> None:
        """Fetch stock data for the specified date range and save it to CSV files."""
        for ticker, name in self.stocks.items():
            stock_folder = self.data_folder / name
            stock_folder.mkdir(parents=True, exist_ok=True)

            stock_data = yf.download(ticker, start=start, end=end)
            self._save_stock_data(stock_data, stock_folder)

# Example usage
if __name__ == "__main__":
    # Create a JSON file with the stock tickers and their names
    stocks = {
        "AAPL": "Apple",
        "GOOGL": "Google",
        "AMZN": "Amazon"
    }
    with open("stocks.json", "w") as f:
        json.dump(stocks, f)

    data_handler = DataHandler()
    data_handler.fetch_and_save_stock_data(start="2020-01-01", end="2023-01-01")
