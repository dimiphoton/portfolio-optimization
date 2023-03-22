import os
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
from typing import Dict, Union
import matplotlib.pyplot as plt

from step1 import load_csv
from constants import DATA_PATH, FORECAST_PATH, FIGURE_PATH, company_dict

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

def download_data(assets: list, start: str, end: str, forecasted: bool = False) -> pd.DataFrame:
    if forecasted:
        data = pd.DataFrame()
        for asset in assets:
            df = load_csv(asset, forecaster=True)
            df = df[(df['ds'] >= start) & (df['ds'] <= end)]
            data = pd.concat([data, df.set_index('ds')['y']], axis=1)
        data.columns = assets
    else:
        data = yf.download(assets, start=start, end=end)
        data = data.loc[:, ('Adj Close', slice(None))]
        data.columns = assets

    return data

def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    return data.pct_change().dropna()

def initialize_weights(assets: list, equal_weights: bool = False) -> Dict[str, float]:
    if equal_weights:
        num_assets = len(assets)
        return {asset: 1/num_assets for asset in assets}
    else:
        # User can initialize the weights here
        return {asset: 1/len(assets) for asset in assets}

def plot_portfolio_repartition(weights: Dict[str, float], start: str, end: str, forecasted: bool = True, show: bool = True, save: bool = True) -> None:
    w = pd.Series(weights)
    title = 'Portfolio Repartition'
    plot_filename = f"portfolio_repartition_{start}_{end}_{'forecasted' if forecasted else 'original'}.png"

    ax = w.plot.pie(autopct='%.1f%%', figsize=(10, 6), cmap="tab20")
    ax.set_title(title)
    ax.set_ylabel("")

    if save:
        plot_path = os.path.join(FIGURE_PATH, plot_filename)
        plt.savefig(plot_path)
    if show:
        plt.show()


if __name__ == '__main__':
    # User settings
    start_date = '2016-01-01'
    end_date = '2019-12-30'
    forecasted_data = False
    equal_weights = True

    assets = list(company_dict.keys())
    assets.sort()

    # Download data
    data = download_data(assets, start=start_date, end=end_date, forecasted=forecasted_data)

    # Calculate returns
    returns = calculate_returns(data)

    # Initialize weights
    weights = initialize_weights(assets, equal_weights=equal_weights)

    # Plot portfolio repartition
    plot_portfolio_repartition(weights, start=start_date, end=end_date, forecasted=forecasted_data, show=True, save=True)
