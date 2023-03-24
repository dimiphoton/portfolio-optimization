import os
import numpy as np
import pandas as pd
import yfinance as yf
import warnings
import matplotlib.pyplot as plt
import riskfolio as rp
from typing import Dict, Union

from step1 import load_csv
from constants import DATA_PATH, FORECAST_PATH, FIGURE_PATH, company_dict

warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.4%}'.format

def read_data(assets: list, forecasted: bool = True) -> pd.DataFrame:
    """
    Download forecasted or historical price data for several assets.
    """
    if forecasted:
        data = pd.DataFrame()
        for asset in assets:
            df = load_csv(asset, forecasted)
        data.columns = assets
    else:
        data = yf.download(assets, start=start, end=end)
        data = data.loc[:, ('Adj Close', slice(None))]
        data.columns = assets

    return data

def calculate_returns(data: pd.DataFrame,forecasted=True) -> pd.DataFrame:
    """
    Calculate the percentage returns of the assets.
    """
    if forecasted:
        return data['yhat'].pct_change().dropna()
    else:
        return data['y'].pct_change().dropna()

def build_portfolio(returns: pd.DataFrame, equal_weights: bool = True) -> pd.DataFrame:
    """
    Build a portfolio object using the given returns and set the initial weights.
    """
    portfolio = rp.Portfolio(returns=returns)
    
    if equal_weights:
        num_assets = len(returns.columns)
        weights = pd.Series({asset: 1/num_assets for asset in returns.columns})
    else:
        # User can initialize the weights here
        weights = pd.Series({asset: 1/len(returns.columns) for asset in returns.columns})

    return weights

def plot_portfolio_repartition(weights: pd.Series, start: str, end: str, forecasted: bool = True, show: bool = True, save: bool = True) -> None:
    """
    Plot the portfolio repartition as a pie chart.
    """
    title = 'Portfolio Repartition'
    plot_filename = f"portfolio_repartition_{start}_{end}_{'forecasted' if forecasted else 'original'}.png"

    ax = weights.plot.pie(autopct='%.1f%%', figsize=(10, 6), cmap="tab20")
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

    # Build portfolio and initialize weights
    weights = build_portfolio(returns, equal_weights=equal_weights)

    # Plot portfolio repartition
    plot_portfolio_repartition(weights, start=start_date, end=end_date, forecasted=forecasted_data, show=True, save=True)

