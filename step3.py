import os
import numpy as np
import pandas as pd
import riskfolio as rp
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from step2 import download_data, calculate_returns, initialize_weights
from constants import FIGURE_PATH, company_dict

def build_portfolio(returns: pd.DataFrame) -> rp.Portfolio:
    port = rp.Portfolio(returns=returns)
    return port

def optimize_portfolio(port: rp.Portfolio, method_mu: str = 'hist', method_cov: str = 'hist', model: str = 'Classic', rm: str = 'MV', obj: str = 'Sharpe', hist: bool = True, rf: float = 0, l: float = 0) -> Dict[str, float]:
    port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
    weights = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
    return weights.to_dict()


def calculate_efficient_frontier(port: rp.Portfolio, model: str = 'Classic', rm: str = 'MV', points: int = 50, rf: float = 0, hist: bool = True) -> pd.DataFrame:
    frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
    return frontier

def plot_efficient_frontier(port: rp.Portfolio, w: Dict[str, float], frontier: pd.DataFrame, start: str, end: str, forecasted: bool = True, show: bool = True, save: bool = True) -> None:
    label = 'Max Risk Adjusted Return Portfolio'
    mu = port.mu
    cov = port.cov
    returns = port.returns

    # Convert the dictionary to a DataFrame
    w_df = pd.DataFrame(list(w.items()), columns=['Symbol', 'Weight']).set_index('Symbol').T

    ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm='MV', rf=0, alpha=0.05, cmap='viridis', w=w_df, label=label, marker='*', s=16, c='r', height=6, width=10, ax=None)

    if save:
        plot_filename = f"efficient_frontier_{start}_{end}_{'forecasted' if forecasted else 'original'}.png"
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

    # Build the portfolio
    port = build_portfolio(returns)

    # Optimize the portfolio
    optimized_weights = optimize_portfolio(port)

    # Calculate the efficient frontier
    frontier = calculate_efficient_frontier(port)

    # Plot the efficient frontier
    plot_efficient_frontier(port, optimized_weights, frontier, start=start_date, end=end_date, forecasted=forecasted_data, show=True, save=True)
