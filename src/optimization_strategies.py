from typing import List, Tuple
import numpy as np
import pandas as pd
from riskfolio.Portfolio import Portfolio

def get_optimal_portfolio(
    returns: pd.DataFrame,
    frequency: str,
    cov_estimator: str,
    target_return: float = None,
    target_risk: float = None
) -> Tuple[List[str], List[float]]:
    """
    Computes the optimal portfolio weights using the mean-variance optimization method.

    Args:
    - returns (pd.DataFrame): a DataFrame of historical returns for each stock
    - frequency (str): the frequency of the returns data, either 'D' for daily, 'W' for weekly, or 'M' for monthly
    - cov_estimator (str): the covariance matrix estimator to use, either 'sample' for sample covariance or 'ledoit_wolf' for the Ledoit-Wolf estimator
    - target_return (float, optional): the target return for the portfolio, in decimal form
    - target_risk (float, optional): the target risk for the portfolio, in decimal form

    Returns:
    - Tuple[List[str], List[float]]: a tuple of two lists representing the tickers and weights of the optimal portfolio
    """
    # Create a Portfolio object from the returns data
    p = Portfolio(returns, frequency=frequency)

    # Set the covariance matrix estimator
    if cov_estimator == 'sample':
        p.estimate_covariance()
    elif cov_estimator == 'ledoit_wolf':
        p.estimate_covariance('lw')

    # Calculate the efficient frontier and optimal portfolio weights
    if target_return is not None:
        weights = p.target_return_portfolio(target_return=target_return)
    elif target_risk is not None:
        weights = p.target_risk_portfolio(target_risk=target_risk)
    else:
        weights = p.optimization_portfolio()

    # Return the optimal portfolio weights as a tuple of tickers and weights
    tickers = list(weights.index)
    weights = weights.values.tolist()
    return tickers, weights


if __name__ == '__main__':
    # Load the stock returns data
    returns = pd.read_csv('data/stock_name/stock_data.csv', index_col=0, parse_dates=True)

    # Compute the optimal portfolio weights
    tickers, weights = get_optimal_portfolio(returns, 'D', 'ledoit_wolf', target_return=0.1)

    # Print the tickers and weights of the optimal portfolio
    print('Optimal Portfolio:')
    for ticker, weight in zip(tickers, weights):
        print(f'{ticker}: {weight:.2%}')
