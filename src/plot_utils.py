import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional
import os
import riskfolio as rp
import seaborn as sns
import unittest

from models.prophet_model import ProphetModel

def plot_efficient_frontier(portfolio: object, file_path: Optional[str] = None) -> None:
    """
    Plot the efficient frontier for the given portfolio object.

    :param portfolio: A portfolio object with portfolio weights and returns calculated.
    :param file_path: An optional path to save the plot as a PNG file.
    """
    ax = plf.plot_efficient_frontier(portfolio)
    if file_path:
        plt.savefig(file_path, dpi=300)
    plt.show()

def plot_forecast(data: pd.DataFrame, model: ProphetModel, periods: int, file_path: Optional[str] = None) -> None:
    """
    Plot stock price forecast with confidence interval using the given model.

    :param data: Input data for forecasting.
    :param model: A trained ProphetModel object.
    :param periods: Number of periods to forecast.
    :param file_path: An optional path to save the plot as a PNG file.
    """
    model.fit(data)
    future = model.model.make_future_dataframe(periods=periods)
    forecast = model.model.predict(future)
    fig = model.model.plot(forecast)
    if file_path:
        fig.savefig(file_path, dpi=300)
    plt.show()

def plot_monte_carlo_simulation(portfolio: object, periods: int, simulations: int, file_path: Optional[str] = None) -> None:
    """
    Plot Monte Carlo simulations of the portfolio return.

    :param portfolio: A portfolio object with portfolio weights and returns calculated.
    :param periods: Number of periods to simulate.
    :param simulations: Number of simulations to run.
    :param file_path: An optional path to save the plot as a PNG file.
    """
    returns = portfolio.returns
    weights = portfolio.weights
    ax = plf.plot_monte_carlo(returns, weights, periods=periods, simulations=simulations)
    if file_path:
        plt.savefig(file_path, dpi=300)
    plt.show()

def plot_donut_repartition(portfolio: object, file_path: Optional[str] = None) -> None:
    """
    Plot a donut chart representing the repartition of stocks in the portfolio.

    :param portfolio: A portfolio object with portfolio weights and returns calculated.
    :param file_path: An optional path to save the plot as a PNG file.
    """
    weights = portfolio.weights
    fig, ax = plt.subplots()
    ax.pie(weights, labels=portfolio.assets, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.gca().set(title='Stock Repartition in Portfolio')
    if file_path:
        plt.savefig(file_path, dpi=300)
    plt.show()


