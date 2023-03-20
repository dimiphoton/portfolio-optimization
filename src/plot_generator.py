import matplotlib.pyplot as plt
from typing import List, Tuple
import os

# Define the directory for saving plots
PLOT_DIRECTORY = 'figures'

def plot_performance(dates: List[str], prices: List[float], title: str, ylabel: str, plot_type: str, stock_name: str, save_plot: bool = False) -> None:
    """
    Plots the historical performance of a stock.

    Parameters:
    dates (List[str]): A list of dates for the stock prices.
    prices (List[float]): A list of stock prices.
    title (str): The title of the plot.
    ylabel (str): The label for the y-axis.
    plot_type (str): The type of plot ('line' or 'candlestick').
    stock_name (str): The name of the stock.
    save_plot (bool): If True, save the plot to a file. Default is False.
    """
    # Create the plot
    fig, ax = plt.subplots()
    if plot_type == 'line':
        ax.plot(dates, prices)
    elif plot_type == 'candlestick':
        # Create a candlestick plot using a library such as mplfinance or plotly
        pass
    else:
        raise ValueError(f'Invalid plot type: {plot_type}')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=45)

    # Save the plot to a file if requested
    if save_plot:
        os.makedirs(os.path.join(PLOT_DIRECTORY, plot_type), exist_ok=True)
        file_path = os.path.join(PLOT_DIRECTORY, plot_type, f'{plot_type}_{stock_name}.png')
        plt.savefig(file_path)

    # Show the plot
    plt.show()

def plot_efficient_frontier(risks: List[float], returns: List[float], title: str, xlabel: str, ylabel: str, stock_name: str, save_plot: bool = False) -> None:
    """
    Plots the efficient frontier for a given stock.

    Parameters:
    risks (List[float]): A list of risks for the portfolio.
    returns (List[float]): A list of expected returns for the portfolio.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    stock_name (str): The name of the stock.
    save_plot (bool): If True, save the plot to a file. Default is False.
    """
    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(risks, returns)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot to a file if requested
    if save_plot:
        os.makedirs(os.path.join(PLOT_DIRECTORY, 'efficient_frontier'), exist_ok=True)
        file_path = os.path.join(PLOT_DIRECTORY, 'efficient_frontier', f'efficient_frontier_{stock_name}.png')
        plt.savefig(file_path)

    # Show the plot
    plt.show()

def plot_portfolio_donut(portfolio_weights: List[float], stock_names: List[str], title: str, save_plot: bool = False) -> None:
    """
    Plots a donut chart showing the weight of each stock in the portfolio.

    Parameters:
    portfolio_weights (List[float]): A list of weights for each stock in the portfolio.
    stock_names (List[str]): A list of stock names in the portfolio.
    title (str): The title of the plot.
    save_plot (bool): If True, save the plot to a file. Default is False.
    """
    # Create the plot
    fig, ax = plt.subplots()
    ax.pie(portfolio_weights, labels=stock_names, autopct='%1.1f%%', startangle=90, counterclock=False)
    ax.set_title(title)

    # Save the plot to a file if requested
    if save_plot:
        os.makedirs(os.path.join(PLOT_DIRECTORY, 'portfolio_donut'), exist_ok=True)
        file_path = os.path.join(PLOT_DIRECTORY, 'portfolio_donut', f'portfolio_donut_{stock_names[0]}.png')
        plt.savefig(file_path)

    # Show the plot
    plt.show()

def plot_montecarlo_simulation(returns: List[float], title: str, xlabel: str, ylabel: str, stock_name: str, save_plot: bool = False) -> None:
    """
    Plots the results of a Monte Carlo simulation for a portfolio.
    Parameters:
    returns (List[float]): A list of portfolio returns from the simulation.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.
    ylabel (str): The label for the y-axis.
    stock_name (str): The name of the stock.
    save_plot (bool): If True, save the plot to a file. Default is False.
    """
    # Create the plot
    fig, ax = plt.subplots()
    ax.hist(returns, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the plot to a file if requested
    if save_plot:
        os.makedirs(os.path.join(PLOT_DIRECTORY, 'montecarlo_simulation'), exist_ok=True)
        file_path = os.path.join(PLOT_DIRECTORY, 'montecarlo_simulation', f'montecarlo_simulation_{stock_name}.png')
        plt.savefig(file_path)

    # Show the plot
    plt.show()


def plot_portfolio_history(portfolio_returns: List[float], benchmark_returns: List[float], title: str, ylabel: str, stock_name: str, save_plot: bool = False) -> None:
    """Plots the historical performance of a portfolio compared to a benchmark.
    Parameters:
    portfolio_returns (List[float]): A list of returns for the portfolio.
    benchmark_returns (List[float]): A list of returns for the benchmark.
    title (str): The title of the plot.
    ylabel (str): The label for the y-axis.
    stock_name (str): The name of the stock.
    save_plot (bool): If True, save the plot to a file. Default is False.
    """
    # Create the plot
    fig, ax = plt.subplots()
    ax.plot(portfolio_returns, label='Portfolio')
    ax.plot(benchmark_returns, label='Benchmark')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', labelrotation=45)
    ax.legend()

    # Save the plot to a file if requested
    if save_plot:
        os.makedirs(os.path.join(PLOT_DIRECTORY, 'portfolio_history'), exist_ok=True)
        file_path = os.path.join(PLOT_DIRECTORY, 'portfolio_history', f'portfolio_history_{stock_name}.png')
        plt.savefig(file_path)

    # Show the plot
    plt.show()


if __name__ == '__main__':
    stock_name = 'AAPL'
    dates, prices = get_stock_data(stock_name)
    model = get_model(stock_name)

    # Plot the historical performance of the stock
    plot_performance(dates, prices, f'{stock_name} Historical Performance', 'Stock Price (USD)', 'line', stock_name, save_plot=True)

    # Generate a list of random portfolio returns and risks for the efficient frontier
    returns = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]
    risks = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11]

    # Plot the efficient frontier for the stock
    plot_efficient_frontier(risks, returns, f'{stock_name} Efficient Frontier', 'Risk', 'Expected Return', stock_name, save_plot=True)

    # Generate a list of portfolio weights and corresponding stock names
    portfolio_weights = [0.4, 0.3, 0.2, 0.1]
    stock_names = ['AAPL', 'GOOG', 'AMZN', 'FB']

    # Plot the portfolio weight donut chart
    plot_portfolio_donut(portfolio_weights, stock_names, f'{stock_name} Portfolio Weight', save_plot=True)

    # Generate a list of simulated portfolio returns
    portfolio_returns = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    benchmark_returns = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3]

    # Plot the historical performance of the portfolio compared to a benchmark
    plot_portfolio_history(portfolio_returns, benchmark_returns, f'{stock_name} Portfolio History', 'Return', stock_name, save_plot=True)

    # Generate a list of simulated portfolio returns for a Monte Carlo simulation
    simulation_returns = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]

    # Plot the results of the Monte Carlo simulation
    plot_montecarlo_simulation(simulation_returns, f'{stock_name} Monte Carlo Simulation', 'Return', 'Frequency', stock_name, save_plot=True)
