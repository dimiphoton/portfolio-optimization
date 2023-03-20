from typing import List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go


class Plotter:
    """
    This class provides various plot types to visualize stock data and portfolio performance.
    """
    def __init__(self):
        pass

    def plot_stock_prices(self, data: pd.DataFrame, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plots a line chart of the stock prices over time.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the stock prices.
            title (str, optional): The title of the chart. Defaults to None.
            save_path (str, optional): The path to save the chart. Defaults to None.
        """
        fig = px.line(data, x=data.index, y=data.columns)
        fig.update_layout(title=title)
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_stock_returns(self, data: pd.DataFrame, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plots a histogram of the daily returns for each stock.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the daily returns.
            title (str, optional): The title of the chart. Defaults to None.
            save_path (str, optional): The path to save the chart. Defaults to None.
        """
        fig = px.histogram(data, x=data.columns)
        fig.update_layout(title=title)
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_portfolio_efficient_frontier(self, data: pd.DataFrame, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plots the efficient frontier of the portfolio.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the efficient frontier data.
            title (str, optional): The title of the chart. Defaults to None.
            save_path (str, optional): The path to save the chart. Defaults to None.
        """
        fig = px.line(data, x='Volatility', y='Return', hover_name=data.index)
        fig.update_layout(title=title, xaxis_title='Volatility', yaxis_title='Return')
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_portfolio_donut(self, data: pd.DataFrame, title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plots a donut chart of the portfolio weights.

        Args:
            data (pd.DataFrame): A pandas DataFrame containing the portfolio weights.
            title (str, optional): The title of the chart. Defaults to None.
            save_path (str, optional): The path to save the chart. Defaults to None.
        """
        fig = go.Figure(data=[go.Pie(labels=data.index, values=data['Weight'], hole=.3)])
        fig.update_layout(title=title)
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()

    def plot_portfolio_simulation(self, data: List[pd.DataFrame], title: Optional[str] = None, save_path: Optional[str] = None) -> None:
        """
        Plots an animated simulation of the portfolio performance over time.

        Args:
            data (List[pd.DataFrame]): A list of pandas DataFrames containing the portfolio value and asset weights over time.
            title (str, optional): The title of the chart. Defaults to None.
        save_path (str, optional): The path to save the chart. Defaults to None.
    """
    frames = []
    for i, df in enumerate(data):
        frame = px.scatter(df, x='Weight', y='Value', animation_frame=df.index, range_y=[0, max(df['Value'])])
        frames.append(frame.frames[0])
        frames[i]['layout']['title']['text'] = f'Time: {df.index[0]}'
    fig = go.Figure(frames=frames)
    fig.update_layout(title=title, xaxis_title='Weights', yaxis_title='Portfolio Value')
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

    

