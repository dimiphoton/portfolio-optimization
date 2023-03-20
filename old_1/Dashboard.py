from typing import List, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo

class Dashboard:
    """
    A class for generating a dashboard of financial plots for a given portfolio.
    """

    def __init__(self, portfolio: 'Portfolio'):
        self.portfolio = portfolio
    
    def generate_plots(self) -> List[Tuple[str, str]]:
        """
        Generate a list of tuples, each containing a plot name and its HTML code.
        """
        plots = []
        
        # Add the stock prediction error plot
        stock_pred_error = self._get_stock_prediction_error()
        stock_pred_error_html = self._plot_to_html(stock_pred_error)
        plots.append(('Stock Prediction Error', stock_pred_error_html))
        
        # Add the stock forecast with confidence interval plot
        stock_forecast = self._get_stock_forecast()
        stock_forecast_html = self._plot_to_html(stock_forecast)
        plots.append(('Stock Forecast', stock_forecast_html))
        
        # Add the portfolio donut repartition plot
        portfolio_donut = self._get_portfolio_donut()
        portfolio_donut_html = self._plot_to_html(portfolio_donut)
        plots.append(('Portfolio Donut', portfolio_donut_html))
        
        # Add the portfolio efficient frontier plot
        portfolio_ef = self._get_portfolio_ef()
        portfolio_ef_html = self._plot_to_html(portfolio_ef)
        plots.append(('Portfolio Efficient Frontier', portfolio_ef_html))
        
        # Add the animated simulation of portfolio trajectories
        portfolio_trajectory = self._get_portfolio_trajectory()
        portfolio_trajectory_html = self._plot_to_html(portfolio_trajectory)
        plots.append(('Portfolio Trajectory', portfolio_trajectory_html))
        
        return plots
    
    def _get_stock_prediction_error(self) -> go.Figure:
        """
        Create a plot of stock prediction errors.
        """
        # Get the stock data
        stock_data = self.portfolio.get_stock_data()
        
        # Get the actual and predicted stock prices
        actual_prices = stock_data['Actual']
        predicted_prices = stock_data['Predicted']
        
        # Calculate the prediction errors
        errors = actual_prices - predicted_prices
        
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_prices.index, y=errors, mode='lines', name='Prediction Error'))
        fig.update_layout(title='Stock Prediction Error', xaxis_title='Date', yaxis_title='Prediction Error')
        
        return fig
    
    def _get_stock_forecast(self) -> go.Figure:
        """
        Create a plot of stock forecasts with confidence intervals.
        """
        # Get the stock data
        stock_data = self.portfolio.get_stock_data()
        
        # Get the actual and predicted stock prices
        actual_prices = stock_data['Actual']
        predicted_prices = stock_data['Predicted']
        
        # Get the confidence intervals
        upper_bounds = stock_data['Upper Bound']
        lower_bounds = stock_data['Lower Bound']
        
        # Create the plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=actual_prices.index, y=actual_prices, mode='lines', name='Actual'))
        fig.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices, mode='lines', name='Predicted'))
        fig.add_trace(go.Scatter(x=upper_bounds.index, y=upper_bounds, fill=None, mode='lines', line_color='gray', name='Upper Bound'))
        fig.add_trace(go.Scatter(x=lower_bounds.index, y=lower_bounds, fill='tonexty', mode='lines', line_color='gray', name='Lower Bound'))
        fig.update_layout(title='Stock Forecast', xaxis_title='Date', yaxis_title='Stock Price')
            return fig

def _get_portfolio_donut(self) -> go.Figure:
    """
    Create a donut chart showing the percentage of the portfolio invested in each stock.
    """
    # Get the portfolio data
    portfolio_data = self.portfolio.get_portfolio_data()
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Pie(labels=portfolio_data.index, values=portfolio_data['Weight']))
    fig.update_layout(title='Portfolio Donut', xaxis_title='Date', yaxis_title='Weight')
    
    return fig

def _get_portfolio_ef(self) -> go.Figure:
    """
    Create a plot of the portfolio efficient frontier.
    """
    # Get the portfolio data
    portfolio_data = self.portfolio.get_portfolio_data()
    
    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=portfolio_data['Volatility'], y=portfolio_data['Return'], mode='markers', name='Portfolio'))
    fig.add_trace(go.Scatter(x=portfolio_data['Volatility'].iloc[self.portfolio.get_optimal_portfolio_index()], y=portfolio_data['Return'].iloc[self.portfolio.get_optimal_portfolio_index()], mode='markers', name='Optimal Portfolio'))
    fig.update_layout(title='Portfolio Efficient Frontier', xaxis_title='Volatility', yaxis_title='Return')
    
    return fig

def _get_portfolio_trajectory(self) -> go.Figure:
    """
    Create an animated plot showing the performance of the portfolio over time.
    """
    # Get the portfolio data
    portfolio_data = self.portfolio.get_portfolio_data()
    
    # Create the plot
    fig = go.Figure()
    for i in range(len(portfolio_data)):
        x = portfolio_data.iloc[i]['Date']
        y = portfolio_data.iloc[i]['Return']
        fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers', name=str(x)))
    fig.update_layout(title='Portfolio Trajectory', xaxis_title='Date', yaxis_title='Return', updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 500, 'redraw': False}, 'fromcurrent': True, 'transition': {'duration': 0}}]), dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])])], frames=[go.Frame(data=[go.Scatter(x=[portfolio_data.iloc[i]['Date']], y=[portfolio_data.iloc[i]['Return']])]) for i in range(len(portfolio_data))])
    
    return fig

def _plot_to_html(self, fig: go.Figure) -> str:
    """
    Convert a plotly Figure object to HTML code.
    """
    return pyo.plot(fig, output_type='div', include_plotlyjs=False)

