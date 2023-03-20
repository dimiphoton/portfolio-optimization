import streamlit as st
import pandas as pd
import yfinance as yf
from portfolio import PortfolioOptimizer
from forecaster import ProphetForecaster
from plot_generator import PlotGenerator

# Load constants from config.json
with open('config.json') as f:
    config = json.load(f)
    STOCK_TICKERS = config['STOCK_TICKERS']
    START_DATE = config['START_DATE']
    END_DATE = config['END_DATE']
    OPTIMIZATION_METHODS = config['OPTIMIZATION_METHODS']
    FORECAST_HORIZON = config['FORECAST_HORIZON']
    PLOT_TYPES = config['PLOT_TYPES']

# Create PortfolioOptimizer and ProphetForecaster objects
portfolio_optimizer = PortfolioOptimizer()
forecaster = ProphetForecaster()

# Download stock data from Yahoo Finance and store in /data folder
data_folder = './data'
for ticker in STOCK_TICKERS:
    stock_data = yf.download(ticker, start=START_DATE, end=END_DATE)
    stock_data.to_csv(f'{data_folder}/{ticker}/{ticker}_data.csv')

# Load stock data from /data folder
stock_data = {}
for ticker in STOCK_TICKERS:
    stock_data[ticker] = pd.read_csv(f'{data_folder}/{ticker}/{ticker}_data.csv', index_col='Date', parse_dates=True)

# Sidebar options
st.sidebar.title('Options')
ticker = st.sidebar.selectbox('Select Ticker', STOCK_TICKERS)
frequency = st.sidebar.selectbox('Frequency', ['Daily', 'Weekly', 'Monthly'])

# Get stock data for selected ticker and frequency
if frequency == 'Daily':
    data = stock_data[ticker]
elif frequency == 'Weekly':
    data = stock_data[ticker].resample('W').last()
elif frequency == 'Monthly':
    data = stock_data[ticker].resample('M').last()

# Calculate optimal portfolio weights and metrics for selected optimization method
optimization_method = st.sidebar.selectbox('Optimization Method', OPTIMIZATION_METHODS)
portfolio = portfolio_optimizer.optimize_portfolio(data, method=optimization_method)
st.sidebar.subheader('Portfolio Metrics')
st.sidebar.write(f'Expected Return: {portfolio.expected_return:.2f}')
st.sidebar.write(f'Volatility: {portfolio.volatility:.2f}')
st.sidebar.write(f'Sharpe Ratio: {portfolio.sharpe_ratio:.2f}')
st.sidebar.write(f'Sortino Ratio: {portfolio.sortino_ratio:.2f}')

# Make stock price forecast using ProphetForecaster
forecast_dates, forecast_prices = forecaster.make_prediction(data.index.strftime('%Y-%m-%d').tolist(), horizon=FORECAST_HORIZON)

# Generate plots
plot_generator = PlotGenerator()
plot_types = st.sidebar.multiselect('Select Plot Types', PLOT_TYPES, default=PLOT_TYPES)
for plot_type in plot_types:
    fig = plot_generator.generate_plot(data, portfolio, forecast_dates, forecast_prices, plot_type)
    st.plotly_chart(fig)

# Show stock data and portfolio weights in table
st.subheader('Stock Data')
st.write(data)
st.subheader('Portfolio Weights')
st.write(portfolio.weights)

if __name__ == '__main__':
    # Basic unit test for Streamlit app
    st.write('Hello, world!')
