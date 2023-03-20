import json
from typing import List
from User import User
from Visualization import Visualization

def main():
# Load the configuration file
    with open('config.json', 'r') as f:
        config = json.load(f)
    # Create a list of stock symbols
    stock_symbols = config['stocks']

    # Create a user account
    user = User('John Smith', 'john.smith@example.com', config['initial_balance'])

    # Buy some stocks
    for symbol in stock_symbols:
        user.buy_stock(symbol, 10, config['commission'])

        # Update the data
        user.update_data()

        # Create a visualization object
        viz = Visualization(stock_symbols)

        # Plot the historical stock prices and returns
        viz.plot_stock_prices(config['start_date'], config['end_date'])
        viz.plot_stock_returns(config['start_date'], config['end_date'])

        # Plot the current portfolio allocation and returns
        viz.plot_portfolio_allocation()
        viz.plot_portfolio_returns()

        # Train forecasting models and plot forecasts
        for symbol in stock_symbols:
            # Train the forecasting model
            model = viz.forecasting_models[symbol]
            model.train_model()
    
            # Plot the forecast
            viz.plot_forecast(symbol, config['forecast_start_date'], config['forecast_end_date'])

