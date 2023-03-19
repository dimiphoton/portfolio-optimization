import json
from data_handler import DataHandler
from optimization import PortfolioOptimizer
from modeling import TimeSeriesModeler
from performance_metrics import PortfolioPerformance
from visualization import Visualization

def main():
    # Load the configuration
    with open("config.json") as f:
        config = json.load(f)

    # Create instances of the classes
    data_handler = DataHandler(config)
    optimizer = PortfolioOptimizer(config)
    modeler = TimeSeriesModeler(config)
    performance = PortfolioPerformance(config)
    viz = Visualization(config)

    # Download stock data, save it, and load it
    stock_data = {}
    for ticker in config['tickers']:
        data = data_handler.download_stock_data(ticker)
        data_handler.save_stock_data(ticker, data)
        stock_data[ticker] = data_handler.load_stock_data(ticker)

    # Calculate the mean-variance optimized portfolio
    optimal_weights = optimizer.calculate_optimal_weights(stock_data)

    # Train models and forecast future prices
    future_prices = {}
    for ticker in config['tickers']:
        # Train the model and save it
        model = modeler.train_model(stock_data[ticker])
        data_handler.save_trained_model(ticker, model)

        # Load the trained model and forecast future prices
        loaded_model = data_handler.load_trained_model(ticker)
        future_prices[ticker] = modeler.forecast_prices(loaded_model)

    # Find the optimal portfolio over the new time range
    future_optimal_weights = optimizer.calculate_optimal_weights(future_prices)

    # Calculate performance metrics for the portfolios
    historical_portfolio = performance.calculate_portfolio(stock_data, optimal_weights)
    forecasted_portfolio = performance.calculate_portfolio(future_prices, optimal_weights)
    actual_portfolio = performance.calculate_portfolio(future_prices, future_optimal_weights)

    # Create visualizations
    viz.plot_efficient_frontier(stock_data)
    viz.plot_donut_portfolio(optimal_weights)
    viz.plot_prediction_error(historical_portfolio, forecasted_portfolio, actual_portfolio)
    viz.plot_forecast_confidence(future_prices)

if __name__ == "__main__":
    main()
