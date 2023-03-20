import matplotlib.pyplot as plt
import seaborn as sns
import riskfolio.PlotFunctions as rpplt
import riskfolio as rp
import pandas as pd

class Visualization:
    def __init__(self, config):
        self.config = config

    def plot_efficient_frontier(self, stock_data):
        returns = pd.DataFrame(stock_data).pct_change().dropna()
        cov_matrix = returns.cov()
        mu = returns.mean()

        ax = rpplt.plot_efficient_frontier(mu, cov_matrix, rm="MV", points=100, cmap="viridis")
        plt.title("Efficient Frontier")
        plt.show()

    def plot_donut_portfolio(self, weights):
        plt.pie(weights, labels=self.config['tickers'], autopct="%.1f%%", startangle=90, pctdistance=0.85)
        centre_circle = plt.Circle((0, 0), 0.70, fc="white")
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.axis("equal")
        plt.title("Portfolio Allocation")
        plt.tight_layout()
        plt.show()

    def plot_prediction_error(self, historical_portfolio, forecasted_portfolio, actual_portfolio):
        portfolios = pd.concat([historical_portfolio, forecasted_portfolio, actual_portfolio], axis=1)
        portfolios.columns = ["Historical", "Forecasted", "Actual"]
        error = (portfolios["Forecasted"] - portfolios["Actual"]).abs()

        sns.lineplot(data=error, palette="tab10")
        plt.title("Prediction Error")
        plt.xlabel("Time")
        plt.ylabel("Absolute Error")
        plt.show()

    def plot_forecast_confidence(self, future_prices):
        fig, axes = plt.subplots(len(self.config['tickers']), 1, figsize=(10, 20), sharex=True)
        for i, ticker in enumerate(self.config['tickers']):
            axes[i].plot(future_prices[ticker]['fcst'], label="Forecast")
            axes[i].fill_between(
                future_prices[ticker]['time'],
                future_prices[ticker]['fcst_lower'],
                future_prices[ticker]['fcst_upper'],
                alpha=0.2,
                label="Confidence Interval"
            )
            axes[i].set_title(ticker)
            axes[i].legend()
        plt.xlabel("Time")
        plt.tight_layout()
        plt.show()
