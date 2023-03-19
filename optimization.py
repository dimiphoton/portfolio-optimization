import pandas as pd
import riskfolio as rp

class PortfolioOptimizer:
    def __init__(self, config):
        self.config = config

    def calculate_optimal_weights(self, stock_data):
        data = pd.DataFrame(stock_data)
        returns = data.pct_change().dropna()

        cov_matrix = returns.cov()
        mu = returns.mean()

        # Calculate the mean-variance optimized portfolio
        opt_model = rp.MeanVarianceOptimization(mu, cov_matrix)
        weights = opt_model.optimize()
        return weights
