import numpy as np
from typing import List, Tuple

from optimization_strategies import OptimizationStrategies

class PortfolioOptimizer:
    def __init__(self, returns: np.ndarray, cov_matrix: np.ndarray, n_portfolios: int = 10000):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.n_assets = returns.shape[0]
        self.n_portfolios = n_portfolios

    def generate_random_portfolios(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights = np.zeros((self.n_portfolios, self.n_assets))
        portfolio_returns = np.zeros(self.n_portfolios)
        portfolio_volatility = np.zeros(self.n_portfolios)

        for i in range(self.n_portfolios):
            w = np.random.random(self.n_assets)
            w /= np.sum(w)
            weights[i,:] = w
            portfolio_returns[i] = np.sum(self.returns * w)
            portfolio_volatility[i] = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))

        return weights, portfolio_returns, portfolio_volatility

    def calculate_efficient_frontier(self, risk_free_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        weights, returns, volatility = self.generate_random_portfolios()
        sharpe_ratio = (returns - risk_free_rate) / volatility
        sortino_ratio = (returns - risk_free_rate) / np.sqrt(np.sum(np.where(returns < risk_free_rate, (returns - risk_free_rate) ** 2, 0)))

        opt_weights, opt_returns, opt_volatility, opt_sharpe, opt_sortino = OptimizationStrategies.mean_variance_optimization(self.returns, self.cov_matrix, risk_free_rate)

        return weights, returns, volatility, sharpe_ratio, sortino_ratio, opt_weights, opt_returns, opt_volatility, opt_sharpe, opt_sortino

    def calculate_monte_carlo_simulations(self, n_simulations: int, risk_free_rate: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        weights, returns, volatility = self.generate_random_portfolios()
        opt_weights, opt_returns, opt_volatility, _, _ = OptimizationStrategies.mean_variance_optimization(self.returns, self.cov_matrix, risk_free_rate)
        sim_returns = np.zeros(n_simulations)
        sim_volatility = np.zeros(n_simulations)

        for i in range(n_simulations):
            w = weights[np.random.randint(weights.shape[0])]
            sim_returns[i] = np.sum(self.returns * w)
            sim_volatility[i] = np.sqrt(np.dot(w.T, np.dot(self.cov_matrix, w)))

        return sim_returns, sim_volatility, opt_returns, opt_volatility
