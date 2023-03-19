import numpy as np
import pandas as pd

class PortfolioPerformance:
    def __init__(self, config):
        self.config = config

    def calculate_portfolio(self, prices, weights):
        returns = prices.pct_change().dropna()
        portfolio_returns = returns.dot(weights)
        return portfolio_returns

    def annualized_return(self, portfolio_returns):
        avg_daily_return = np.mean(portfolio_returns)
        annualized_return = (1 + avg_daily_return) ** 252 - 1
        return annualized_return

    def annualized_volatility(self, portfolio_returns):
        daily_volatility = np.std(portfolio_returns)
        annualized_volatility = daily_volatility * np.sqrt(252)
        return annualized_volatility

    def sharpe_ratio(self, portfolio_returns, risk_free_rate=0.02):
        annualized_ret = self.annualized_return(portfolio_returns)
        annualized_vol = self.annualized_volatility(portfolio_returns)
        sharpe_ratio = (annualized_ret - risk_free_rate) / annualized_vol
        return sharpe_ratio

    def sortino_ratio(self, portfolio_returns, risk_free_rate=0.02, target_return=0):
        annualized_ret = self.annualized_return(portfolio_returns)
        returns_below_target = portfolio_returns[portfolio_returns < target_return]
        downside_deviation = np.std(returns_below_target)
        annualized_downside_deviation = downside_deviation * np.sqrt(252)
        sortino_ratio = (annualized_ret - risk_free_rate) / annualized_downside_deviation
        return sortino_ratio

    def calculate_metrics(self, portfolio_returns):
        metrics = {
            'Annualized Return': self.annualized_return(portfolio_returns),
            'Annualized Volatility': self.annualized_volatility(portfolio_returns),
            'Sharpe Ratio': self.sharpe_ratio(portfolio_returns),
            'Sortino Ratio': self.sortino_ratio(portfolio_returns),
        }
        return metrics
