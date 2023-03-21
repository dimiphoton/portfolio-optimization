import pandas as pd
import numpy as np
import unittest
from typing import List, Optional
#import riskfolio-lib.Portfolio as pf
import riskfolio as rp


class PortfolioOptimizer:
    def __init__(self, data: pd.DataFrame, stock_list: Optional[List[str]] = None):
        self.data = data
        self.stock_list = stock_list if stock_list else data.columns
        self.portfolio = rp.Portfolio(returns=self.data[self.stock_list])

    def calculate_metrics(self) -> None:
        """
        Calculate portfolio metrics such as historical return, volatility, Sharpe, and Sortino ratios.
        """
        self.portfolio.assets_stats(method_mu='hist', method_cov='hist')

    def optimize(self) -> None:
        """
        Optimize the portfolio using mean-variance optimization.
        """
        self.weights = self.portfolio.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0)

    def get_metrics(self) -> pd.DataFrame:
        """
        Get a DataFrame containing the calculated metrics.

        :return: A DataFrame with the calculated metrics.
        """
        return self.portfolio.assets[['mu', 'sigma', 'sharpe', 'sortino']]
    
    def get_efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)


if __name__ == "__main__":
    pass
