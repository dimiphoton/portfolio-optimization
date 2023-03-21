import pandas as pd
import numpy as np
from prophet import Prophet
from typing import Tuple
import unittest

from models.base_model import BaseModel

class ProphetModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Fit the model with given data.

        :param data: Input data to train the model.
        """
        df = data.reset_index()
        df.columns = ['ds', 'y']
        self.model = Prophet()
        self.model.fit(df)

    def predict(self, periods: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict using the trained model.

        :param periods: Number of periods to forecast.
        :return: Tuple containing predicted returns and volatilities.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet.")

        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        predicted_returns = forecast['yhat'].values[-periods:]
        predicted_volatilities = forecast['yhat_upper'].values[-periods:] - forecast['yhat_lower'].values[-periods:]
        return predicted_returns, predicted_volatilities

# Test cases for ProphetModel
class TestProphetModel(unittest.TestCase):
    def setUp(self) -> None:
        self.prophet_model = ProphetModel()
        self.data = pd.DataFrame({'y': np.random.randn(100)}, index=pd.date_range(start='2020-01-01', periods=100))

    def test_fit(self) -> None:
        self.prophet_model.fit(self.data)
        self.assertIsNotNone(self.prophet_model.model)

    def test_predict(self) -> None:
        self.prophet_model.fit(self.data)
        returns, volatilities = self.prophet_model.predict(10)
        self.assertEqual(len(returns), 10)
        self.assertEqual(len(volatilities), 10)

def main() -> None:
    unittest.main()

if __name__ == "__main__":
    main()
