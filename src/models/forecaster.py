"""Forecasting models for currency exchange rates.

Implements ARIMA, Simple Moving Average, and Exponential Smoothing
forecasters with a consistent interface.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class SMAForecaster:
    """Simple Moving Average baseline forecaster."""

    def __init__(self, window: int = 21):
        """Initialize SMA forecaster.

        Args:
            window: Moving average window size.
        """
        self.window = window
        self.name = f"SMA({window})"
        self._last_values = None

    def fit(self, series: pd.Series) -> "SMAForecaster":
        """Fit the model by storing the last window values.

        Args:
            series: Training time series.

        Returns:
            Self.
        """
        self._last_values = series.iloc[-self.window:].values.copy()
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts.

        The SMA forecast is the mean of the last window values,
        extended forward for the forecast horizon.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of forecasted values.
        """
        if self._last_values is None:
            raise ValueError("Model not fitted. Call fit() first.")

        sma_value = np.mean(self._last_values)
        return np.full(horizon, sma_value)


class EMAForecaster:
    """Exponential Moving Average forecaster."""

    def __init__(self, span: int = 21):
        """Initialize EMA forecaster.

        Args:
            span: EMA span parameter.
        """
        self.span = span
        self.name = f"EMA({span})"
        self._last_ema = None

    def fit(self, series: pd.Series) -> "EMAForecaster":
        """Fit the model by computing EMA of training data.

        Args:
            series: Training time series.

        Returns:
            Self.
        """
        ema = series.ewm(span=self.span, adjust=False).mean()
        self._last_ema = float(ema.iloc[-1])
        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts using last EMA value.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of forecasted values.
        """
        if self._last_ema is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return np.full(horizon, self._last_ema)


class ARIMAForecaster:
    """ARIMA model wrapper for forex forecasting."""

    def __init__(self, order: Tuple[int, int, int] = (2, 1, 2)):
        """Initialize ARIMA forecaster.

        Args:
            order: ARIMA (p, d, q) order.
        """
        self.order = order
        self.name = f"ARIMA{order}"
        self._model_fit = None
        self._train_series = None

    def fit(self, series: pd.Series) -> "ARIMAForecaster":
        """Fit ARIMA model to training data.

        Args:
            series: Training time series.

        Returns:
            Self.
        """
        self._train_series = series.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ARIMA(series.values, order=self.order)
                self._model_fit = model.fit()
            except Exception:
                # Fallback to simpler model if fitting fails
                try:
                    model = ARIMA(series.values, order=(1, 1, 0))
                    self._model_fit = model.fit()
                except Exception:
                    self._model_fit = None

        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate ARIMA forecasts.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of forecasted values.
        """
        if self._model_fit is None:
            # Fallback: return last value
            if self._train_series is not None:
                return np.full(horizon, self._train_series.iloc[-1])
            raise ValueError("Model not fitted. Call fit() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model_fit.forecast(steps=horizon)

        return np.array(forecast)


class ExpSmoothingForecaster:
    """Holt-Winters Exponential Smoothing forecaster."""

    def __init__(
        self,
        trend: str = "add",
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ):
        """Initialize Exponential Smoothing forecaster.

        Args:
            trend: Trend component type ('add', 'mul', or None).
            seasonal: Seasonal component type.
            seasonal_periods: Number of periods in seasonal cycle.
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.name = f"ExpSmoothing(trend={trend})"
        self._model_fit = None
        self._train_series = None

    def fit(self, series: pd.Series) -> "ExpSmoothingForecaster":
        """Fit the exponential smoothing model.

        Args:
            series: Training time series.

        Returns:
            Self.
        """
        self._train_series = series.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = ExponentialSmoothing(
                    series.values,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    seasonal_periods=self.seasonal_periods,
                )
                self._model_fit = model.fit(optimized=True)
            except Exception:
                # Fallback to simple exponential smoothing
                try:
                    model = ExponentialSmoothing(
                        series.values,
                        trend=None,
                        seasonal=None,
                    )
                    self._model_fit = model.fit(optimized=True)
                except Exception:
                    self._model_fit = None

        return self

    def forecast(self, horizon: int) -> np.ndarray:
        """Generate forecasts.

        Args:
            horizon: Number of periods to forecast.

        Returns:
            Array of forecasted values.
        """
        if self._model_fit is None:
            if self._train_series is not None:
                return np.full(horizon, self._train_series.iloc[-1])
            raise ValueError("Model not fitted. Call fit() first.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self._model_fit.forecast(horizon)

        return np.array(forecast)


def run_all_models(
    train_series: pd.Series,
    horizon: int = 30,
    arima_order: Tuple[int, int, int] = (2, 1, 2),
    sma_window: int = 21,
    ema_span: int = 21,
) -> Dict[str, np.ndarray]:
    """Run all forecasting models and return predictions.

    Args:
        train_series: Training time series.
        horizon: Forecast horizon.
        arima_order: ARIMA order.
        sma_window: SMA window size.
        ema_span: EMA span.

    Returns:
        Dictionary mapping model name to forecast array.
    """
    models = {
        "SMA": SMAForecaster(window=sma_window),
        "EMA": EMAForecaster(span=ema_span),
        "ARIMA": ARIMAForecaster(order=arima_order),
        "ExpSmoothing": ExpSmoothingForecaster(trend="add"),
    }

    forecasts = {}
    for name, model in models.items():
        model.fit(train_series)
        forecasts[name] = model.forecast(horizon)

    return forecasts
