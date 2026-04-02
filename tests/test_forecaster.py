"""Tests for forecasting models and evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_dataset import generate_forex_data
from src.models.forecaster import (
    ARIMAForecaster,
    EMAForecaster,
    ExpSmoothingForecaster,
    SMAForecaster,
    run_all_models,
)
from src.evaluation.metrics import (
    compare_models,
    directional_accuracy,
    evaluate_forecast,
    mae,
    mape,
    rmse,
)


@pytest.fixture
def train_series():
    """Generate a training time series."""
    data = generate_forex_data(pairs=["EUR/USD"], years=2, seed=42)
    return data["EUR/USD"]["close"]


@pytest.fixture
def short_series():
    """A short known series for metric verification."""
    return pd.Series([1.0, 1.1, 1.05, 1.15, 1.10, 1.20, 1.15, 1.25, 1.20, 1.30])


class TestSMAForecaster:
    def test_forecast_length(self, train_series):
        """Test that forecast has correct length."""
        model = SMAForecaster(window=21)
        model.fit(train_series)
        pred = model.forecast(10)
        assert len(pred) == 10

    def test_forecast_constant(self, train_series):
        """Test that SMA forecast is constant."""
        model = SMAForecaster(window=21)
        model.fit(train_series)
        pred = model.forecast(5)
        assert np.allclose(pred, pred[0])

    def test_unfitted_raises(self):
        """Test that forecasting without fitting raises error."""
        model = SMAForecaster()
        with pytest.raises(ValueError):
            model.forecast(5)


class TestEMAForecaster:
    def test_forecast_length(self, train_series):
        """Test EMA forecast length."""
        model = EMAForecaster(span=21)
        model.fit(train_series)
        pred = model.forecast(10)
        assert len(pred) == 10

    def test_forecast_constant(self, train_series):
        """Test that EMA forecast is constant."""
        model = EMAForecaster(span=21)
        model.fit(train_series)
        pred = model.forecast(5)
        assert np.allclose(pred, pred[0])


class TestARIMAForecaster:
    def test_forecast_length(self, train_series):
        """Test ARIMA forecast length."""
        model = ARIMAForecaster(order=(1, 1, 0))
        model.fit(train_series)
        pred = model.forecast(10)
        assert len(pred) == 10

    def test_forecast_values_reasonable(self, train_series):
        """Test that ARIMA forecasts are within reasonable range."""
        model = ARIMAForecaster(order=(1, 1, 0))
        model.fit(train_series)
        pred = model.forecast(10)
        last_val = train_series.iloc[-1]
        # Forecasts should be within 20% of last value
        assert np.all(np.abs(pred - last_val) / last_val < 0.20)


class TestExpSmoothingForecaster:
    def test_forecast_length(self, train_series):
        """Test ExpSmoothing forecast length."""
        model = ExpSmoothingForecaster(trend="add")
        model.fit(train_series)
        pred = model.forecast(10)
        assert len(pred) == 10


class TestRunAllModels:
    def test_all_models_return_forecasts(self, train_series):
        """Test that run_all_models returns forecasts for all models."""
        forecasts = run_all_models(train_series, horizon=10)
        assert "SMA" in forecasts
        assert "EMA" in forecasts
        assert "ARIMA" in forecasts
        assert "ExpSmoothing" in forecasts
        for name, pred in forecasts.items():
            assert len(pred) == 10, f"{name} forecast length mismatch"


class TestMetrics:
    def test_mae_known_values(self):
        """Test MAE with known values."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.1, 2.2, 2.7])
        result = mae(actual, predicted)
        expected = np.mean([0.1, 0.2, 0.3])
        assert abs(result - expected) < 1e-10

    def test_rmse_known_values(self):
        """Test RMSE with known values."""
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.0, 2.0, 3.0])  # Perfect prediction
        assert rmse(actual, predicted) == 0.0

    def test_mape_known_values(self):
        """Test MAPE with known values."""
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 180.0])
        result = mape(actual, predicted)
        expected = (10 / 100 + 20 / 200) / 2 * 100
        assert abs(result - expected) < 1e-10

    def test_directional_accuracy(self):
        """Test directional accuracy."""
        actual = np.array([1.0, 1.1, 1.0, 1.2])  # up, down, up
        predicted = np.array([1.0, 1.05, 1.1, 1.15])  # up, up, up
        result = directional_accuracy(actual, predicted)
        # Actual dirs: up, down, up; Predicted dirs: up, up, up
        # Match: 1st (up==up), 3rd (up==up); Mismatch: 2nd (down!=up)
        assert abs(result - 66.67) < 1.0

    def test_evaluate_forecast_keys(self):
        """Test that evaluate_forecast returns all metrics."""
        actual = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        predicted = np.array([1.0, 1.05, 1.15, 1.25, 1.35])
        result = evaluate_forecast(actual, predicted)
        assert "MAE" in result
        assert "RMSE" in result
        assert "MAPE" in result
        assert "Directional_Accuracy" in result

    def test_compare_models_shape(self):
        """Test compare_models returns correct shape."""
        actual = np.array([1.0, 1.1, 1.2, 1.3, 1.4])
        forecasts = {
            "Model_A": np.array([1.0, 1.05, 1.15, 1.25, 1.35]),
            "Model_B": np.array([1.0, 1.10, 1.20, 1.30, 1.40]),
        }
        result = compare_models(actual, forecasts)
        assert result.shape[0] == 2
        assert "MAE" in result.columns
