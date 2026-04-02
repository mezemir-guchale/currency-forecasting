"""Tests for time series feature engineering."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.generate_dataset import generate_forex_data
from src.features.time_series import (
    add_lag_features,
    add_return_features,
    add_rolling_features,
    add_technical_indicators,
    build_feature_matrix,
    decompose_series,
)


@pytest.fixture
def sample_df():
    """Generate sample forex data for one pair."""
    data = generate_forex_data(pairs=["EUR/USD"], years=1, seed=42)
    return data["EUR/USD"]


class TestLagFeatures:
    def test_lag_columns_created(self, sample_df):
        """Test that lag columns are created."""
        result = add_lag_features(sample_df, lags=[1, 5, 10])
        assert "close_lag_1" in result.columns
        assert "close_lag_5" in result.columns
        assert "close_lag_10" in result.columns

    def test_lag_values_correct(self, sample_df):
        """Test that lag values are shifted correctly."""
        result = add_lag_features(sample_df, lags=[1])
        # Lag 1 should equal the previous close
        assert result["close_lag_1"].iloc[5] == result["close"].iloc[4]

    def test_lag_first_value_nan(self, sample_df):
        """Test that first values of lag columns are NaN."""
        result = add_lag_features(sample_df, lags=[3])
        assert pd.isna(result["close_lag_3"].iloc[0])
        assert pd.isna(result["close_lag_3"].iloc[2])
        assert not pd.isna(result["close_lag_3"].iloc[3])


class TestRollingFeatures:
    def test_rolling_columns_created(self, sample_df):
        """Test that rolling statistic columns are created."""
        result = add_rolling_features(sample_df, windows=[5])
        assert "close_sma_5" in result.columns
        assert "close_std_5" in result.columns
        assert "close_min_5" in result.columns
        assert "close_max_5" in result.columns

    def test_sma_calculation(self, sample_df):
        """Test SMA values are correct."""
        result = add_rolling_features(sample_df, windows=[5])
        # Manual check: SMA at index 4 should be mean of first 5 values
        expected = sample_df["close"].iloc[:5].mean()
        assert abs(result["close_sma_5"].iloc[4] - expected) < 1e-10


class TestReturnFeatures:
    def test_return_columns_created(self, sample_df):
        """Test that return columns are created."""
        result = add_return_features(sample_df)
        assert "daily_return" in result.columns
        assert "log_return" in result.columns
        assert "volatility_5d" in result.columns

    def test_daily_return_values(self, sample_df):
        """Test daily return calculation."""
        result = add_return_features(sample_df)
        expected = (sample_df["close"].iloc[1] - sample_df["close"].iloc[0]) / sample_df["close"].iloc[0]
        assert abs(result["daily_return"].iloc[1] - expected) < 1e-10


class TestTechnicalIndicators:
    def test_rsi_range(self, sample_df):
        """Test that RSI is bounded between 0 and 100."""
        result = add_technical_indicators(sample_df)
        rsi = result["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all()

    def test_macd_columns(self, sample_df):
        """Test MACD columns are created."""
        result = add_technical_indicators(sample_df)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_bollinger_bands(self, sample_df):
        """Test Bollinger Bands relationships."""
        result = add_technical_indicators(sample_df)
        valid = result.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()


class TestDecomposition:
    def test_decomposition_output_lengths(self, sample_df):
        """Test that decomposition outputs match input length."""
        trend, seasonal, residual = decompose_series(sample_df["close"], period=21)
        assert len(trend) == len(sample_df)
        assert len(seasonal) == len(sample_df)
        assert len(residual) == len(sample_df)


class TestFeatureMatrix:
    def test_build_feature_matrix(self, sample_df):
        """Test complete feature matrix building."""
        result = build_feature_matrix(sample_df, lag_periods=[1, 5], rolling_windows=[5])
        # Should have many more columns than original
        assert len(result.columns) > len(sample_df.columns)
