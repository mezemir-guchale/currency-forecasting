"""Time series feature engineering for forex data."""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple


def add_lag_features(
    df: pd.DataFrame,
    column: str = "close",
    lags: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add lagged features to the DataFrame.

    Args:
        df: Input DataFrame.
        column: Column to create lags from.
        lags: List of lag periods.

    Returns:
        DataFrame with lag columns added.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 7, 14, 21]

    result = df.copy()
    for lag in lags:
        result[f"{column}_lag_{lag}"] = result[column].shift(lag)

    return result


def add_rolling_features(
    df: pd.DataFrame,
    column: str = "close",
    windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Add rolling statistics (mean, std, min, max) as features.

    Args:
        df: Input DataFrame.
        column: Column to compute rolling stats from.
        windows: List of rolling window sizes.

    Returns:
        DataFrame with rolling feature columns added.
    """
    if windows is None:
        windows = [5, 10, 21, 50]

    result = df.copy()
    for w in windows:
        result[f"{column}_sma_{w}"] = result[column].rolling(window=w).mean()
        result[f"{column}_std_{w}"] = result[column].rolling(window=w).std()
        result[f"{column}_min_{w}"] = result[column].rolling(window=w).min()
        result[f"{column}_max_{w}"] = result[column].rolling(window=w).max()

    return result


def add_return_features(
    df: pd.DataFrame,
    column: str = "close",
) -> pd.DataFrame:
    """Add return-based features (daily return, log return, cumulative).

    Args:
        df: Input DataFrame.
        column: Column to compute returns from.

    Returns:
        DataFrame with return columns added.
    """
    result = df.copy()

    result["daily_return"] = result[column].pct_change()
    result["log_return"] = np.log(result[column] / result[column].shift(1))
    result["cum_return_5d"] = result[column].pct_change(periods=5)
    result["cum_return_21d"] = result[column].pct_change(periods=21)

    # Volatility features
    result["volatility_5d"] = result["daily_return"].rolling(5).std()
    result["volatility_21d"] = result["daily_return"].rolling(21).std()

    return result


def add_technical_indicators(
    df: pd.DataFrame,
    column: str = "close",
) -> pd.DataFrame:
    """Add common technical indicators.

    Args:
        df: Input DataFrame with OHLCV data.
        column: Price column name.

    Returns:
        DataFrame with technical indicator columns.
    """
    result = df.copy()

    # RSI (14-day)
    delta = result[column].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = result[column].ewm(span=12, adjust=False).mean()
    ema_26 = result[column].ewm(span=26, adjust=False).mean()
    result["macd"] = ema_12 - ema_26
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    # Bollinger Bands
    sma_20 = result[column].rolling(20).mean()
    std_20 = result[column].rolling(20).std()
    result["bb_upper"] = sma_20 + 2 * std_20
    result["bb_lower"] = sma_20 - 2 * std_20
    result["bb_width"] = (result["bb_upper"] - result["bb_lower"]) / sma_20

    # Average True Range
    if "high" in result.columns and "low" in result.columns:
        tr1 = result["high"] - result["low"]
        tr2 = abs(result["high"] - result[column].shift(1))
        tr3 = abs(result["low"] - result[column].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        result["atr_14"] = true_range.rolling(14).mean()

    return result


def decompose_series(
    series: pd.Series,
    period: int = 252,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Decompose time series into trend, seasonal, and residual components.

    Uses a simple moving average approach for decomposition.

    Args:
        series: Time series to decompose.
        period: Period for seasonal decomposition.

    Returns:
        Tuple of (trend, seasonal, residual) Series.
    """
    # Trend: moving average
    trend = series.rolling(window=period, center=True, min_periods=1).mean()

    # Detrended
    detrended = series - trend

    # Seasonal: average of detrended values at each position in the cycle
    seasonal = pd.Series(index=series.index, dtype=float)
    for i in range(period):
        mask = np.arange(len(series)) % period == i
        if mask.any():
            seasonal.iloc[mask] = detrended.iloc[mask].mean()

    # Residual
    residual = series - trend - seasonal

    return trend, seasonal, residual


def build_feature_matrix(
    df: pd.DataFrame,
    lag_periods: Optional[List[int]] = None,
    rolling_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """Build complete feature matrix for modeling.

    Args:
        df: Raw forex DataFrame.
        lag_periods: Lag periods to use.
        rolling_windows: Rolling window sizes.

    Returns:
        DataFrame with all features, NaN rows dropped.
    """
    result = df.copy()
    result = add_lag_features(result, lags=lag_periods)
    result = add_rolling_features(result, windows=rolling_windows)
    result = add_return_features(result)
    result = add_technical_indicators(result)

    return result
