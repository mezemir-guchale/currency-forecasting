"""Evaluation metrics for time series forecasting."""

import numpy as np
import pandas as pd
from typing import Dict, Optional


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        MAE value.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        RMSE value.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Args:
        actual: Actual values (must not contain zeros).
        predicted: Predicted values.

    Returns:
        MAPE value as a percentage.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    # Avoid division by zero
    mask = actual != 0
    if not mask.any():
        return float("inf")

    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Directional Accuracy - percentage of correct direction predictions.

    Compares whether the predicted direction of change matches
    the actual direction of change.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        Directional accuracy as a percentage (0-100).
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    if len(actual) < 2:
        return 0.0

    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0

    correct = np.sum(actual_direction == predicted_direction)
    total = len(actual_direction)

    return float(correct / total * 100) if total > 0 else 0.0


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """Compute all evaluation metrics for a forecast.

    Args:
        actual: Actual values.
        predicted: Predicted values.

    Returns:
        Dictionary of metric name to value.
    """
    n = min(len(actual), len(predicted))
    actual = np.asarray(actual)[:n]
    predicted = np.asarray(predicted)[:n]

    return {
        "MAE": round(mae(actual, predicted), 6),
        "RMSE": round(rmse(actual, predicted), 6),
        "MAPE": round(mape(actual, predicted), 4),
        "Directional_Accuracy": round(directional_accuracy(actual, predicted), 2),
    }


def compare_models(
    actual: np.ndarray,
    forecasts: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Compare multiple forecasting models.

    Args:
        actual: Actual values.
        forecasts: Dictionary mapping model name to forecast array.

    Returns:
        DataFrame with metrics for each model.
    """
    results = []
    for model_name, predicted in forecasts.items():
        metrics = evaluate_forecast(actual, predicted)
        metrics["Model"] = model_name
        results.append(metrics)

    df = pd.DataFrame(results)
    df = df.set_index("Model")
    return df
