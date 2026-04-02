"""Visualization module for Currency Forecasting."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecasts: Dict[str, np.ndarray],
    pair: str,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Plot actual prices with forecast overlay.

    Args:
        train: Training data Series.
        test: Test data Series.
        forecasts: Dict of model name to forecast arrays.
        pair: Currency pair name.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot training data (last 100 points for clarity)
    train_tail = train.iloc[-100:]
    ax.plot(range(len(train_tail)), train_tail.values,
            color="#2c3e50", linewidth=1.5, label="Training", alpha=0.7)

    # Plot test data
    test_x = range(len(train_tail), len(train_tail) + len(test))
    ax.plot(test_x, test.values, color="#2c3e50", linewidth=2, label="Actual")

    # Plot forecasts
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    for i, (name, pred) in enumerate(forecasts.items()):
        n = min(len(pred), len(test))
        forecast_x = range(len(train_tail), len(train_tail) + n)
        ax.plot(forecast_x, pred[:n], linestyle="--", linewidth=2,
                color=colors[i % len(colors)], label=name)

    # Vertical line at forecast start
    ax.axvline(x=len(train_tail), color="gray", linestyle=":", alpha=0.7,
               label="Forecast Start")

    ax.set_xlabel("Trading Days", fontsize=12)
    ax.set_ylabel("Exchange Rate", fontsize=12)
    ax.set_title(f"{pair} Forecast Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residuals(
    actual: np.ndarray,
    predicted: np.ndarray,
    model_name: str,
    pair: str,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 8),
) -> plt.Figure:
    """Plot residual analysis for a forecast model.

    Args:
        actual: Actual values.
        predicted: Predicted values.
        model_name: Name of the model.
        pair: Currency pair name.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    n = min(len(actual), len(predicted))
    residuals = np.asarray(actual)[:n] - np.asarray(predicted)[:n]

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Residuals over time
    axes[0, 0].plot(residuals, color="#2980b9", linewidth=1)
    axes[0, 0].axhline(y=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 0].set_title("Residuals Over Time", fontweight="bold")
    axes[0, 0].set_xlabel("Period")
    axes[0, 0].set_ylabel("Residual")

    # Histogram
    axes[0, 1].hist(residuals, bins=min(20, max(5, n // 3)),
                     color="#3498db", edgecolor="white", alpha=0.8)
    axes[0, 1].axvline(x=0, color="red", linestyle="--", alpha=0.7)
    axes[0, 1].set_title("Residual Distribution", fontweight="bold")
    axes[0, 1].set_xlabel("Residual")

    # Actual vs Predicted
    axes[1, 0].scatter(actual[:n], predicted[:n], alpha=0.6, color="#2ecc71", s=20)
    min_val = min(np.min(actual[:n]), np.min(predicted[:n]))
    max_val = max(np.max(actual[:n]), np.max(predicted[:n]))
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.7)
    axes[1, 0].set_title("Actual vs Predicted", fontweight="bold")
    axes[1, 0].set_xlabel("Actual")
    axes[1, 0].set_ylabel("Predicted")

    # Autocorrelation of residuals
    if n > 2:
        lags = min(20, n - 1)
        acf_vals = [1.0]
        for lag in range(1, lags + 1):
            if lag < n:
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                acf_vals.append(corr if not np.isnan(corr) else 0)
        axes[1, 1].bar(range(len(acf_vals)), acf_vals, color="#9b59b6", alpha=0.8)
        axes[1, 1].axhline(y=0, color="gray", linestyle="-")
        ci = 1.96 / np.sqrt(n)
        axes[1, 1].axhline(y=ci, color="red", linestyle="--", alpha=0.5)
        axes[1, 1].axhline(y=-ci, color="red", linestyle="--", alpha=0.5)
    axes[1, 1].set_title("Residual ACF", fontweight="bold")
    axes[1, 1].set_xlabel("Lag")

    fig.suptitle(f"{model_name} Residual Analysis - {pair}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_decomposition(
    series: pd.Series,
    trend: pd.Series,
    seasonal: pd.Series,
    residual: pd.Series,
    pair: str,
    output_path: Optional[str] = None,
    figsize: tuple = (14, 12),
) -> plt.Figure:
    """Plot time series decomposition.

    Args:
        series: Original time series.
        trend: Trend component.
        seasonal: Seasonal component.
        residual: Residual component.
        pair: Currency pair name.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)

    axes[0].plot(series.values, color="#2c3e50", linewidth=1)
    axes[0].set_title("Original", fontweight="bold")
    axes[0].grid(alpha=0.3)

    axes[1].plot(trend.values, color="#e74c3c", linewidth=1.5)
    axes[1].set_title("Trend", fontweight="bold")
    axes[1].grid(alpha=0.3)

    axes[2].plot(seasonal.values, color="#2ecc71", linewidth=1)
    axes[2].set_title("Seasonal", fontweight="bold")
    axes[2].grid(alpha=0.3)

    axes[3].plot(residual.values, color="#3498db", linewidth=1)
    axes[3].set_title("Residual", fontweight="bold")
    axes[3].set_xlabel("Trading Days")
    axes[3].grid(alpha=0.3)

    fig.suptitle(f"Time Series Decomposition: {pair}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_comparison(
    comparison_df: pd.DataFrame,
    pair: str,
    output_path: Optional[str] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Plot model comparison bar chart.

    Args:
        comparison_df: DataFrame with metrics for each model.
        pair: Currency pair name.
        output_path: Path to save figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    metrics_to_plot = [c for c in ["MAE", "RMSE", "MAPE"] if c in comparison_df.columns]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=figsize)
    if len(metrics_to_plot) == 1:
        axes = [axes]

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    for i, metric in enumerate(metrics_to_plot):
        vals = comparison_df[metric]
        bars = axes[i].bar(
            range(len(vals)),
            vals.values,
            color=colors[:len(vals)],
            edgecolor="white",
        )
        axes[i].set_xticks(range(len(vals)))
        axes[i].set_xticklabels(vals.index, rotation=30, ha="right", fontsize=9)
        axes[i].set_title(metric, fontsize=12, fontweight="bold")
        axes[i].grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, vals.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Model Comparison: {pair}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
