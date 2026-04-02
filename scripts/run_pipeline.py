"""Main pipeline script for Currency Exchange Rate Forecasting."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from src.utils.config import load_config, ensure_directories
from src.utils.logger import setup_logger
from src.data.generate_dataset import generate_forex_data, save_datasets
from src.data.loader import load_pair_data, train_test_split_ts
from src.features.time_series import (
    build_feature_matrix, decompose_series,
)
from src.models.forecaster import run_all_models
from src.evaluation.metrics import evaluate_forecast, compare_models
from src.visualization.charts import (
    plot_forecast, plot_residuals, plot_decomposition,
    plot_model_comparison,
)


def main():
    """Run the full currency forecasting pipeline."""
    config = load_config()
    ensure_directories(config)
    logger = setup_logger("forex_forecast", config, "pipeline.log")

    logger.info("=" * 60)
    logger.info("Currency Exchange Rate Forecasting Pipeline")
    logger.info("=" * 60)

    # Step 1: Generate data
    logger.info("Step 1: Generating synthetic forex data...")
    pairs = config["data"]["currency_pairs"]
    data = generate_forex_data(
        pairs=pairs,
        years=config["data"]["years"],
        seed=config["data"]["seed"],
    )
    raw_dir = str(project_root / config["data"]["raw_dir"])
    paths = save_datasets(data, raw_dir)
    logger.info(f"Generated data for {len(data)} pairs, saved {len(paths)} files")

    fig_dir = project_root / config["visualization"]["output_dir"]
    fig_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = project_root / config["data"]["processed_dir"]

    all_results = []

    for pair in pairs:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {pair}")
        logger.info(f"{'='*50}")

        # Step 2: Load data
        df = load_pair_data(pair, raw_dir)
        logger.info(f"  Loaded {len(df)} trading days")

        # Step 3: Feature engineering
        logger.info("  Building feature matrix...")
        featured = build_feature_matrix(
            df,
            lag_periods=config["features"]["lag_periods"],
            rolling_windows=config["features"]["rolling_windows"],
        )

        # Step 4: Decomposition
        logger.info("  Decomposing time series...")
        trend, seasonal, residual = decompose_series(
            df["close"],
            period=min(config["features"]["decomposition_period"], len(df) // 3),
        )

        pair_slug = pair.replace("/", "_").lower()
        fig = plot_decomposition(
            df["close"], trend, seasonal, residual, pair,
            str(fig_dir / f"decomposition_{pair_slug}.png"),
        )
        plt_close(fig)

        # Step 5: Train/test split
        train_ratio = config["models"]["train_ratio"]
        train_df, test_df = train_test_split_ts(df, train_ratio)
        train_series = train_df["close"]
        test_series = test_df["close"]

        horizon = min(config["models"]["forecast_horizon"], len(test_series))
        logger.info(f"  Train: {len(train_df)}, Test: {len(test_df)}, Horizon: {horizon}")

        # Step 6: Run models
        logger.info("  Running forecasting models...")
        forecasts = run_all_models(
            train_series,
            horizon=horizon,
            arima_order=tuple(config["models"]["arima"]["order"]),
            sma_window=config["models"]["sma_window"],
            ema_span=config["models"]["ema_span"],
        )

        # Step 7: Evaluate
        logger.info("  Evaluating forecasts...")
        actual = test_series.iloc[:horizon].values
        comparison = compare_models(actual, forecasts)

        logger.info(f"\n  Model Comparison for {pair}:")
        logger.info(f"\n{comparison.to_string()}")

        # Find best model
        best_model = comparison["MAE"].idxmin()
        logger.info(f"\n  Best model (by MAE): {best_model}")

        # Store results
        for model_name in forecasts:
            metrics = evaluate_forecast(actual, forecasts[model_name])
            metrics["pair"] = pair
            metrics["model"] = model_name
            all_results.append(metrics)

        # Step 8: Visualizations
        logger.info("  Generating visualizations...")

        fig = plot_forecast(
            train_series, test_series.iloc[:horizon], forecasts, pair,
            str(fig_dir / f"forecast_{pair_slug}.png"),
        )
        plt_close(fig)

        fig = plot_model_comparison(
            comparison, pair,
            str(fig_dir / f"comparison_{pair_slug}.png"),
        )
        plt_close(fig)

        # Residual analysis for best model
        fig = plot_residuals(
            actual, forecasts[best_model], best_model, pair,
            str(fig_dir / f"residuals_{pair_slug}.png"),
        )
        plt_close(fig)

    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(processed_dir / "forecast_results.csv", index=False)
    logger.info(f"\nSaved results to {processed_dir / 'forecast_results.csv'}")

    # Overall summary
    logger.info("\n" + "=" * 60)
    logger.info("Overall Summary")
    logger.info("=" * 60)
    summary = results_df.groupby("model")[["MAE", "RMSE", "MAPE"]].mean()
    logger.info(f"\nAverage Metrics Across All Pairs:")
    logger.info(f"\n{summary.to_string()}")

    best_overall = summary["MAE"].idxmin()
    logger.info(f"\nBest overall model (avg MAE): {best_overall}")

    logger.info("\n" + "=" * 60)
    logger.info("Pipeline completed successfully!")
    logger.info("=" * 60)


def plt_close(fig):
    """Close a matplotlib figure."""
    import matplotlib.pyplot as plt
    plt.close(fig)


if __name__ == "__main__":
    main()
