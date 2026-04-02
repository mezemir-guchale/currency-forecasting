"""Generate synthetic forex data for currency pairs."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.config import get_project_root


# Base rates and volatility profiles for currency pairs
PAIR_PROFILES = {
    "EUR/USD": {"base_rate": 1.10, "volatility": 0.0045, "drift": 0.00002},
    "GBP/USD": {"base_rate": 1.27, "volatility": 0.0055, "drift": -0.00001},
    "USD/JPY": {"base_rate": 140.0, "volatility": 0.0050, "drift": 0.00003},
    "AUD/USD": {"base_rate": 0.67, "volatility": 0.0060, "drift": -0.00002},
    "USD/CHF": {"base_rate": 0.89, "volatility": 0.0040, "drift": 0.00001},
}


def generate_forex_data(
    pairs: Optional[List[str]] = None,
    years: int = 3,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Generate synthetic daily forex data for multiple currency pairs.

    Uses geometric Brownian motion with mean-reverting component
    to simulate realistic exchange rate movements.

    Args:
        pairs: List of currency pair names. Defaults to all defined pairs.
        years: Number of years of data to generate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping pair name to DataFrame with OHLCV data.
    """
    if pairs is None:
        pairs = list(PAIR_PROFILES.keys())

    rng = np.random.RandomState(seed)
    trading_days = int(years * 252)

    # Generate date index (business days)
    start_date = pd.Timestamp("2023-01-02")
    dates = pd.bdate_range(start=start_date, periods=trading_days)

    all_data = {}

    for pair in pairs:
        profile = PAIR_PROFILES.get(pair)
        if profile is None:
            continue

        base = profile["base_rate"]
        vol = profile["volatility"]
        drift = profile["drift"]

        # Generate close prices using GBM with mean reversion
        prices = np.zeros(trading_days)
        prices[0] = base

        mean_reversion_speed = 0.01

        for t in range(1, trading_days):
            # Mean-reverting GBM
            reversion = mean_reversion_speed * (base - prices[t - 1])
            shock = rng.normal(0, vol * prices[t - 1])
            trend = drift * prices[t - 1]

            # Add some seasonal pattern (quarterly)
            seasonal = 0.0005 * prices[t - 1] * np.sin(2 * np.pi * t / 63)

            prices[t] = prices[t - 1] + trend + reversion + shock + seasonal

            # Ensure positive
            prices[t] = max(prices[t], base * 0.5)

        # Generate OHLCV from close prices
        daily_range = vol * prices * rng.uniform(0.5, 2.0, trading_days)
        high = prices + daily_range * rng.uniform(0.3, 0.7, trading_days)
        low = prices - daily_range * rng.uniform(0.3, 0.7, trading_days)
        open_prices = low + (high - low) * rng.uniform(0.2, 0.8, trading_days)

        # Volume (synthetic)
        base_volume = rng.uniform(50_000, 200_000)
        volume = (base_volume * rng.lognormal(0, 0.3, trading_days)).astype(int)

        df = pd.DataFrame({
            "date": dates,
            "pair": pair,
            "open": np.round(open_prices, 5),
            "high": np.round(high, 5),
            "low": np.round(low, 5),
            "close": np.round(prices, 5),
            "volume": volume,
        })

        all_data[pair] = df

    return all_data


def save_datasets(
    data: Dict[str, pd.DataFrame],
    output_dir: Optional[str] = None,
) -> List[Path]:
    """Save forex data to CSV files.

    Args:
        data: Dictionary of pair name to DataFrame.
        output_dir: Output directory. Defaults to data/raw.

    Returns:
        List of paths to saved files.
    """
    if output_dir is None:
        output_dir = get_project_root() / "data" / "raw"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Save individual pair files
    for pair, df in data.items():
        filename = pair.replace("/", "_").lower() + ".csv"
        path = output_dir / filename
        df.to_csv(path, index=False)
        paths.append(path)

    # Save combined file
    combined = pd.concat(data.values(), ignore_index=True)
    combined_path = output_dir / "all_pairs.csv"
    combined.to_csv(combined_path, index=False)
    paths.append(combined_path)

    return paths


if __name__ == "__main__":
    data = generate_forex_data()
    paths = save_datasets(data)
    for pair, df in data.items():
        print(f"{pair}: {len(df)} days, range [{df['close'].min():.4f}, {df['close'].max():.4f}]")
