"""Data loading utilities for forex data."""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

from src.utils.config import get_project_root


def load_pair_data(
    pair: str,
    data_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Load data for a single currency pair.

    Args:
        pair: Currency pair name (e.g., 'EUR/USD').
        data_dir: Directory containing CSV files.

    Returns:
        DataFrame with forex data.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    filename = pair.replace("/", "_").lower() + ".csv"
    filepath = data_dir / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_all_pairs(
    data_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Load data for all available currency pairs.

    Args:
        data_dir: Directory containing CSV files.

    Returns:
        Dictionary mapping pair name to DataFrame.
    """
    if data_dir is None:
        data_dir = get_project_root() / "data" / "raw"
    else:
        data_dir = Path(data_dir)

    all_data = {}
    for f in sorted(data_dir.glob("*.csv")):
        if f.name == "all_pairs.csv":
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        if "pair" in df.columns:
            pair = df["pair"].iloc[0]
            all_data[pair] = df.sort_values("date").reset_index(drop=True)

    return all_data


def train_test_split_ts(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
) -> tuple:
    """Split time series data into train and test sets.

    Args:
        df: DataFrame sorted by date.
        train_ratio: Proportion of data for training.

    Returns:
        Tuple of (train_df, test_df).
    """
    n = len(df)
    split_idx = int(n * train_ratio)
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
