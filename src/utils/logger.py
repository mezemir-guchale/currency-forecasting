"""Logging configuration for the Currency Forecasting project."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.config import get_project_root


def setup_logger(
    name: str,
    config: Optional[Dict[str, Any]] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a configured logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = "INFO"
    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    if config and "logging" in config:
        level = config["logging"].get("level", level)
        fmt = config["logging"].get("format", fmt)

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file and config and "logging" in config:
        log_dir = get_project_root() / config["logging"]["log_dir"]
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
