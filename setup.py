"""Setup script for Currency Exchange Rate Forecasting."""

from setuptools import setup, find_packages

setup(
    name="currency-forecasting",
    version="1.0.0",
    description="Currency Exchange Rate Forecasting - Time series analysis and forecasting for forex data",
    author="Mezemir Neway Guchale",
    author_email="gumezemir@gmail.com",
    url="https://linkedin.com/in/mezemir-guchale",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "matplotlib>=3.6.0",
        "statsmodels>=0.13.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0"],
    },
)
