# Currency Exchange Rate Forecasting

A time series forecasting system for currency exchange rates using ARIMA, Exponential Smoothing, and baseline models. Analyzes synthetic forex data for 5 major currency pairs with comprehensive feature engineering, model evaluation, and visualization.

## Author

**Mezemir Neway Guchale**
- Email: gumezemir@gmail.com
- LinkedIn: [linkedin.com/in/mezemir-guchale](https://linkedin.com/in/mezemir-guchale)

## Project Overview

This project generates 3 years of synthetic daily forex data for 5 currency pairs (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF) and applies multiple forecasting approaches:

### Models
| Model | Description |
|-------|-------------|
| SMA | Simple Moving Average baseline |
| EMA | Exponential Moving Average |
| ARIMA(2,1,2) | Auto-Regressive Integrated Moving Average |
| Holt-Winters | Exponential Smoothing with trend |

### Features Engineered
- Lag features (1, 2, 3, 5, 7, 14, 21 days)
- Rolling statistics (mean, std, min, max) at 5, 10, 21, 50 day windows
- Return features (daily, log, cumulative)
- Technical indicators (RSI, MACD, Bollinger Bands, ATR)
- Time series decomposition (trend, seasonal, residual)

### Evaluation Metrics
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- MAPE (Mean Absolute Percentage Error)
- Directional Accuracy

## Project Structure

```
07-currency-forecasting/
├── configs/config.yaml
├── scripts/run_pipeline.py
├── src/
│   ├── data/
│   │   ├── generate_dataset.py
│   │   └── loader.py
│   ├── features/
│   │   └── time_series.py
│   ├── models/
│   │   └── forecaster.py
│   ├── evaluation/
│   │   └── metrics.py
│   ├── visualization/
│   │   └── charts.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── test_features.py
│   └── test_forecaster.py
├── requirements.txt
├── setup.py
└── README.md
```

## Setup and Usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python scripts/run_pipeline.py
pytest tests/ -v
```

## Output

- **Forecast plots**: Actual vs predicted for each pair and model
- **Residual analysis**: Distribution, autocorrelation, actual vs predicted scatter
- **Decomposition**: Trend, seasonal, and residual components
- **Model comparison**: Bar charts of metrics across models

## License

MIT License
