"""
Currency Exchange Rate Forecasting Dashboard
=============================================
Interactive Streamlit app for forecasting forex exchange rates
using ARIMA, Exponential Smoothing, SMA, and EMA models.
"""

import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Forex Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Currency pair profiles (from generate_dataset.py)
# ---------------------------------------------------------------------------
PAIR_PROFILES = {
    "EUR/USD": {"base_rate": 1.10, "volatility": 0.0045, "drift": 0.00002},
    "GBP/USD": {"base_rate": 1.27, "volatility": 0.0055, "drift": -0.00001},
    "USD/JPY": {"base_rate": 140.0, "volatility": 0.0050, "drift": 0.00003},
    "AUD/USD": {"base_rate": 0.67, "volatility": 0.0060, "drift": -0.00002},
    "USD/CHF": {"base_rate": 0.89, "volatility": 0.0040, "drift": 0.00001},
}

PAIRS = list(PAIR_PROFILES.keys())

# ---------------------------------------------------------------------------
# Data generation (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Generating forex data ...")
def generate_forex_data(years: int = 3, seed: int = 42) -> dict:
    """Generate synthetic daily forex OHLCV data using geometric Brownian
    motion with mean-reverting drift and quarterly seasonality."""

    rng = np.random.RandomState(seed)
    trading_days = int(years * 252)
    start_date = pd.Timestamp("2023-01-02")
    dates = pd.bdate_range(start=start_date, periods=trading_days)

    all_data = {}
    for pair, profile in PAIR_PROFILES.items():
        base = profile["base_rate"]
        vol = profile["volatility"]
        drift = profile["drift"]
        mean_reversion_speed = 0.01

        prices = np.zeros(trading_days)
        prices[0] = base
        for t in range(1, trading_days):
            reversion = mean_reversion_speed * (base - prices[t - 1])
            shock = rng.normal(0, vol * prices[t - 1])
            trend = drift * prices[t - 1]
            seasonal = 0.0005 * prices[t - 1] * np.sin(2 * np.pi * t / 63)
            prices[t] = prices[t - 1] + trend + reversion + shock + seasonal
            prices[t] = max(prices[t], base * 0.5)

        daily_range = vol * prices * rng.uniform(0.5, 2.0, trading_days)
        high = prices + daily_range * rng.uniform(0.3, 0.7, trading_days)
        low = prices - daily_range * rng.uniform(0.3, 0.7, trading_days)
        open_prices = low + (high - low) * rng.uniform(0.2, 0.8, trading_days)
        base_volume = rng.uniform(50_000, 200_000)
        volume = (base_volume * rng.lognormal(0, 0.3, trading_days)).astype(int)

        df = pd.DataFrame({
            "date": dates,
            "open": np.round(open_prices, 5),
            "high": np.round(high, 5),
            "low": np.round(low, 5),
            "close": np.round(prices, 5),
            "volume": volume,
        }).set_index("date")

        all_data[pair] = df

    return all_data


# ---------------------------------------------------------------------------
# Forecasting helpers (cached)
# ---------------------------------------------------------------------------

def _fit_sma(train: np.ndarray, horizon: int, window: int = 21) -> np.ndarray:
    sma_val = np.mean(train[-window:])
    return np.full(horizon, sma_val)


def _fit_ema(train: np.ndarray, horizon: int, span: int = 21) -> np.ndarray:
    s = pd.Series(train)
    ema_val = float(s.ewm(span=span, adjust=False).mean().iloc[-1])
    return np.full(horizon, ema_val)


def _fit_arima(train: np.ndarray, horizon: int, order=(2, 1, 2)) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(train, order=order)
            fit = model.fit()
            return np.array(fit.forecast(steps=horizon))
        except Exception:
            try:
                model = ARIMA(train, order=(1, 1, 0))
                fit = model.fit()
                return np.array(fit.forecast(steps=horizon))
            except Exception:
                return np.full(horizon, train[-1])


def _fit_exp_smoothing(train: np.ndarray, horizon: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ExponentialSmoothing(train, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            return np.array(fit.forecast(horizon))
        except Exception:
            try:
                model = ExponentialSmoothing(train, trend=None, seasonal=None)
                fit = model.fit(optimized=True)
                return np.array(fit.forecast(horizon))
            except Exception:
                return np.full(horizon, train[-1])


@st.cache_data(show_spinner="Running forecasting models ...")
def run_forecasts(
    close_values: tuple,  # tuple for hashability
    horizon: int = 30,
    train_ratio: float = 0.8,
) -> dict:
    """Split data, fit all models, return forecasts + actuals."""

    close = np.array(close_values)
    split = int(len(close) * train_ratio)
    train = close[:split]
    test = close[split:]

    # Only evaluate on min(horizon, len(test)) actual points
    eval_len = min(horizon, len(test))
    actual = test[:eval_len]

    forecasts = {
        "ARIMA(2,1,2)": _fit_arima(train, horizon),
        "Exp. Smoothing": _fit_exp_smoothing(train, horizon),
        "SMA(21)": _fit_sma(train, horizon),
        "EMA(21)": _fit_ema(train, horizon),
    }

    return {
        "train": train,
        "test": test,
        "actual": actual,
        "forecasts": forecasts,
        "split_idx": split,
        "horizon": horizon,
        "eval_len": eval_len,
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    n = min(len(actual), len(predicted))
    a, p = actual[:n], predicted[:n]
    mae_val = float(np.mean(np.abs(a - p)))
    rmse_val = float(np.sqrt(np.mean((a - p) ** 2)))
    mask = a != 0
    mape_val = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else float("inf")
    return {"RMSE": rmse_val, "MAE": mae_val, "MAPE (%)": mape_val}


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Settings")
selected_pair = st.sidebar.selectbox("Currency Pair", PAIRS, index=0)
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 5, 60, 30)
train_ratio = st.sidebar.slider("Train / Test Split", 0.6, 0.95, 0.80, 0.05)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Data is synthetically generated using geometric Brownian motion "
    "with mean reversion and quarterly seasonality."
)

# ---------------------------------------------------------------------------
# Generate data and run models
# ---------------------------------------------------------------------------

all_data = generate_forex_data()
df = all_data[selected_pair]
close_series = df["close"]

results = run_forecasts(
    tuple(close_series.values),
    horizon=forecast_horizon,
    train_ratio=train_ratio,
)

train_vals = results["train"]
test_vals = results["test"]
actual_vals = results["actual"]
forecasts = results["forecasts"]
split_idx = results["split_idx"]
eval_len = results["eval_len"]

dates = df.index
train_dates = dates[:split_idx]
test_dates = dates[split_idx:]
forecast_dates = test_dates[:eval_len]

# Build comparison table
metrics_rows = []
for model_name, pred in forecasts.items():
    m = compute_metrics(actual_vals, pred[:eval_len])
    m["Model"] = model_name
    metrics_rows.append(m)

metrics_df = pd.DataFrame(metrics_rows).set_index("Model")
best_model = metrics_df["RMSE"].idxmin()

# ---------------------------------------------------------------------------
# Dashboard header
# ---------------------------------------------------------------------------

st.title("Currency Exchange Rate Forecasting")
st.markdown(f"### {selected_pair}")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
latest_close = close_series.iloc[-1]
prev_close = close_series.iloc[-2]
pct_change = (latest_close - prev_close) / prev_close * 100
high_52w = close_series.iloc[-252:].max()
low_52w = close_series.iloc[-252:].min()
avg_vol = df["volume"].iloc[-21:].mean()

col1.metric("Latest Close", f"{latest_close:.5f}", f"{pct_change:+.3f}%")
col2.metric("52-Week High", f"{high_52w:.5f}")
col3.metric("52-Week Low", f"{low_52w:.5f}")
col4.metric("Avg Volume (21d)", f"{avg_vol:,.0f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_overview, tab_models, tab_residuals = st.tabs(
    ["Overview & Forecast", "Model Comparison", "Residual Analysis"]
)

# ===== TAB 1: Overview & Forecast ==========================================
with tab_overview:
    # --- Historical price chart with forecast overlay ----------------------
    fig_main = go.Figure()

    # Training period
    fig_main.add_trace(go.Scatter(
        x=train_dates, y=train_vals,
        mode="lines", name="Train",
        line=dict(color="#636EFA"),
    ))

    # Test period
    fig_main.add_trace(go.Scatter(
        x=test_dates, y=test_vals,
        mode="lines", name="Test (Actual)",
        line=dict(color="#EF553B"),
    ))

    # Forecasts
    colors = {"ARIMA(2,1,2)": "#00CC96", "Exp. Smoothing": "#AB63FA",
              "SMA(21)": "#FFA15A", "EMA(21)": "#19D3F3"}
    for model_name, pred in forecasts.items():
        fig_main.add_trace(go.Scatter(
            x=forecast_dates, y=pred[:eval_len],
            mode="lines", name=model_name,
            line=dict(color=colors.get(model_name, "#B6E880"), dash="dash"),
        ))

    # Train/test split line
    fig_main.add_vline(
        x=train_dates[-1], line_dash="dot", line_color="gray",
        annotation_text="Train / Test Split",
    )

    fig_main.update_layout(
        title=f"{selected_pair} — Historical Prices & Forecast Overlay",
        xaxis_title="Date", yaxis_title="Exchange Rate",
        template="plotly_white", height=520, legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_main, use_container_width=True)

    # --- Best model callout ------------------------------------------------
    best_rmse = metrics_df.loc[best_model, "RMSE"]
    best_mae = metrics_df.loc[best_model, "MAE"]
    best_mape = metrics_df.loc[best_model, "MAPE (%)"]

    st.success(f"**Best model: {best_model}**  —  RMSE {best_rmse:.6f}  |  MAE {best_mae:.6f}  |  MAPE {best_mape:.4f}%")

    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.metric(f"{best_model} RMSE", f"{best_rmse:.6f}")
    bcol2.metric(f"{best_model} MAE", f"{best_mae:.6f}")
    bcol3.metric(f"{best_model} MAPE", f"{best_mape:.4f}%")

# ===== TAB 2: Model Comparison =============================================
with tab_models:
    st.subheader("Model Performance Comparison")

    # Styled comparison table
    def highlight_best(s):
        is_min = s == s.min()
        return ["background-color: #d4edda" if v else "" for v in is_min]

    styled = metrics_df.style.apply(highlight_best).format({
        "RMSE": "{:.6f}", "MAE": "{:.6f}", "MAPE (%)": "{:.4f}",
    })
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # Individual forecast charts
    st.subheader("Individual Model Forecasts")

    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)
    chart_slots = [row1_left, row1_right, row2_left, row2_right]

    for (model_name, pred), slot in zip(forecasts.items(), chart_slots):
        with slot:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=actual_vals,
                mode="lines", name="Actual",
                line=dict(color="#EF553B"),
            ))
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=pred[:eval_len],
                mode="lines", name=model_name,
                line=dict(color=colors.get(model_name, "#636EFA"), dash="dash"),
            ))
            fig.update_layout(
                title=model_name, template="plotly_white",
                height=320, margin=dict(t=40, b=30),
                showlegend=True, legend=dict(orientation="h", y=-0.25),
                xaxis_title="", yaxis_title="Rate",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Bar chart comparison
    st.subheader("Metric Comparison (Bar Chart)")
    bar_df = metrics_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Value")

    fig_bar = px.bar(
        bar_df, x="Model", y="Value", color="Model", facet_col="Metric",
        template="plotly_white", height=380,
    )
    fig_bar.update_yaxes(matches=None)
    fig_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig_bar, use_container_width=True)

# ===== TAB 3: Residual Analysis ============================================
with tab_residuals:
    st.subheader("Residual Analysis")

    res_model = st.selectbox("Select model for residual analysis", list(forecasts.keys()))
    pred_vals = forecasts[res_model][:eval_len]
    residuals = actual_vals - pred_vals

    rcol1, rcol2 = st.columns(2)

    with rcol1:
        # Residuals over time
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=forecast_dates, y=residuals,
            mode="lines+markers", name="Residuals",
            line=dict(color="#636EFA"),
            marker=dict(size=4),
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_res.update_layout(
            title=f"{res_model} — Residuals Over Time",
            xaxis_title="Date", yaxis_title="Residual",
            template="plotly_white", height=380,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with rcol2:
        # Residual histogram
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Histogram(
            x=residuals, nbinsx=25, name="Residuals",
            marker_color="#AB63FA", opacity=0.8,
        ))
        fig_hist.update_layout(
            title=f"{res_model} — Residual Distribution",
            xaxis_title="Residual", yaxis_title="Count",
            template="plotly_white", height=380,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Residual summary metrics
    st.markdown("#### Residual Summary")
    scol1, scol2, scol3, scol4 = st.columns(4)
    scol1.metric("Mean", f"{np.mean(residuals):.6f}")
    scol2.metric("Std Dev", f"{np.std(residuals):.6f}")
    scol3.metric("Min", f"{np.min(residuals):.6f}")
    scol4.metric("Max", f"{np.max(residuals):.6f}")

    # Actual vs Predicted scatter
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=actual_vals, y=pred_vals,
        mode="markers", name="Predictions",
        marker=dict(color="#00CC96", size=7, opacity=0.7),
    ))
    min_val = min(actual_vals.min(), pred_vals.min())
    max_val = max(actual_vals.max(), pred_vals.max())
    fig_scatter.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val],
        mode="lines", name="Perfect Forecast",
        line=dict(color="gray", dash="dash"),
    ))
    fig_scatter.update_layout(
        title=f"{res_model} — Actual vs Predicted",
        xaxis_title="Actual", yaxis_title="Predicted",
        template="plotly_white", height=420,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Built with Streamlit, Plotly & statsmodels  |  "
    "Data is synthetically generated for demonstration purposes."
)
