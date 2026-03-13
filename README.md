# Market Direction Probability Dashboard

Local-first quantitative finance tool that ingests historical market data, engineers time-series features, trains three directional ML models, optionally applies live overrides, runs Monte Carlo simulation, and exports a self-contained interactive HTML dashboard.

## What It Does

- Loads CSV or Excel market data with a required date column and primary index column.
- Builds leakage-safe features including lagged returns, moving-average deviations, momentum, volatility, and yield spread when available.
- Trains Logistic Regression, Random Forest, and Gradient Boosting models with time-series cross-validation.
- Produces per-model probabilities, weighted ensemble metrics, evaluation diagnostics, and 2D/3D Plotly charts.
- Exports a single browser-openable HTML file with all scripts and data embedded.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
market-dashboard \
  --input examples/sample_market_data.csv \
  --index-column NIFTY \
  --output dashboard.html
```

Optional live overrides:

```bash
market-dashboard \
  --input examples/sample_market_data.csv \
  --index-column NIFTY \
  --output dashboard.html \
  --live-as-of 2026-03-13 \
  --override NIFTY=24158 \
  --override INDIA_VIX=23.36 \
  --override USD_INR=92.30
```

## Input Expectations

- One date column such as `Date`, `Month`, or `Datetime`
- One primary market index column supplied via `--index-column`
- Additional numeric columns for related markets, commodities, FX, or yields
- Monthly or weekly history with at least 60 rows recommended

## Tests

```bash
python3 -m unittest discover -s tests
```
