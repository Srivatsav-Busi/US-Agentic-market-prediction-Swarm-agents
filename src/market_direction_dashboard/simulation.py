from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    paths: np.ndarray
    summary: dict[str, float]
    density_surface: np.ndarray
    price_axis: np.ndarray


def run_gbm_simulation(
    index_prices: pd.Series,
    index_returns: pd.Series,
    date_series: pd.Series,
    vix_level: float | None = None,
    days: int = 30,
    simulations: int = 10_000,
    seed: int = 42,
) -> MonteCarloResult:
    start_price = float(index_prices.iloc[-1])
    clean_returns = index_returns.dropna()
    daily_scaler = _infer_period_days(date_series) or 21.0
    mu_daily = float(clean_returns.mean() / daily_scaler)
    sigma_hist = float(clean_returns.std() / np.sqrt(daily_scaler)) if len(clean_returns) > 1 else 0.01
    sigma_implied = float(vix_level) / 100.0 / np.sqrt(252.0) if vix_level is not None else sigma_hist
    sigma = 0.7 * sigma_implied + 0.3 * sigma_hist

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((days, simulations))
    dt = 1.0
    increments = (mu_daily - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * shocks
    log_paths = np.vstack([np.zeros(simulations), np.cumsum(increments, axis=0)])
    paths = start_price * np.exp(log_paths)
    terminal = paths[-1]
    summary = {
        "expected_price": float(terminal.mean()),
        "median_price": float(np.median(terminal)),
        "p5": float(np.percentile(terminal, 5)),
        "p95": float(np.percentile(terminal, 95)),
        "p_up": float((terminal > start_price).mean()),
        "p_down": float((terminal <= start_price).mean()),
        "p_plus_5": float((terminal >= start_price * 1.05).mean()),
        "p_minus_5": float((terminal <= start_price * 0.95).mean()),
        "sigma_daily": float(sigma),
        "mu_daily": float(mu_daily),
        "start_price": start_price,
    }

    density_surface, price_axis = _build_density_surface(paths)
    return MonteCarloResult(paths=paths, summary=summary, density_surface=density_surface, price_axis=price_axis)


def combined_signal(ensemble_up: float, monte_carlo_up: float) -> tuple[str, float]:
    combined = (ensemble_up + monte_carlo_up) / 2.0
    if combined >= 0.65:
        label = "STRONG BULL SIGNAL"
    elif combined >= 0.55:
        label = "BULLISH BIAS"
    elif combined <= 0.35:
        label = "STRONG BEAR SIGNAL"
    elif combined <= 0.45:
        label = "BEARISH BIAS"
    else:
        label = "NEUTRAL / MIXED"
    return label, abs(combined - 0.5) * 200.0


def _infer_period_days(date_series: pd.Series) -> float:
    normalized = pd.to_datetime(date_series, errors="coerce")
    diffs = normalized.diff().dropna().dt.days
    if diffs.empty:
        return 21.0
    return float(max(diffs.median(), 1))


def _build_density_surface(paths: np.ndarray, bins: int = 40) -> tuple[np.ndarray, np.ndarray]:
    low = paths.min()
    high = paths.max()
    price_axis = np.linspace(low, high, bins)
    density = np.zeros((bins - 1, paths.shape[0]))
    for day in range(paths.shape[0]):
        hist, _ = np.histogram(paths[day], bins=price_axis, density=True)
        density[:, day] = hist
    return density, price_axis[:-1]
