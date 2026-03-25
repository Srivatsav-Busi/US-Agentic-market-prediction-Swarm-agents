from __future__ import annotations

from dataclasses import dataclass, asdict
from math import sqrt
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data_sources import fetch_historical_market_data
from ..storage.db import database_session
from ..storage.repositories import MarketRepository


@dataclass
class BaselineForecast:
    horizon_days: int
    regime_label: str
    expected_return_30d: float
    expected_volatility_30d: float
    projection: dict[str, Any]
    sectors: list[dict[str, Any]]
    top_drivers: list[dict[str, Any]]
    confidence_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_baseline_forecast(target: str, snapshot: dict, config: dict, *, horizon_days: int = 30) -> BaselineForecast:
    history_frame = _load_history_frame(target=target, snapshot=snapshot, config=config)
    if target not in history_frame.columns:
        return _empty_forecast(horizon_days)

    target_series = history_frame[target].dropna()
    if len(target_series) < 40:
        return _empty_forecast(horizon_days)

    daily_returns = target_series.pct_change().dropna()
    short_momentum = float(target_series.pct_change(20).iloc[-1]) if len(target_series) > 20 else 0.0
    medium_momentum = float(target_series.pct_change(60).iloc[-1]) if len(target_series) > 60 else short_momentum
    daily_drift = float(np.nanmean([daily_returns.tail(5).mean(), daily_returns.tail(20).mean(), daily_returns.tail(60).mean()]))
    daily_vol = float(daily_returns.tail(20).std() or daily_returns.std() or 0.0)
    expected_return_30d = daily_drift * horizon_days
    expected_volatility_30d = daily_vol * sqrt(horizon_days)
    regime_label = _regime_label(short_momentum, daily_vol)

    projection = _build_projection(
        latest_price=float(target_series.iloc[-1]),
        latest_date=target_series.index[-1],
        daily_drift=daily_drift,
        daily_vol=daily_vol,
        horizon_days=horizon_days,
        target=target,
    )
    sectors = _build_sector_outlook(history_frame, target, horizon_days)
    top_drivers = _top_drivers(short_momentum, medium_momentum, daily_vol, sectors, regime_label)
    confidence_notes = _confidence_notes(daily_vol, sectors, len(target_series))

    return BaselineForecast(
        horizon_days=horizon_days,
        regime_label=regime_label,
        expected_return_30d=expected_return_30d,
        expected_volatility_30d=expected_volatility_30d,
        projection=projection,
        sectors=sectors,
        top_drivers=top_drivers,
        confidence_notes=confidence_notes,
    )


def apply_hybrid_overlay(
    baseline: BaselineForecast,
    ml_result: dict[str, Any],
    evidence_bias: float,
    regime_probabilities: dict[str, float],
) -> BaselineForecast:
    if ml_result.get("status") != "ok":
        return baseline
    regime_bias = float(regime_probabilities.get("risk_on", 0.0) - regime_probabilities.get("risk_off", 0.0))
    blended_return = (
        baseline.expected_return_30d * 0.45
        + float(ml_result.get("predicted_return_30d", 0.0)) * 0.35
        + regime_bias * 0.10
        + evidence_bias * 0.10
    )
    scaled_projection = _rescale_projection(baseline.projection, blended_return, baseline.expected_return_30d)
    adjusted_sectors = []
    for row in baseline.sectors:
        adjusted_return = row["expected_return_30d"] * 0.8 + blended_return * 0.2
        adjusted_score = row["ranking_score"] * 0.85 + (adjusted_return - row["expected_volatility_30d"] * 0.3) * 0.15
        label = "FAVOR" if adjusted_score > 0.02 else "AVOID" if adjusted_score < -0.02 else "NEUTRAL"
        adjusted_sectors.append(
            {
                **row,
                "expected_return_30d": round(adjusted_return, 6),
                "ranking_score": round(adjusted_score, 6),
                "recommendation_label": label,
            }
        )
    adjusted_sectors.sort(key=lambda item: item["ranking_score"], reverse=True)
    return BaselineForecast(
        horizon_days=baseline.horizon_days,
        regime_label=baseline.regime_label,
        expected_return_30d=blended_return,
        expected_volatility_30d=baseline.expected_volatility_30d * 0.85 + float(ml_result.get("expected_volatility_30d", 0.0)) * 0.15,
        projection=scaled_projection,
        sectors=adjusted_sectors,
        top_drivers=baseline.top_drivers,
        confidence_notes=baseline.confidence_notes,
    )


def _load_history_frame(target: str, snapshot: dict, config: dict) -> pd.DataFrame:
    if config.get("database_url"):
        try:
            with database_session(config["database_url"]) as session:
                frame = MarketRepository(session).load_price_history_frame()
            if not frame.empty and target in frame.columns and len(frame[target].dropna()) >= 40:
                return frame
        except Exception:
            pass

    history = snapshot.get("history", {})
    series_map: dict[str, pd.Series] = {}
    for label, rows in history.items():
        values = {}
        for row in rows:
            dt = pd.to_datetime(row.get("date"), errors="coerce")
            if pd.isna(dt):
                continue
            values[dt] = float(row.get("value", 0.0))
        if values:
            series_map[label] = pd.Series(values)
    if target in snapshot.get("series", {}) and target not in series_map:
        latest_date = pd.to_datetime(snapshot["series"][target].get("date") or snapshot.get("latest_date"), errors="coerce")
        latest_value = snapshot["series"][target].get("latest")
        if pd.notna(latest_date) and latest_value is not None:
            series_map[target] = pd.Series({latest_date: float(latest_value)})
    frame = pd.DataFrame(series_map).sort_index().ffill()
    sector_names = set(config.get("sector_symbols", {}).keys())
    if (target not in frame.columns or not sector_names.intersection(set(frame.columns))) and config:
        try:
            fetched_history, _ = fetch_historical_market_data(config, years=1)
            for label, rows in fetched_history.items():
                values = {}
                for row in rows:
                    dt = pd.to_datetime(row.get("date"), errors="coerce")
                    if pd.isna(dt):
                        continue
                    values[dt] = float(row.get("value", 0.0))
                if values:
                    frame[label] = pd.Series(values)
            frame = frame.sort_index().ffill()
        except Exception:
            pass
    return frame


def _build_projection(*, latest_price: float, latest_date: pd.Timestamp, daily_drift: float, daily_vol: float, horizon_days: int, target: str) -> dict[str, Any]:
    future_dates = pd.bdate_range(start=latest_date + pd.offsets.BDay(1), periods=horizon_days)
    base_path = []
    bull_path = []
    bear_path = []
    lower_path = []
    upper_path = []
    current_price = latest_price
    for index, forecast_date in enumerate(future_dates, start=1):
        base_return = daily_drift * index
        band = daily_vol * sqrt(index)
        base_price = latest_price * (1.0 + base_return)
        bull_price = latest_price * (1.0 + base_return + band * 0.6)
        bear_price = latest_price * (1.0 + base_return - band * 0.6)
        lower_band = latest_price * (1.0 + base_return - band)
        upper_band = latest_price * (1.0 + base_return + band)
        point = {
            "forecast_date": forecast_date.date().isoformat(),
            "horizon_day": index,
            "target_symbol": target,
            "predicted_price": round(float(base_price), 4),
            "predicted_return": round(float(base_price / latest_price - 1.0), 6),
            "lower_band": round(float(lower_band), 4),
            "upper_band": round(float(upper_band), 4),
        }
        base_path.append(point)
        bull_path.append({**point, "scenario_type": "bull", "predicted_price": round(float(bull_price), 4)})
        bear_path.append({**point, "scenario_type": "bear", "predicted_price": round(float(bear_price), 4)})
        lower_path.append(round(float(lower_band), 4))
        upper_path.append(round(float(upper_band), 4))
        current_price = base_price
    return {
        "latest_price": round(latest_price, 4),
        "latest_date": latest_date.date().isoformat(),
        "base": [{**point, "scenario_type": "base"} for point in base_path],
        "bull": bull_path,
        "bear": bear_path,
        "confidence_band": {"lower": lower_path, "upper": upper_path},
    }


def _build_sector_outlook(history_frame: pd.DataFrame, target: str, horizon_days: int) -> list[dict[str, Any]]:
    sectors = []
    if target not in history_frame.columns:
        return sectors
    target_series = history_frame[target].dropna()
    target_return_20 = float(target_series.pct_change(20).iloc[-1]) if len(target_series) > 20 else 0.0
    for column in history_frame.columns:
        if column == target:
            continue
        series = history_frame[column].dropna()
        if len(series) < 25:
            continue
        ret_20 = float(series.pct_change(20).iloc[-1])
        daily_vol = float(series.pct_change().tail(20).std() or 0.0)
        rel_strength = ret_20 - target_return_20
        expected_return = rel_strength * 0.65 + ret_20 * 0.35
        expected_volatility = daily_vol * sqrt(horizon_days)
        ranking_score = expected_return - expected_volatility * 0.35
        confidence = max(35.0, min(82.0, 68.0 - expected_volatility * 100.0 + abs(rel_strength) * 40.0))
        label = "FAVOR" if ranking_score > 0.02 else "AVOID" if ranking_score < -0.02 else "NEUTRAL"
        sectors.append(
            {
                "sector_symbol": column,
                "sector_name": column,
                "ranking_score": round(ranking_score, 6),
                "expected_return_30d": round(expected_return, 6),
                "expected_volatility_30d": round(expected_volatility, 6),
                "confidence": round(confidence, 1),
                "recommendation_label": label,
                "rationale": _sector_rationale(rel_strength, expected_volatility, label),
            }
        )
    sectors.sort(key=lambda item: item["ranking_score"], reverse=True)
    return sectors[:11]


def _top_drivers(short_momentum: float, medium_momentum: float, daily_vol: float, sectors: list[dict[str, Any]], regime_label: str) -> list[dict[str, Any]]:
    leaders = ", ".join(item["sector_name"] for item in sectors[:3]) if sectors else "none"
    laggards = ", ".join(item["sector_name"] for item in sectors[-3:]) if sectors else "none"
    return [
        {"name": "20d momentum", "direction": "positive" if short_momentum >= 0 else "negative", "value": round(short_momentum, 6), "summary": "Recent target momentum is driving the baseline drift estimate."},
        {"name": "60d trend", "direction": "positive" if medium_momentum >= 0 else "negative", "value": round(medium_momentum, 6), "summary": "Medium-horizon trend anchors the 30-day path."},
        {"name": "20d realized volatility", "direction": "negative" if daily_vol > 0.015 else "neutral", "value": round(daily_vol, 6), "summary": "Recent volatility determines confidence-band width."},
        {"name": "Sector leadership", "direction": "positive", "value": leaders, "summary": f"Leading sectors in the current {regime_label} regime: {leaders}."},
        {"name": "Sector drag", "direction": "negative", "value": laggards, "summary": f"Weaker sectors currently include {laggards}."},
    ]


def _confidence_notes(daily_vol: float, sectors: list[dict[str, Any]], history_length: int) -> list[str]:
    notes = []
    if history_length < 120:
        notes.append("History coverage remains relatively short for a 30-day baseline model.")
    if daily_vol > 0.02:
        notes.append("Realized volatility is elevated, so confidence bands are wider than normal.")
    if not sectors:
        notes.append("Sector ranking used limited or unavailable sector history.")
    elif any(item["recommendation_label"] == "NEUTRAL" for item in sectors[:3]):
        notes.append("Sector leadership is mixed, which lowers conviction in rotation calls.")
    return notes


def _regime_label(short_momentum: float, daily_vol: float) -> str:
    if short_momentum > 0.03 and daily_vol < 0.015:
        return "risk_on"
    if short_momentum < -0.03 and daily_vol > 0.015:
        return "risk_off"
    return "mixed"


def _sector_rationale(rel_strength: float, expected_volatility: float, label: str) -> str:
    if label == "FAVOR":
        return f"Relative strength is supportive ({rel_strength:+.2%}) with manageable 30-day volatility."
    if label == "AVOID":
        return f"Relative strength is weak ({rel_strength:+.2%}) or volatility is elevated."
    return "Signals are mixed versus the broad market over the recent lookback."


def _empty_forecast(horizon_days: int) -> BaselineForecast:
    return BaselineForecast(
        horizon_days=horizon_days,
        regime_label="insufficient_data",
        expected_return_30d=0.0,
        expected_volatility_30d=0.0,
        projection={"latest_price": None, "latest_date": None, "base": [], "bull": [], "bear": [], "confidence_band": {"lower": [], "upper": []}},
        sectors=[],
        top_drivers=[],
        confidence_notes=["Insufficient stored history for a 30-day baseline forecast."],
    )


def _rescale_projection(projection: dict[str, Any], target_return: float, baseline_return: float) -> dict[str, Any]:
    latest_price = projection.get("latest_price")
    if not latest_price or not projection.get("base"):
        return projection
    adjustment = target_return - baseline_return
    adjusted = {"latest_price": latest_price, "latest_date": projection.get("latest_date"), "confidence_band": projection.get("confidence_band", {})}
    for scenario_name in ("base", "bull", "bear"):
        points = []
        source = projection.get(scenario_name, [])
        for point in source:
            horizon_day = point["horizon_day"]
            incremental = adjustment * (horizon_day / max(len(source), 1))
            new_return = point["predicted_return"] + incremental
            new_price = latest_price * (1.0 + new_return)
            points.append(
                {
                    **point,
                    "predicted_return": round(float(new_return), 6),
                    "predicted_price": round(float(new_price), 4),
                }
            )
        adjusted[scenario_name] = points
    return adjusted
