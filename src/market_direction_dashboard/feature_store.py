from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from uuid import uuid4

import numpy as np
import pandas as pd

from .storage.models import FeatureSnapshotRecord


def build_feature_snapshots(
    price_history: dict[str, list[dict]],
    evidence_items: list,
    target_scope: str = "S&P 500",
    run_id: str | None = None,
) -> list[FeatureSnapshotRecord]:
    run_token = run_id or str(uuid4())
    pivot = _build_price_frame(price_history)
    if pivot.empty:
        return []

    features: list[FeatureSnapshotRecord] = []
    target_column = target_scope if target_scope in pivot.columns else next(iter(pivot.columns))
    target_close = pivot[target_column]
    target_returns = target_close.pct_change()

    for column in pivot.columns:
        series = pivot[column]
        scope = column
        derived = {
            f"{_slug(column)}__return_1d": series.pct_change(1),
            f"{_slug(column)}__return_5d": series.pct_change(5),
            f"{_slug(column)}__return_20d": series.pct_change(20),
            f"{_slug(column)}__volatility_20d": series.pct_change().rolling(20).std(),
            f"{_slug(column)}__drawdown_20d": series / series.rolling(20).max() - 1.0,
        }
        if column != target_column:
            derived[f"{_slug(column)}__relative_strength_20d"] = series.pct_change(20) - target_close.pct_change(20)
        for feature_name, values in derived.items():
            for snapshot_date, value in values.dropna().items():
                features.append(
                    FeatureSnapshotRecord(
                        snapshot_date=snapshot_date.date().isoformat(),
                        feature_name=feature_name,
                        feature_value=float(value),
                        feature_group=_infer_feature_group(feature_name),
                        target_scope=scope,
                        generation_run_id=run_token,
                    )
                )

    # Finance-specific cross-asset confirmation features for next-session direction quality.
    derived_global = {
        "cross_asset__breadth_ratio_1d": _breadth_ratio(pivot).dropna(),
        "cross_asset__risk_on_dispersion_5d": _risk_on_dispersion(pivot).dropna(),
        "cross_asset__target_vs_vix_5d": _relative_pair_signal(pivot, target_column, "VIX", window=5).dropna(),
        "cross_asset__target_vs_dxy_5d": _relative_pair_signal(pivot, target_column, "DXY", window=5).dropna(),
        "cross_asset__target_vs_yield_5d": _relative_pair_signal(pivot, target_column, "US 10 YR TREASURY", window=5).dropna(),
        "cross_asset__target_autocorr_5d": target_returns.rolling(20).corr(target_returns.shift(1)).dropna(),
    }
    for feature_name, values in derived_global.items():
        for snapshot_date, value in values.items():
            if pd.isna(value):
                continue
            features.append(
                FeatureSnapshotRecord(
                    snapshot_date=snapshot_date.date().isoformat(),
                    feature_name=feature_name,
                    feature_value=float(value),
                    feature_group="cross_asset",
                    target_scope=target_scope,
                    generation_run_id=run_token,
                )
            )

    for snapshot_date in pivot.index:
        dt = snapshot_date.to_pydatetime()
        calendar_values = {
            "calendar__day_of_week": float(dt.weekday()),
            "calendar__month": float(dt.month),
            "calendar__week_of_month": float(((dt.day - 1) // 7) + 1),
        }
        for feature_name, value in calendar_values.items():
            features.append(
                FeatureSnapshotRecord(
                    snapshot_date=dt.date().isoformat(),
                    feature_name=feature_name,
                    feature_value=value,
                    feature_group="calendar",
                    target_scope=target_scope,
                    generation_run_id=run_token,
                )
            )

    sentiment_by_day = aggregate_sentiment_features(evidence_items)
    for snapshot_date, metrics in sentiment_by_day.items():
        for feature_name, value in metrics.items():
            features.append(
                FeatureSnapshotRecord(
                    snapshot_date=snapshot_date,
                    feature_name=feature_name,
                    feature_value=float(value),
                    feature_group="sentiment",
                    target_scope=target_scope,
                    generation_run_id=run_token,
                )
            )

    return features


def aggregate_sentiment_features(evidence_items: list) -> dict[str, dict[str, float]]:
    aggregates: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "sentiment__bullish_count": 0.0,
            "sentiment__bearish_count": 0.0,
            "sentiment__net_impact": 0.0,
            "sentiment__avg_credibility": 0.0,
            "sentiment__item_count": 0.0,
            "sentiment__duplicate_pressure": 0.0,
            "sentiment__conflict_ratio": 0.0,
        }
    )
    duplicate_counter: dict[str, dict[str, float]] = defaultdict(lambda: {"items": 0.0, "clusters": 0.0})
    for item in evidence_items or []:
        published = getattr(item, "published_at", None) or ""
        if not published:
            continue
        date_key = published.split("T", 1)[0]
        slot = aggregates[date_key]
        impact = float(getattr(item, "impact_score", 0.0) or 0.0)
        slot["sentiment__item_count"] += 1.0
        slot["sentiment__net_impact"] += impact
        slot["sentiment__avg_credibility"] += float(getattr(item, "credibility_score", 0.0) or 0.0)
        if impact > 0:
            slot["sentiment__bullish_count"] += 1.0
        elif impact < 0:
            slot["sentiment__bearish_count"] += 1.0
        duplicate_counter[date_key]["items"] += 1.0
        if getattr(item, "duplicate_cluster", ""):
            duplicate_counter[date_key]["clusters"] += 1.0
    for slot in aggregates.values():
        count = max(slot["sentiment__item_count"], 1.0)
        slot["sentiment__avg_credibility"] /= count
        slot["sentiment__conflict_ratio"] = min(slot["sentiment__bullish_count"], slot["sentiment__bearish_count"]) / count
    for date_key, counts in duplicate_counter.items():
        aggregates[date_key]["sentiment__duplicate_pressure"] = counts["clusters"] / max(counts["items"], 1.0)
    return dict(aggregates)


def _build_price_frame(price_history: dict[str, list[dict]]) -> pd.DataFrame:
    series_map = {}
    for label, rows in price_history.items():
        if not rows:
            continue
        values = {}
        for row in rows:
            date_value = pd.to_datetime(row["date"], errors="coerce")
            if pd.isna(date_value):
                continue
            values[date_value] = float(row["value"])
        if values:
            series_map[label] = pd.Series(values).sort_index()
    if not series_map:
        return pd.DataFrame()
    frame = pd.DataFrame(series_map).sort_index().ffill().dropna(how="all")
    return frame


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _infer_feature_group(feature_name: str) -> str:
    if feature_name.startswith("cross_asset__"):
        return "cross_asset"
    if "volatility" in feature_name:
        return "volatility"
    if "relative_strength" in feature_name:
        return "relative_strength"
    if "drawdown" in feature_name:
        return "drawdown"
    return "price"


def _breadth_ratio(frame: pd.DataFrame) -> pd.Series:
    returns = frame.pct_change()
    advancing = (returns > 0).sum(axis=1)
    declining = (returns < 0).sum(axis=1)
    return (advancing - declining) / np.maximum((advancing + declining), 1)


def _risk_on_dispersion(frame: pd.DataFrame) -> pd.Series:
    labels = [label for label in ("S&P 500", "NASDAQ 100", "DOW JONES", "RUSSELL 2000") if label in frame.columns]
    if len(labels) < 2:
        return pd.Series(dtype=float)
    return frame[labels].pct_change(5).std(axis=1)


def _relative_pair_signal(frame: pd.DataFrame, target: str, other: str, *, window: int) -> pd.Series:
    if target not in frame.columns or other not in frame.columns:
        return pd.Series(dtype=float)
    return frame[target].pct_change(window) - frame[other].pct_change(window)
