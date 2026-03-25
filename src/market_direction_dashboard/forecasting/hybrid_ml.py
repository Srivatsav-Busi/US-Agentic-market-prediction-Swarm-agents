from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

from .baseline_30d import _load_history_frame
from ..graph_features import GRAPH_FEATURE_COLUMNS, GraphFeatureVector
from ..storage.db import database_session
from ..storage.models import ModelMetricRecord, RetrainingEventRecord
from ..storage.repositories import MarketRepository


@dataclass
class HybridMLResult:
    predicted_return_30d: float
    expected_volatility_30d: float
    cv_mae: float
    feature_importances: list[dict[str, Any]]
    training_rows: int
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrainingDiagnostics:
    model_name: str
    model_version: str | None
    scheduled_date: str
    status: str
    training_window_start: str | None
    training_window_end: str | None
    training_row_count: int
    cv_mae: float
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_hybrid_ml_overlay(
    target: str,
    snapshot: dict,
    config: dict,
    *,
    graph_feature_vector: GraphFeatureVector | dict[str, Any] | None = None,
    horizon_days: int = 30,
) -> HybridMLResult:
    frame = _load_history_frame(target=target, snapshot=snapshot, config=config)
    if target not in frame.columns or len(frame) < 90:
        return HybridMLResult(0.0, 0.0, 0.0, [], 0, "insufficient_history")

    graph_frame = _load_graph_feature_frame(config=config, target=target, graph_feature_vector=graph_feature_vector)
    feature_frame = _build_feature_frame(frame, target=target, graph_feature_frame=graph_frame)
    dataset = _build_ml_dataset(feature_frame, frame[target], horizon_days=horizon_days)
    if len(dataset) < 60:
        return HybridMLResult(0.0, 0.0, 0.0, [], len(dataset), "insufficient_features")

    feature_columns = [column for column in dataset.columns if column not in {"target_return_30d"}]
    X = dataset[feature_columns]
    y = dataset["target_return_30d"]
    splitter = TimeSeriesSplit(n_splits=min(5, max(2, len(dataset) // 25)))
    
    estimators = [
        ("gbr", GradientBoostingRegressor(random_state=42, n_estimators=150, learning_rate=0.05, max_depth=3)),
        ("rf", RandomForestRegressor(random_state=42, n_estimators=150, max_depth=5, min_samples_leaf=3)),
        ("ridge", Ridge(alpha=10.0))
    ]
    model = VotingRegressor(estimators=estimators)

    maes = []
    for train_idx, test_idx in splitter.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        maes.append(mean_absolute_error(y.iloc[test_idx], preds))

    model.fit(X, y)
    inference_frame = feature_frame.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    if inference_frame.empty:
        return HybridMLResult(0.0, 0.0, 0.0, [], len(dataset), "insufficient_features")
    latest_prediction = float(model.predict(inference_frame[feature_columns].iloc[[-1]])[0])
    latest_vol = float(frame[target].pct_change().tail(20).std() or 0.0) * sqrt(horizon_days)
    
    # Extract importances from the GBR component for diagnostics
    gbr_model = model.named_estimators_["gbr"]
    importance = permutation_importance(gbr_model, X.tail(min(len(X), 40)), y.tail(min(len(y), 40)), n_repeats=5, random_state=42)
    ranked = sorted(
        (
            {"name": name, "importance": round(float(score), 6)}
            for name, score in zip(feature_columns, importance.importances_mean, strict=False)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )[:6]
    return HybridMLResult(
        predicted_return_30d=latest_prediction,
        expected_volatility_30d=latest_vol,
        cv_mae=float(np.mean(maes)) if maes else 0.0,
        feature_importances=ranked,
        training_rows=len(dataset),
        status="ok",
    )


def _build_feature_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    graph_feature_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    dataset = pd.DataFrame(index=frame.index)
    target_series = frame[target]
    dataset["target_return_1d"] = target_series.pct_change(1)
    dataset["target_return_5d"] = target_series.pct_change(5)
    dataset["target_return_20d"] = target_series.pct_change(20)
    dataset["target_volatility_20d"] = target_series.pct_change().rolling(20).std()
    dataset["target_drawdown_20d"] = target_series / target_series.rolling(20).max() - 1.0
    
    # RSI (14-day)
    delta = target_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    dataset["target_rsi_14d"] = 100 - (100 / (1 + rs))
    dataset["target_rsi_14d"] = dataset["target_rsi_14d"].fillna(50)
    
    # MACD (12, 26, 9)
    ema_12 = target_series.ewm(span=12, adjust=False).mean()
    ema_26 = target_series.ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    dataset["target_macd_hist"] = macd_line - signal_line
    
    # Bollinger Bands Position (%B)
    ma_20 = target_series.rolling(20).mean()
    std_20 = target_series.rolling(20).std()
    upper_band = ma_20 + (std_20 * 2)
    lower_band = ma_20 - (std_20 * 2)
    band_width = upper_band - lower_band
    dataset["target_bollinger_pb"] = (target_series - lower_band) / band_width.replace(0, np.nan)
    dataset["target_bollinger_pb"] = dataset["target_bollinger_pb"].fillna(0.5)

    for column in frame.columns:
        if column == target:
            continue
        slug = _slug(column)
        series = frame[column]
        dataset[f"{slug}__return_5d"] = series.pct_change(5)
        dataset[f"{slug}__return_20d"] = series.pct_change(20)
        dataset[f"{slug}__relative_strength_20d"] = series.pct_change(20) - target_series.pct_change(20)
        
        # Cross-asset momentum
        dataset[f"{slug}__momentum_ratio"] = series.rolling(5).mean() / series.rolling(20).mean() - 1.0
    if graph_feature_frame is not None and not graph_feature_frame.empty:
        aligned_graph = graph_feature_frame.reindex(dataset.index).ffill().fillna(0.0)
    else:
        aligned_graph = pd.DataFrame(
            {column: 0.0 for column in [*GRAPH_FEATURE_COLUMNS, "graph__history_coverage_20d"]},
            index=dataset.index,
        )
    dataset = dataset.join(aligned_graph, how="left")
    return dataset


def _build_ml_dataset(feature_frame: pd.DataFrame, target_series: pd.Series, horizon_days: int) -> pd.DataFrame:
    dataset = feature_frame.copy()
    dataset["target_return_30d"] = target_series.shift(-horizon_days) / target_series - 1.0
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return dataset


def _load_graph_feature_frame(
    *,
    config: dict,
    target: str,
    graph_feature_vector: GraphFeatureVector | dict[str, Any] | None,
) -> pd.DataFrame:
    graph_frame = pd.DataFrame(columns=list(GRAPH_FEATURE_COLUMNS), dtype=float)
    if config.get("database_url"):
        try:
            with database_session(config["database_url"]) as session:
                repo = MarketRepository(session)
                stored = repo.load_daily_graph_summary_frame(target=target)
                delta_stored = repo.load_daily_graph_delta_frame(target=target)
            if not stored.empty:
                graph_frame = stored.copy()
            if not delta_stored.empty:
                graph_frame = graph_frame.join(delta_stored, how="outer") if not graph_frame.empty else delta_stored.copy()
        except Exception:
            graph_frame = pd.DataFrame(columns=list(GRAPH_FEATURE_COLUMNS), dtype=float)
    live_frame = _normalize_graph_feature_frame(graph_feature_vector)
    if not live_frame.empty:
        graph_frame = pd.concat([graph_frame, live_frame])
        graph_frame = graph_frame[~graph_frame.index.duplicated(keep="last")].sort_index()
    if graph_frame.empty:
        return pd.DataFrame(columns=list(GRAPH_FEATURE_COLUMNS) + ["graph__history_coverage_20d"], dtype=float)
    graph_frame = graph_frame.reindex(columns=list(GRAPH_FEATURE_COLUMNS), fill_value=0.0).sort_index().astype(float)
    graph_frame["graph__history_coverage_20d"] = (
        graph_frame["graph__feature_available"].rolling(window=20, min_periods=1).mean().fillna(0.0)
    )
    return graph_frame


def _normalize_graph_feature_frame(graph_feature_vector: GraphFeatureVector | dict[str, Any] | None) -> pd.DataFrame:
    if graph_feature_vector is None:
        return pd.DataFrame()
    if isinstance(graph_feature_vector, GraphFeatureVector):
        prediction_date = graph_feature_vector.prediction_date
        features = dict(graph_feature_vector.features)
    else:
        prediction_date = str(graph_feature_vector.get("prediction_date") or "")
        features = {str(key): float(value) for key, value in (graph_feature_vector.get("features") or {}).items()}
    if not prediction_date:
        return pd.DataFrame()
    row = {column: float(features.get(column, 0.0)) for column in GRAPH_FEATURE_COLUMNS}
    return pd.DataFrame([row], index=pd.to_datetime([prediction_date]))


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def run_weekly_retraining(
    repo: MarketRepository,
    *,
    target: str,
    snapshot: dict,
    config: dict,
    as_of_date: str,
    model_name: str = "hybrid_v1",
    horizon_days: int = 30,
) -> dict[str, Any]:
    scheduled_date = weekly_anchor(as_of_date)
    existing = repo.get_retraining_event(model_name=model_name, scheduled_date=scheduled_date)
    active_version = repo.latest_successful_model_version(model_name)
    if existing is not None:
        return {
            "triggered": False,
            "scheduled_date": scheduled_date,
            "event": existing,
            "active_model_version": active_version or existing.get("model_version") or model_name,
        }

    diagnostics = build_retraining_diagnostics(
        target=target,
        snapshot=snapshot,
        config=config,
        as_of_date=as_of_date,
        model_name=model_name,
        horizon_days=horizon_days,
    )
    now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    event = RetrainingEventRecord(
        retraining_event_id=f"retrain:{uuid4().hex[:12]}",
        model_name=model_name,
        model_version=diagnostics.model_version,
        scheduled_date=scheduled_date,
        started_at=now_iso,
        completed_at=now_iso,
        status="complete" if diagnostics.status == "ok" else "failed",
        training_window_start=diagnostics.training_window_start,
        training_window_end=diagnostics.training_window_end,
        training_row_count=diagnostics.training_row_count,
        notes=diagnostics.notes,
    )
    repo.upsert_retraining_event(event)
    metric_version = diagnostics.model_version or active_version or model_name
    repo.upsert_model_metrics(
        [
            ModelMetricRecord(
                model_name=model_name,
                model_version=metric_version,
                evaluation_date=as_of_date,
                horizon=horizon_days,
                metric_name="hybrid_cv_mae",
                metric_value=float(diagnostics.cv_mae),
                training_window_start=diagnostics.training_window_start,
                training_window_end=diagnostics.training_window_end,
            ),
            ModelMetricRecord(
                model_name=model_name,
                model_version=metric_version,
                evaluation_date=as_of_date,
                horizon=horizon_days,
                metric_name="hybrid_training_rows",
                metric_value=float(diagnostics.training_row_count),
                training_window_start=diagnostics.training_window_start,
                training_window_end=diagnostics.training_window_end,
            ),
        ]
    )
    promotion_decision = {
        "candidate_model_version": diagnostics.model_version,
        "candidate_cv_mae": float(diagnostics.cv_mae),
        "baseline_model_version": active_version,
        "baseline_cv_mae": None,
        "promoted": diagnostics.status == "ok",
        "reason": diagnostics.status,
    }
    active_model_version = active_version or model_name
    if diagnostics.status == "ok" and diagnostics.model_version:
        prior_metric = repo.latest_model_metric(
            model_name=model_name,
            metric_name="hybrid_cv_mae",
            horizon=horizon_days,
            exclude_model_version=diagnostics.model_version,
        )
        promotion_decision["baseline_cv_mae"] = None if prior_metric is None else float(prior_metric["metric_value"])
        if prior_metric is None or float(diagnostics.cv_mae) <= float(prior_metric["metric_value"]):
            active_model_version = diagnostics.model_version
            promotion_decision["promoted"] = True
            promotion_decision["reason"] = "improved_or_first_model"
        else:
            promotion_decision["promoted"] = False
            promotion_decision["reason"] = "candidate_underperformed_active_model"
    elif diagnostics.status != "ok":
        promotion_decision["promoted"] = False
        promotion_decision["reason"] = diagnostics.status
    return {
        "triggered": True,
        "scheduled_date": scheduled_date,
        "event": asdict(event),
        "active_model_version": active_model_version,
        "diagnostics": diagnostics.to_dict(),
        "promotion_decision": promotion_decision,
    }


def build_retraining_diagnostics(
    *,
    target: str,
    snapshot: dict,
    config: dict,
    as_of_date: str,
    model_name: str = "hybrid_v1",
    horizon_days: int = 30,
) -> RetrainingDiagnostics:
    frame = _load_history_frame(target=target, snapshot=snapshot, config=config)
    if target not in frame.columns or len(frame) < 90:
        return RetrainingDiagnostics(
            model_name=model_name,
            model_version=None,
            scheduled_date=weekly_anchor(as_of_date),
            status="insufficient_history",
            training_window_start=None,
            training_window_end=None,
            training_row_count=0,
            cv_mae=0.0,
            notes="Not enough stored history to retrain the hybrid overlay.",
        )

    graph_frame = _load_graph_feature_frame(config=config, target=target, graph_feature_vector=None)
    feature_frame = _build_feature_frame(frame, target=target, graph_feature_frame=graph_frame)
    dataset = _build_ml_dataset(feature_frame, frame[target], horizon_days=horizon_days)
    if len(dataset) < 60:
        return RetrainingDiagnostics(
            model_name=model_name,
            model_version=None,
            scheduled_date=weekly_anchor(as_of_date),
            status="insufficient_features",
            training_window_start=None,
            training_window_end=None,
            training_row_count=len(dataset),
            cv_mae=0.0,
            notes="Not enough ML rows to retrain the hybrid overlay.",
        )

    overlay = build_hybrid_ml_overlay(target=target, snapshot=snapshot, config=config, horizon_days=horizon_days)
    return RetrainingDiagnostics(
        model_name=model_name,
        model_version=f"{model_name}:{as_of_date}",
        scheduled_date=weekly_anchor(as_of_date),
        status=overlay.status,
        training_window_start=str(dataset.index.min().date()),
        training_window_end=str(dataset.index.max().date()),
        training_row_count=len(dataset),
        cv_mae=float(overlay.cv_mae),
        notes=None if overlay.status == "ok" else f"Hybrid overlay returned status={overlay.status}.",
    )


def weekly_anchor(as_of_date: str) -> str:
    current = datetime.fromisoformat(as_of_date).date()
    return (current - timedelta(days=current.weekday())).isoformat()
