from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import sqrt
from uuid import uuid4

import numpy as np
import pandas as pd

from .modeling import ModelArtifacts
from .storage.models import (
    EvaluationRunRecord,
    ForecastOutcomeRecord,
    ModelMetricRecord,
    SectorOutcomeMetricRecord,
)
from .storage.repositories import MarketRepository


@dataclass(frozen=True)
class EvaluationResult:
    evaluation_run_id: str
    evaluation_date: str
    horizon_days: int
    eligible_run_count: int
    completed_run_count: int
    insufficient_run_count: int
    market_outcomes: list[ForecastOutcomeRecord]
    sector_outcomes: list[SectorOutcomeMetricRecord]
    metrics_written: int


def evaluate_mature_forecasts(
    repo: MarketRepository,
    *,
    as_of_date: str,
    horizon_days: int = 30,
    tolerance_days: int = 3,
) -> EvaluationResult:
    evaluation_run_id = f"eval:{as_of_date}:{horizon_days}:{uuid4().hex[:8]}"
    now_iso = _utc_now_iso()
    repo.create_evaluation_run(
        EvaluationRunRecord(
            evaluation_run_id=evaluation_run_id,
            evaluation_date=as_of_date,
            horizon_days=horizon_days,
            status="running",
            created_at=now_iso,
        )
    )

    price_history = repo.load_price_history_frame()
    mature_runs = repo.list_mature_prediction_runs(as_of_date=as_of_date, horizon_days=horizon_days)
    market_outcomes: list[ForecastOutcomeRecord] = []
    sector_outcomes: list[SectorOutcomeMetricRecord] = []

    for run in mature_runs:
        projected_paths = repo.load_projected_path_rows(run["run_id"], scenario_type="base")
        sector_rows = repo.load_sector_outlook_rows(run["run_id"])
        market_outcome = _score_market_run(
            run=run,
            projected_paths=projected_paths,
            price_history=price_history,
            evaluation_run_id=evaluation_run_id,
            evaluated_at=now_iso,
            horizon_days=horizon_days,
            tolerance_days=tolerance_days,
        )
        market_outcomes.append(market_outcome)
        sector_outcomes.extend(
            _score_sector_rows(
                run=run,
                sector_rows=sector_rows,
                price_history=price_history,
                evaluation_run_id=evaluation_run_id,
                evaluated_at=now_iso,
                horizon_days=horizon_days,
                tolerance_days=tolerance_days,
            )
        )

    repo.upsert_forecast_outcomes(market_outcomes)
    repo.upsert_sector_outcome_metrics(sector_outcomes)

    metrics = _build_metric_records(
        repo=repo,
        evaluation_date=as_of_date,
        horizon_days=horizon_days,
    )
    repo.upsert_model_metrics(metrics)

    completed_run_count = sum(1 for row in market_outcomes if row.outcome_status == "complete")
    insufficient_run_count = sum(1 for row in market_outcomes if row.outcome_status != "complete")
    repo.complete_evaluation_run(
        evaluation_run_id,
        status="complete",
        eligible_run_count=len(mature_runs),
        completed_run_count=completed_run_count,
        completed_at=now_iso,
        notes=f"Scored {completed_run_count} runs; {insufficient_run_count} had insufficient realized data.",
    )

    return EvaluationResult(
        evaluation_run_id=evaluation_run_id,
        evaluation_date=as_of_date,
        horizon_days=horizon_days,
        eligible_run_count=len(mature_runs),
        completed_run_count=completed_run_count,
        insufficient_run_count=insufficient_run_count,
        market_outcomes=market_outcomes,
        sector_outcomes=sector_outcomes,
        metrics_written=len(metrics),
    )


def build_walk_forward_accuracy(
    dates: pd.Series,
    y_true: pd.Series,
    model_artifacts: dict[str, ModelArtifacts],
    window: int = 12,
) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dates.iloc[:-1].reset_index(drop=True), "actual": y_true.iloc[:-1].reset_index(drop=True)})
    for name, artifact in model_artifacts.items():
        preds = pd.Series(artifact.oof_predictions)
        correctness = (preds == frame["actual"]).astype(float)
        correctness[preds.isna()] = np.nan
        frame[name] = correctness.rolling(window=window, min_periods=3).mean()
        frame[f"{name}_prob"] = artifact.oof_probabilities
    return frame


def top_feature_importance(model_artifacts: dict[str, ModelArtifacts], limit: int = 15) -> list[str]:
    aggregate: dict[str, float] = {}
    for artifact in model_artifacts.values():
        for feature, importance in artifact.feature_importance.items():
            aggregate[feature] = aggregate.get(feature, 0.0) + importance
    return [name for name, _ in sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:limit]]


def correlation_heatmap_data(dataset: pd.DataFrame, feature_columns: list[str], limit: int = 20) -> pd.DataFrame:
    selected = feature_columns[:limit]
    return dataset[selected].corr().fillna(0.0)


def _score_market_run(
    *,
    run: dict,
    projected_paths: list[dict],
    price_history: pd.DataFrame,
    evaluation_run_id: str,
    evaluated_at: str,
    horizon_days: int,
    tolerance_days: int,
) -> ForecastOutcomeRecord:
    target = run["target"]
    horizon_row = next((row for row in projected_paths if row["horizon_day"] == horizon_days), None)
    if horizon_row is None:
        return ForecastOutcomeRecord(
            run_id=run["run_id"],
            model_version=run.get("model_version"),
            evaluation_run_id=evaluation_run_id,
            target=target,
            forecast_start_date=run["run_date"],
            forecast_target_date=run.get("forecast_target_date", run["run_date"]),
            horizon_days=horizon_days,
            outcome_status="insufficient_data",
            run_health=run.get("run_health"),
            evaluated_at=evaluated_at,
        )

    latest_price = _resolve_start_price(price_history, target, run["run_date"], projected_paths)
    actual_price = _resolve_price_on_or_after(price_history, target, horizon_row["forecast_date"], tolerance_days=tolerance_days)
    if latest_price is None or actual_price is None:
        return ForecastOutcomeRecord(
            run_id=run["run_id"],
            model_version=run.get("model_version"),
            evaluation_run_id=evaluation_run_id,
            target=target,
            forecast_start_date=run["run_date"],
            forecast_target_date=horizon_row["forecast_date"],
            horizon_days=horizon_days,
            latest_price=latest_price,
            predicted_price=horizon_row.get("predicted_price"),
            predicted_return=horizon_row.get("predicted_return"),
            run_health=run.get("run_health"),
            outcome_status="insufficient_data",
            evaluated_at=evaluated_at,
        )

    actual_return = float(actual_price / latest_price - 1.0)
    predicted_return = float(horizon_row.get("predicted_return") or 0.0)
    prediction_error = actual_return - predicted_return
    band_hit = int(
        horizon_row.get("lower_band") is not None
        and horizon_row.get("upper_band") is not None
        and float(horizon_row["lower_band"]) <= actual_price <= float(horizon_row["upper_band"])
    )
    return ForecastOutcomeRecord(
        run_id=run["run_id"],
        model_version=run.get("model_version"),
        evaluation_run_id=evaluation_run_id,
        target=target,
        forecast_start_date=run["run_date"],
        forecast_target_date=horizon_row["forecast_date"],
        horizon_days=horizon_days,
        latest_price=round(float(latest_price), 6),
        predicted_price=horizon_row.get("predicted_price"),
        actual_price=round(float(actual_price), 6),
        predicted_return=round(predicted_return, 6),
        actual_return=round(actual_return, 6),
        predicted_direction_label=_direction_label(predicted_return),
        actual_direction_label=_direction_label(actual_return),
        prediction_error=round(prediction_error, 6),
        absolute_error=round(abs(prediction_error), 6),
        band_hit_flag=band_hit,
        run_health=run.get("run_health"),
        outcome_status="complete",
        evaluated_at=evaluated_at,
    )


def _score_sector_rows(
    *,
    run: dict,
    sector_rows: list[dict],
    price_history: pd.DataFrame,
    evaluation_run_id: str,
    evaluated_at: str,
    horizon_days: int,
    tolerance_days: int,
) -> list[SectorOutcomeMetricRecord]:
    forecast_target_date = run.get("forecast_target_date") or _forecast_target_date_from_sectors(sector_rows) or run["run_date"]
    scored: list[dict] = []
    for index, row in enumerate(sorted(sector_rows, key=lambda item: float(item.get("ranking_score") or 0.0), reverse=True), start=1):
        start_price = _resolve_start_price(price_history, row["sector_symbol"], run["run_date"], [])
        actual_price = _resolve_price_on_or_after(
            price_history,
            row["sector_symbol"],
            forecast_target_date,
            tolerance_days=tolerance_days,
        )
        if start_price is None or actual_price is None:
            scored.append(
                {
                    "row": row,
                    "predicted_rank": index,
                    "actual_return": None,
                }
            )
            continue
        actual_return = float(actual_price / start_price - 1.0)
        scored.append(
            {
                "row": row,
                "predicted_rank": index,
                "actual_return": actual_return,
            }
        )

    complete_rows = [item for item in scored if item["actual_return"] is not None]
    actual_rank_map: dict[str, int] = {}
    for index, item in enumerate(sorted(complete_rows, key=lambda payload: payload["actual_return"], reverse=True), start=1):
        actual_rank_map[item["row"]["sector_symbol"]] = index

    records: list[SectorOutcomeMetricRecord] = []
    for item in scored:
        row = item["row"]
        predicted_rank = item["predicted_rank"]
        actual_return = item["actual_return"]
        actual_rank = actual_rank_map.get(row["sector_symbol"])
        recommendation = row.get("recommendation_label")
        if actual_return is None or actual_rank is None:
            records.append(
                SectorOutcomeMetricRecord(
                    run_id=run["run_id"],
                    model_version=run.get("model_version"),
                    evaluation_run_id=evaluation_run_id,
                    sector_symbol=row["sector_symbol"],
                    forecast_start_date=run["run_date"],
                    forecast_target_date=forecast_target_date,
                    horizon_days=horizon_days,
                    predicted_rank_score=row.get("ranking_score"),
                    predicted_return_30d=row.get("expected_return_30d"),
                    outcome_status="insufficient_data",
                    evaluated_at=evaluated_at,
                )
            )
            continue

        top_bucket_hit_flag = int(predicted_rank <= 3 and actual_rank <= 3)
        favor_hit_flag = None
        if recommendation == "FAVOR":
            favor_hit_flag = int(actual_return > 0)
        elif recommendation == "AVOID":
            favor_hit_flag = int(actual_return < 0)

        records.append(
            SectorOutcomeMetricRecord(
                run_id=run["run_id"],
                model_version=run.get("model_version"),
                evaluation_run_id=evaluation_run_id,
                sector_symbol=row["sector_symbol"],
                forecast_start_date=run["run_date"],
                forecast_target_date=forecast_target_date,
                horizon_days=horizon_days,
                predicted_rank_score=row.get("ranking_score"),
                predicted_return_30d=row.get("expected_return_30d"),
                actual_return_30d=round(float(actual_return), 6),
                actual_rank=actual_rank,
                top_bucket_hit_flag=top_bucket_hit_flag,
                favor_hit_flag=favor_hit_flag,
                rank_error=float(actual_rank - predicted_rank),
                outcome_status="complete",
                evaluated_at=evaluated_at,
            )
        )
    return records


def _build_metric_records(
    *,
    repo: MarketRepository,
    evaluation_date: str,
    horizon_days: int,
) -> list[ModelMetricRecord]:
    market_rows = pd.DataFrame(repo.load_forecast_outcomes(horizon_days=horizon_days, outcome_status="complete"))
    sector_rows = pd.DataFrame(repo.load_sector_outcome_metrics(horizon_days=horizon_days, outcome_status="complete"))
    metrics: list[ModelMetricRecord] = []

    if not market_rows.empty:
        for model_version, frame in market_rows.groupby(market_rows["model_version"].fillna("unknown")):
            pred = frame["predicted_return"].astype(float)
            actual = frame["actual_return"].astype(float)
            errors = actual - pred
            direction_hit_rate = float((frame["predicted_direction_label"] == frame["actual_direction_label"]).mean())
            band_values = frame["band_hit_flag"].dropna().astype(float)
            metrics.extend(
                [
                    ModelMetricRecord("market_forecast", str(model_version), evaluation_date, horizon_days, "market_direction_hit_rate", direction_hit_rate),
                    ModelMetricRecord("market_forecast", str(model_version), evaluation_date, horizon_days, "market_return_mae", float(errors.abs().mean())),
                    ModelMetricRecord("market_forecast", str(model_version), evaluation_date, horizon_days, "market_return_rmse", float(sqrt((errors**2).mean()))),
                    ModelMetricRecord("market_forecast", str(model_version), evaluation_date, horizon_days, "market_return_bias", float(errors.mean())),
                    ModelMetricRecord(
                        "market_forecast",
                        str(model_version),
                        evaluation_date,
                        horizon_days,
                        "market_band_hit_rate",
                        float(band_values.mean()) if not band_values.empty else 0.0,
                    ),
                ]
            )

    if not sector_rows.empty:
        for model_version, frame in sector_rows.groupby(sector_rows["model_version"].fillna("unknown")):
            rank_error = frame["rank_error"].dropna().astype(float)
            favor_frame = frame[frame["favor_hit_flag"].notna()]
            top3_frame = frame[frame["top_bucket_hit_flag"].notna()]
            spearman = (
                frame["predicted_rank_score"].astype(float).corr(frame["actual_return_30d"].astype(float), method="spearman")
                if len(frame) > 1
                else 0.0
            )
            metrics.extend(
                [
                    ModelMetricRecord(
                        "sector_ranking",
                        str(model_version),
                        evaluation_date,
                        horizon_days,
                        "sector_top3_hit_rate",
                        float(top3_frame["top_bucket_hit_flag"].astype(float).mean()) if not top3_frame.empty else 0.0,
                    ),
                    ModelMetricRecord(
                        "sector_ranking",
                        str(model_version),
                        evaluation_date,
                        horizon_days,
                        "sector_favor_precision",
                        float(favor_frame["favor_hit_flag"].astype(float).mean()) if not favor_frame.empty else 0.0,
                    ),
                    ModelMetricRecord(
                        "sector_ranking",
                        str(model_version),
                        evaluation_date,
                        horizon_days,
                        "sector_rank_error_mean",
                        float(rank_error.abs().mean()) if not rank_error.empty else 0.0,
                    ),
                    ModelMetricRecord(
                        "sector_ranking",
                        str(model_version),
                        evaluation_date,
                        horizon_days,
                        "sector_spearman_rank_corr",
                        float(spearman) if pd.notna(spearman) else 0.0,
                    ),
                ]
            )
    return metrics


def _resolve_start_price(
    price_history: pd.DataFrame,
    symbol: str,
    forecast_start_date: str,
    projected_paths: list[dict],
) -> float | None:
    resolved = _resolve_price_on_or_before(price_history, symbol, forecast_start_date)
    if resolved is not None:
        return resolved
    first_path = projected_paths[0] if projected_paths else None
    if not first_path:
        return None
    predicted_return = float(first_path.get("predicted_return") or 0.0)
    predicted_price = first_path.get("predicted_price")
    if predicted_price is None or (1.0 + predicted_return) == 0.0:
        return None
    return float(predicted_price) / (1.0 + predicted_return)


def _resolve_price_on_or_before(price_history: pd.DataFrame, symbol: str, target_date: str) -> float | None:
    if symbol not in price_history.columns:
        return None
    series = price_history[symbol].dropna()
    if series.empty:
        return None
    cutoff = pd.Timestamp(target_date)
    eligible = series.loc[series.index <= cutoff]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def _resolve_price_on_or_after(price_history: pd.DataFrame, symbol: str, target_date: str, *, tolerance_days: int) -> float | None:
    if symbol not in price_history.columns:
        return None
    series = price_history[symbol].dropna()
    if series.empty:
        return None
    lower = pd.Timestamp(target_date)
    upper = lower + timedelta(days=tolerance_days)
    eligible = series.loc[(series.index >= lower) & (series.index <= upper)]
    if eligible.empty:
        return None
    return float(eligible.iloc[0])


def _forecast_target_date_from_sectors(rows: list[dict]) -> str | None:
    del rows
    return None


def _direction_label(value: float, neutral_band: float = 0.002) -> str:
    if value > neutral_band:
        return "UP"
    if value < -neutral_band:
        return "DOWN"
    return "NEUTRAL"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
