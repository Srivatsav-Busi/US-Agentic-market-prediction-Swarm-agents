from __future__ import annotations

import json
from pathlib import Path

from ..evaluation import evaluate_mature_forecasts
from ..config import list_tracked_instruments
from ..data_sources.historical_prices import (
    fetch_historical_market_data,
    fetch_macro_series_history,
    tracked_instruments,
    utc_now_iso,
)
from ..feature_store import build_feature_snapshots
from ..storage.db import create_schema, database_session
from ..storage.models import DailyGraphDeltaRecord, DailyGraphSummaryRecord, DailyPriceRecord, MacroObservationRecord, NewsEvidenceRecord
from ..storage.repositories import MarketRepository


def bootstrap_history(
    config: dict,
    years: int = 2,
    target: str = "S&P 500",
    diagnostics_path: str | Path | None = None,
) -> dict:
    history, price_diagnostics = fetch_historical_market_data(config, years=years)
    macro_history, macro_diagnostics = fetch_macro_series_history(config, years=years)

    with database_session(config["database_url"]) as connection:
        create_schema(connection)
        repo = MarketRepository(connection)
        instrument_ids = repo.upsert_instruments(tracked_instruments(config))

        price_rows: list[DailyPriceRecord] = []
        for label, rows in history.items():
            symbol = _lookup_symbol(config, label)
            instrument_id = instrument_ids[symbol]
            for row in rows:
                price_rows.append(
                    DailyPriceRecord(
                        trade_date=row["date"],
                        instrument_id=instrument_id,
                        close=float(row["value"]),
                        source=str(row.get("provider", "unknown")),
                        ingestion_timestamp=utc_now_iso(),
                        adjusted_close=float(row["value"]),
                        proxy_used=1 if row.get("proxy_for") else 0,
                    )
                )
        repo.upsert_daily_prices(price_rows)

        macro_rows: list[MacroObservationRecord] = []
        for series_name, rows in macro_history.items():
            frequency = config.get("macro_series", {}).get(series_name, {}).get("frequency", "unknown")
            for row in rows:
                macro_rows.append(
                    MacroObservationRecord(
                        series_name=series_name,
                        observation_date=row["date"],
                        value=float(row["value"]),
                        release_date=row["date"],
                        source=str(row.get("provider", "fred")),
                        frequency=frequency,
                    )
                )
        repo.upsert_macro_series(macro_rows)

        features = build_feature_snapshots(history, evidence_items=[], target_scope=target, run_id=f"bootstrap:{target}:{years}")
        repo.upsert_feature_snapshots(features)

        diagnostics = {
            "database_url": config["database_url"],
            "years": years,
            "target": target,
            "instrument_count": repo.table_count("instruments"),
            "daily_price_count": repo.table_count("daily_prices"),
            "macro_series_count": repo.table_count("macro_series"),
            "feature_snapshot_count": repo.table_count("daily_feature_snapshot"),
            "price_diagnostics": price_diagnostics,
            "macro_diagnostics": macro_diagnostics,
        }

    if diagnostics_path:
        path = Path(diagnostics_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics


def persist_daily_run(
    *,
    config: dict,
    prediction_date: str,
    snapshot: dict,
    items: list,
    target: str,
    artifacts: dict | None = None,
    graph_feature_vector: dict | None = None,
    graph_delta_summary: dict | None = None,
    run_evaluation: bool = False,
) -> dict:
    latest_history = {label: rows for label, rows in snapshot.get("history", {}).items() if rows}
    with database_session(config["database_url"]) as connection:
        create_schema(connection)
        repo = MarketRepository(connection)
        instrument_ids = repo.upsert_instruments(tracked_instruments(config))

        prices: list[DailyPriceRecord] = []
        for label, values in snapshot.get("series", {}).items():
            symbol = _lookup_symbol(config, label)
            instrument_id = instrument_ids[symbol]
            latest_value = values.get("latest")
            if latest_value is None:
                continue
            prices.append(
                DailyPriceRecord(
                    trade_date=prediction_date,
                    instrument_id=instrument_id,
                    close=float(latest_value),
                    adjusted_close=float(latest_value),
                    source=str(values.get("provider", "snapshot")),
                    proxy_used=1 if values.get("proxy_for") else 0,
                    ingestion_timestamp=utc_now_iso(),
                )
            )
        repo.upsert_daily_prices(prices)

        evidence = [
            NewsEvidenceRecord(
                evidence_id=item.id,
                run_date=prediction_date,
                published_at=item.published_at,
                source=item.source,
                category=item.category,
                title=item.title,
                summary=item.summary,
                url=item.url,
                sentiment_score=float(item.impact_score),
                impact_score=float(item.impact_score),
                freshness_score=float(item.freshness_score),
                credibility_score=float(item.credibility_score),
                duplicate_cluster=item.duplicate_cluster,
            )
            for item in items
        ]
        repo.upsert_news_evidence(evidence)

        feature_snapshots = build_feature_snapshots(latest_history, items, target_scope=target, run_id=f"daily:{prediction_date}:{target}")
        repo.upsert_feature_snapshots(feature_snapshots)
        graph_summary_count = 0
        graph_delta_count = 0
        if graph_feature_vector and graph_feature_vector.get("features"):
            graph_rows = [
                DailyGraphSummaryRecord(
                    prediction_date=prediction_date,
                    target=target,
                    feature_name=feature_name,
                    feature_value=float(feature_value),
                    feature_group=str((graph_feature_vector.get("feature_groups") or {}).get(feature_name, "graph")),
                    schema_version=str(graph_feature_vector.get("schema_version") or "graph_feature_vector:v1"),
                    generation_run_id=str(graph_feature_vector.get("generation_run_id") or f"daily:{prediction_date}:{target}"),
                )
                for feature_name, feature_value in (graph_feature_vector.get("features") or {}).items()
                if not str(feature_name).startswith("graph_delta__")
            ]
            repo.upsert_daily_graph_summaries(graph_rows)
            graph_summary_count = len(graph_rows)
        if graph_delta_summary and graph_delta_summary.get("features"):
            graph_delta_rows = [
                DailyGraphDeltaRecord(
                    prediction_date=prediction_date,
                    target=target,
                    feature_name=feature_name,
                    feature_value=float(feature_value),
                    feature_group=str((graph_delta_summary.get("feature_groups") or {}).get(feature_name, "delta")),
                    schema_version=str(graph_delta_summary.get("schema_version") or "graph_delta_summary:v1"),
                    generation_run_id=str(graph_delta_summary.get("generation_run_id") or f"daily:{prediction_date}:{target}"),
                )
                for feature_name, feature_value in (graph_delta_summary.get("features") or {}).items()
            ]
            repo.upsert_daily_graph_deltas(graph_delta_rows)
            graph_delta_count = len(graph_delta_rows)

        projected_path_count = 0
        sector_outlook_count = 0
        evaluation_summary = None
        if artifacts:
            repo.upsert_prediction_run(
                {
                    "run_id": artifacts.get("run_id"),
                    "run_date": prediction_date,
                    "target": target,
                    "prediction_label": artifacts.get("prediction_label"),
                    "confidence": artifacts.get("confidence"),
                    "run_health": artifacts.get("run_health"),
                    "expected_return": artifacts.get("forecast_summary", {}).get("expected_return_30d"),
                    "expected_volatility": artifacts.get("forecast_summary", {}).get("expected_volatility_30d"),
                    "posterior_up": artifacts.get("posterior_probabilities", {}).get("UP"),
                    "posterior_neutral": artifacts.get("posterior_probabilities", {}).get("NEUTRAL"),
                    "posterior_down": artifacts.get("posterior_probabilities", {}).get("DOWN"),
                    "model_version": artifacts.get("model_version")
                    or artifacts.get("ensemble_diagnostics", {}).get("mode")
                    or "baseline-30d-v1",
                    "feature_snapshot_version": artifacts.get("feature_snapshot_version"),
                    "model_stack_version": artifacts.get("model_stack_version"),
                    "calibration_version": artifacts.get("calibration_version"),
                    "regime_slice": artifacts.get("regime_slice"),
                    "agreement_score": artifacts.get("agreement_features", {}).get("agreement_score"),
                    "pipeline_stage_status": artifacts.get("pipeline_stage_status"),
                    "stage_diagnostics": artifacts.get("stage_diagnostics"),
                    "created_at": utc_now_iso(),
                }
            )
            path_rows = []
            projection = artifacts.get("market_projection", {})
            for scenario_type in ("base", "bull", "bear"):
                for point in projection.get(scenario_type, []):
                    path_rows.append(
                        {
                            "run_id": artifacts.get("run_id"),
                            "forecast_date": point["forecast_date"],
                            "horizon_day": point["horizon_day"],
                            "target_symbol": point.get("target_symbol", target),
                            "predicted_price": point.get("predicted_price"),
                            "predicted_return": point.get("predicted_return"),
                            "lower_band": point.get("lower_band"),
                            "upper_band": point.get("upper_band"),
                            "scenario_type": scenario_type,
                        }
                    )
            repo.upsert_projected_paths(path_rows)
            projected_path_count = len(path_rows)

            sector_rows = [
                {
                    "run_id": artifacts.get("run_id"),
                    **row,
                }
                for row in artifacts.get("sector_outlook", [])
            ]
            repo.upsert_sector_outlooks(sector_rows)
            sector_outlook_count = len(sector_rows)
            if run_evaluation:
                evaluation_result = evaluate_mature_forecasts(
                    repo,
                    as_of_date=prediction_date,
                    horizon_days=int(artifacts.get("forecast_summary", {}).get("horizon_days", 30) or 30),
                )
                evaluation_summary = {
                    "evaluation_run_id": evaluation_result.evaluation_run_id,
                    "eligible_run_count": evaluation_result.eligible_run_count,
                    "completed_run_count": evaluation_result.completed_run_count,
                    "insufficient_run_count": evaluation_result.insufficient_run_count,
                    "metrics_written": evaluation_result.metrics_written,
                }

        return {
            "daily_price_count": len(prices),
            "evidence_count": len(evidence),
            "feature_snapshot_count": len(feature_snapshots),
            "graph_summary_count": graph_summary_count,
            "graph_delta_count": graph_delta_count,
            "projected_path_count": projected_path_count,
            "sector_outlook_count": sector_outlook_count,
            "evaluation": evaluation_summary,
        }


def _lookup_symbol(config: dict, label: str) -> str:
    if label in config.get("market_symbols", {}):
        return config["market_symbols"][label]
    if label in config.get("sector_symbols", {}):
        return config["sector_symbols"][label]
    raise KeyError(f"No configured symbol for label: {label}")
