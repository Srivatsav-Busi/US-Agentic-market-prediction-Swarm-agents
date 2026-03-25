from __future__ import annotations

import json

from sqlalchemy import func, select
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy.orm import Session

from .models import (
    AnalysisReportModel,
    AnalysisReportRecord,
    AnalysisReportTaskModel,
    AnalysisReportTaskRecord,
    DailyGraphDeltaModel,
    DailyGraphDeltaRecord,
    DailyGraphSummaryModel,
    DailyGraphSummaryRecord,
    DailyPriceModel,
    DailyPriceRecord,
    DailyPredictionRunModel,
    EvaluationRunModel,
    EvaluationRunRecord,
    FeatureSnapshotModel,
    FeatureSnapshotRecord,
    ForecastOutcomeModel,
    ForecastOutcomeRecord,
    GraphBuildTaskModel,
    GraphBuildTaskRecord,
    GraphProjectModel,
    GraphProjectRecord,
    GraphSnapshotModel,
    GraphSnapshotRecord,
    InstrumentModel,
    InstrumentRecord,
    MacroObservationRecord,
    MacroSeriesModel,
    ModelMetricModel,
    ModelMetricRecord,
    NewsEvidenceModel,
    NewsEvidenceRecord,
    ProjectedPathModel,
    RetrainingEventModel,
    RetrainingEventRecord,
    SchedulerRunModel,
    SchedulerRunRecord,
    SectorOutcomeMetricModel,
    SectorOutcomeMetricRecord,
    SectorOutlookModel,
)
import pandas as pd


class MarketRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert_instruments(self, instruments: list[InstrumentRecord]) -> dict[str, int]:
        for instrument in instruments:
            statement = insert(InstrumentModel).values(
                symbol=instrument.symbol,
                display_name=instrument.display_name,
                asset_class=instrument.asset_class,
                category=instrument.category,
                source_priority=instrument.source_priority,
                proxy_symbol=instrument.proxy_symbol,
                active_flag=instrument.active_flag,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[InstrumentModel.symbol],
                    set_={
                        "display_name": instrument.display_name,
                        "asset_class": instrument.asset_class,
                        "category": instrument.category,
                        "source_priority": instrument.source_priority,
                        "proxy_symbol": instrument.proxy_symbol,
                        "active_flag": instrument.active_flag,
                    },
                )
            )
        self.session.flush()
        rows = self.session.execute(select(InstrumentModel.instrument_id, InstrumentModel.symbol)).all()
        return {row.symbol: int(row.instrument_id) for row in rows}

    def upsert_daily_prices(self, prices: list[DailyPriceRecord]) -> None:
        for price in prices:
            statement = insert(DailyPriceModel).values(
                trade_date=price.trade_date,
                instrument_id=price.instrument_id,
                open=price.open,
                high=price.high,
                low=price.low,
                close=price.close,
                adjusted_close=price.adjusted_close,
                volume=price.volume,
                source=price.source,
                proxy_used=price.proxy_used,
                ingestion_timestamp=price.ingestion_timestamp,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[DailyPriceModel.trade_date, DailyPriceModel.instrument_id],
                    set_={
                        "open": price.open,
                        "high": price.high,
                        "low": price.low,
                        "close": price.close,
                        "adjusted_close": price.adjusted_close,
                        "volume": price.volume,
                        "source": price.source,
                        "proxy_used": price.proxy_used,
                        "ingestion_timestamp": price.ingestion_timestamp,
                    },
                )
            )
        self.session.flush()

    def upsert_macro_series(self, observations: list[MacroObservationRecord]) -> None:
        for row in observations:
            statement = insert(MacroSeriesModel).values(
                series_name=row.series_name,
                observation_date=row.observation_date,
                value=row.value,
                release_date=row.release_date,
                source=row.source,
                frequency=row.frequency,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[MacroSeriesModel.series_name, MacroSeriesModel.observation_date],
                    set_={
                        "value": row.value,
                        "release_date": row.release_date,
                        "source": row.source,
                        "frequency": row.frequency,
                    },
                )
            )
        self.session.flush()

    def upsert_news_evidence(self, evidence_rows: list[NewsEvidenceRecord]) -> None:
        for row in evidence_rows:
            statement = insert(NewsEvidenceModel).values(
                evidence_id=row.evidence_id,
                run_date=row.run_date,
                published_at=row.published_at,
                source=row.source,
                category=row.category,
                title=row.title,
                summary=row.summary,
                url=row.url,
                sentiment_score=row.sentiment_score,
                impact_score=row.impact_score,
                freshness_score=row.freshness_score,
                credibility_score=row.credibility_score,
                duplicate_cluster=row.duplicate_cluster,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[NewsEvidenceModel.evidence_id],
                    set_={
                        "run_date": row.run_date,
                        "published_at": row.published_at,
                        "source": row.source,
                        "category": row.category,
                        "title": row.title,
                        "summary": row.summary,
                        "url": row.url,
                        "sentiment_score": row.sentiment_score,
                        "impact_score": row.impact_score,
                        "freshness_score": row.freshness_score,
                        "credibility_score": row.credibility_score,
                        "duplicate_cluster": row.duplicate_cluster,
                    },
                )
            )
        self.session.flush()

    def upsert_feature_snapshots(self, features: list[FeatureSnapshotRecord]) -> None:
        for row in features:
            statement = insert(FeatureSnapshotModel).values(
                snapshot_date=row.snapshot_date,
                feature_name=row.feature_name,
                feature_value=row.feature_value,
                feature_group=row.feature_group,
                target_scope=row.target_scope,
                generation_run_id=row.generation_run_id,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        FeatureSnapshotModel.snapshot_date,
                        FeatureSnapshotModel.feature_name,
                        FeatureSnapshotModel.target_scope,
                    ],
                    set_={
                        "feature_value": row.feature_value,
                        "feature_group": row.feature_group,
                        "generation_run_id": row.generation_run_id,
                    },
                )
            )
        self.session.flush()

    def upsert_daily_graph_summaries(self, rows: list[DailyGraphSummaryRecord]) -> None:
        for row in rows:
            statement = insert(DailyGraphSummaryModel).values(
                prediction_date=row.prediction_date,
                target=row.target,
                feature_name=row.feature_name,
                feature_value=row.feature_value,
                feature_group=row.feature_group,
                schema_version=row.schema_version,
                generation_run_id=row.generation_run_id,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        DailyGraphSummaryModel.prediction_date,
                        DailyGraphSummaryModel.target,
                        DailyGraphSummaryModel.feature_name,
                    ],
                    set_={
                        "feature_value": row.feature_value,
                        "feature_group": row.feature_group,
                        "schema_version": row.schema_version,
                        "generation_run_id": row.generation_run_id,
                    },
                )
            )
        self.session.flush()

    def upsert_daily_graph_deltas(self, rows: list[DailyGraphDeltaRecord]) -> None:
        for row in rows:
            statement = insert(DailyGraphDeltaModel).values(
                prediction_date=row.prediction_date,
                target=row.target,
                feature_name=row.feature_name,
                feature_value=row.feature_value,
                feature_group=row.feature_group,
                schema_version=row.schema_version,
                generation_run_id=row.generation_run_id,
            )
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        DailyGraphDeltaModel.prediction_date,
                        DailyGraphDeltaModel.target,
                        DailyGraphDeltaModel.feature_name,
                    ],
                    set_={
                        "feature_value": row.feature_value,
                        "feature_group": row.feature_group,
                        "schema_version": row.schema_version,
                        "generation_run_id": row.generation_run_id,
                    },
                )
            )
        self.session.flush()

    def table_count(self, table_name: str) -> int:
        model_map = {
            "instruments": InstrumentModel,
            "daily_prices": DailyPriceModel,
            "macro_series": MacroSeriesModel,
            "news_evidence": NewsEvidenceModel,
            "daily_feature_snapshot": FeatureSnapshotModel,
            "daily_graph_summary": DailyGraphSummaryModel,
            "daily_graph_delta": DailyGraphDeltaModel,
            "daily_prediction_runs": DailyPredictionRunModel,
            "projected_paths": ProjectedPathModel,
            "sector_outlook": SectorOutlookModel,
            "model_metrics": ModelMetricModel,
            "evaluation_runs": EvaluationRunModel,
            "forecast_outcomes": ForecastOutcomeModel,
            "sector_outcome_metrics": SectorOutcomeMetricModel,
            "scheduler_runs": SchedulerRunModel,
            "retraining_events": RetrainingEventModel,
            "graph_projects": GraphProjectModel,
            "graph_build_tasks": GraphBuildTaskModel,
            "graph_snapshots": GraphSnapshotModel,
            "analysis_reports": AnalysisReportModel,
            "analysis_report_tasks": AnalysisReportTaskModel,
        }
        model = model_map[table_name]
        return int(self.session.scalar(select(func.count()).select_from(model)) or 0)

    def load_price_history_frame(self) -> pd.DataFrame:
        rows = self.session.execute(
            select(InstrumentModel.display_name, DailyPriceModel.trade_date, DailyPriceModel.close)
            .join(DailyPriceModel, DailyPriceModel.instrument_id == InstrumentModel.instrument_id)
            .order_by(DailyPriceModel.trade_date.asc())
        ).all()
        if not rows:
            return pd.DataFrame()
        records = [
            {"display_name": row.display_name, "trade_date": row.trade_date, "close": float(row.close)}
            for row in rows
        ]
        frame = pd.DataFrame(records)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        pivot = frame.pivot_table(index="trade_date", columns="display_name", values="close", aggfunc="last").sort_index().ffill()
        return pivot

    def load_daily_graph_summary_frame(self, *, target: str) -> pd.DataFrame:
        rows = self.session.execute(
            select(
                DailyGraphSummaryModel.prediction_date,
                DailyGraphSummaryModel.feature_name,
                DailyGraphSummaryModel.feature_value,
            )
            .where(DailyGraphSummaryModel.target == target)
            .order_by(DailyGraphSummaryModel.prediction_date.asc())
        ).all()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [
                {
                    "prediction_date": row.prediction_date,
                    "feature_name": row.feature_name,
                    "feature_value": float(row.feature_value),
                }
                for row in rows
            ]
        )
        frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
        return frame.pivot_table(
            index="prediction_date",
            columns="feature_name",
            values="feature_value",
            aggfunc="last",
        ).sort_index()

    def load_latest_daily_graph_summary(self, *, target: str, before_prediction_date: str) -> dict | None:
        latest_date = self.session.execute(
            select(func.max(DailyGraphSummaryModel.prediction_date)).where(
                DailyGraphSummaryModel.target == target,
                DailyGraphSummaryModel.prediction_date < before_prediction_date,
            )
        ).scalar_one_or_none()
        if latest_date is None:
            return None
        rows = self.session.execute(
            select(DailyGraphSummaryModel).where(
                DailyGraphSummaryModel.target == target,
                DailyGraphSummaryModel.prediction_date == latest_date,
            )
        ).scalars().all()
        if not rows:
            return None
        return {
            "prediction_date": latest_date,
            "target": target,
            "schema_version": rows[0].schema_version,
            "generation_run_id": rows[0].generation_run_id,
            "features": {row.feature_name: float(row.feature_value) for row in rows},
            "feature_groups": {row.feature_name: row.feature_group for row in rows},
        }

    def load_daily_graph_delta_frame(self, *, target: str) -> pd.DataFrame:
        rows = self.session.execute(
            select(
                DailyGraphDeltaModel.prediction_date,
                DailyGraphDeltaModel.feature_name,
                DailyGraphDeltaModel.feature_value,
            )
            .where(DailyGraphDeltaModel.target == target)
            .order_by(DailyGraphDeltaModel.prediction_date.asc())
        ).all()
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [
                {
                    "prediction_date": row.prediction_date,
                    "feature_name": row.feature_name,
                    "feature_value": float(row.feature_value),
                }
                for row in rows
            ]
        )
        frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
        return frame.pivot_table(
            index="prediction_date",
            columns="feature_name",
            values="feature_value",
            aggfunc="last",
        ).sort_index()

    def upsert_prediction_run(self, payload: dict) -> None:
        payload = payload.copy()
        for key in ("pipeline_stage_status", "stage_diagnostics"):
            value = payload.get(key)
            if value is not None and not isinstance(value, str):
                payload[key] = json.dumps(value, sort_keys=True)
        statement = insert(DailyPredictionRunModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[DailyPredictionRunModel.run_id],
                set_=payload,
            )
        )
        self.session.flush()

    def upsert_projected_paths(self, rows: list[dict]) -> None:
        for row in rows:
            statement = insert(ProjectedPathModel).values(**row)
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[ProjectedPathModel.run_id, ProjectedPathModel.forecast_date, ProjectedPathModel.scenario_type],
                    set_=row,
                )
            )
        self.session.flush()

    def upsert_sector_outlooks(self, rows: list[dict]) -> None:
        for row in rows:
            statement = insert(SectorOutlookModel).values(**row)
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[SectorOutlookModel.run_id, SectorOutlookModel.sector_symbol],
                    set_=row,
                )
            )
        self.session.flush()

    def create_evaluation_run(self, record: EvaluationRunRecord) -> None:
        payload = {
            "evaluation_run_id": record.evaluation_run_id,
            "evaluation_date": record.evaluation_date,
            "horizon_days": record.horizon_days,
            "eligible_run_count": record.eligible_run_count,
            "completed_run_count": record.completed_run_count,
            "status": record.status,
            "notes": record.notes,
            "created_at": record.created_at,
            "completed_at": record.completed_at,
        }
        statement = insert(EvaluationRunModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[EvaluationRunModel.evaluation_run_id],
                set_=payload,
            )
        )
        self.session.flush()

    def complete_evaluation_run(
        self,
        evaluation_run_id: str,
        *,
        status: str,
        eligible_run_count: int,
        completed_run_count: int,
        completed_at: str,
        notes: str | None = None,
    ) -> None:
        row = self.session.get(EvaluationRunModel, evaluation_run_id)
        if row is None:
            raise KeyError(f"Unknown evaluation_run_id: {evaluation_run_id}")
        row.status = status
        row.eligible_run_count = eligible_run_count
        row.completed_run_count = completed_run_count
        row.completed_at = completed_at
        row.notes = notes
        self.session.flush()

    def list_mature_prediction_runs(self, *, as_of_date: str, horizon_days: int = 30) -> list[dict]:
        rows = self.session.execute(
            select(
                DailyPredictionRunModel.run_id,
                DailyPredictionRunModel.run_date,
                DailyPredictionRunModel.target,
                DailyPredictionRunModel.model_version,
                DailyPredictionRunModel.run_health,
                ProjectedPathModel.forecast_date,
            )
            .join(
                ProjectedPathModel,
                (ProjectedPathModel.run_id == DailyPredictionRunModel.run_id)
                & (ProjectedPathModel.scenario_type == "base")
                & (ProjectedPathModel.horizon_day == horizon_days),
            )
            .where(ProjectedPathModel.forecast_date <= as_of_date)
            .order_by(DailyPredictionRunModel.run_date.asc())
        ).all()
        return [
            {
                "run_id": row.run_id,
                "run_date": row.run_date,
                "target": row.target,
                "model_version": row.model_version,
                "run_health": row.run_health,
                "forecast_target_date": row.forecast_date,
            }
            for row in rows
        ]

    def load_projected_path_rows(self, run_id: str, *, scenario_type: str | None = None) -> list[dict]:
        query = select(ProjectedPathModel).where(ProjectedPathModel.run_id == run_id)
        if scenario_type is not None:
            query = query.where(ProjectedPathModel.scenario_type == scenario_type)
        rows = self.session.execute(query.order_by(ProjectedPathModel.horizon_day.asc())).scalars().all()
        return [
            {
                "run_id": row.run_id,
                "forecast_date": row.forecast_date,
                "horizon_day": row.horizon_day,
                "target_symbol": row.target_symbol,
                "predicted_price": row.predicted_price,
                "predicted_return": row.predicted_return,
                "lower_band": row.lower_band,
                "upper_band": row.upper_band,
                "scenario_type": row.scenario_type,
            }
            for row in rows
        ]

    def load_sector_outlook_rows(self, run_id: str) -> list[dict]:
        rows = self.session.execute(
            select(SectorOutlookModel).where(SectorOutlookModel.run_id == run_id).order_by(SectorOutlookModel.ranking_score.desc())
        ).scalars().all()
        return [
            {
                "run_id": row.run_id,
                "sector_symbol": row.sector_symbol,
                "sector_name": row.sector_name,
                "ranking_score": row.ranking_score,
                "expected_return_30d": row.expected_return_30d,
                "expected_volatility_30d": row.expected_volatility_30d,
                "confidence": row.confidence,
                "recommendation_label": row.recommendation_label,
                "rationale": row.rationale,
            }
            for row in rows
        ]

    def upsert_forecast_outcomes(self, rows: list[ForecastOutcomeRecord]) -> None:
        for row in rows:
            payload = {
                "run_id": row.run_id,
                "model_version": row.model_version,
                "evaluation_run_id": row.evaluation_run_id,
                "target": row.target,
                "forecast_start_date": row.forecast_start_date,
                "forecast_target_date": row.forecast_target_date,
                "horizon_days": row.horizon_days,
                "latest_price": row.latest_price,
                "predicted_price": row.predicted_price,
                "actual_price": row.actual_price,
                "predicted_return": row.predicted_return,
                "actual_return": row.actual_return,
                "predicted_direction_label": row.predicted_direction_label,
                "actual_direction_label": row.actual_direction_label,
                "prediction_error": row.prediction_error,
                "absolute_error": row.absolute_error,
                "band_hit_flag": row.band_hit_flag,
                "run_health": row.run_health,
                "outcome_status": row.outcome_status,
                "evaluated_at": row.evaluated_at,
            }
            statement = insert(ForecastOutcomeModel).values(**payload)
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[ForecastOutcomeModel.run_id, ForecastOutcomeModel.horizon_days],
                    set_=payload,
                )
            )
        self.session.flush()

    def upsert_sector_outcome_metrics(self, rows: list[SectorOutcomeMetricRecord]) -> None:
        for row in rows:
            payload = {
                "run_id": row.run_id,
                "model_version": row.model_version,
                "evaluation_run_id": row.evaluation_run_id,
                "sector_symbol": row.sector_symbol,
                "forecast_start_date": row.forecast_start_date,
                "forecast_target_date": row.forecast_target_date,
                "horizon_days": row.horizon_days,
                "predicted_rank_score": row.predicted_rank_score,
                "predicted_return_30d": row.predicted_return_30d,
                "actual_return_30d": row.actual_return_30d,
                "actual_rank": row.actual_rank,
                "top_bucket_hit_flag": row.top_bucket_hit_flag,
                "favor_hit_flag": row.favor_hit_flag,
                "rank_error": row.rank_error,
                "outcome_status": row.outcome_status,
                "evaluated_at": row.evaluated_at,
            }
            statement = insert(SectorOutcomeMetricModel).values(**payload)
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[SectorOutcomeMetricModel.run_id, SectorOutcomeMetricModel.sector_symbol],
                    set_=payload,
                )
            )
        self.session.flush()

    def upsert_model_metrics(self, rows: list[ModelMetricRecord]) -> None:
        for row in rows:
            payload = {
                "model_name": row.model_name,
                "model_version": row.model_version,
                "evaluation_date": row.evaluation_date,
                "horizon": row.horizon,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "training_window_start": row.training_window_start,
                "training_window_end": row.training_window_end,
            }
            statement = insert(ModelMetricModel).values(**payload)
            self.session.execute(
                statement.on_conflict_do_update(
                    index_elements=[
                        ModelMetricModel.model_name,
                        ModelMetricModel.model_version,
                        ModelMetricModel.evaluation_date,
                        ModelMetricModel.horizon,
                        ModelMetricModel.metric_name,
                    ],
                    set_=payload,
                )
            )
        self.session.flush()

    def load_forecast_outcomes(self, *, horizon_days: int = 30, outcome_status: str = "complete") -> list[dict]:
        rows = self.session.execute(
            select(ForecastOutcomeModel)
            .where(ForecastOutcomeModel.horizon_days == horizon_days, ForecastOutcomeModel.outcome_status == outcome_status)
            .order_by(ForecastOutcomeModel.forecast_start_date.asc())
        ).scalars().all()
        return [
            {
                "run_id": row.run_id,
                "model_version": row.model_version,
                "forecast_start_date": row.forecast_start_date,
                "forecast_target_date": row.forecast_target_date,
                "predicted_return": row.predicted_return,
                "actual_return": row.actual_return,
                "prediction_error": row.prediction_error,
                "absolute_error": row.absolute_error,
                "band_hit_flag": row.band_hit_flag,
                "predicted_direction_label": row.predicted_direction_label,
                "actual_direction_label": row.actual_direction_label,
                "run_health": row.run_health,
            }
            for row in rows
        ]

    def load_sector_outcome_metrics(self, *, horizon_days: int = 30, outcome_status: str = "complete") -> list[dict]:
        rows = self.session.execute(
            select(SectorOutcomeMetricModel)
            .where(SectorOutcomeMetricModel.horizon_days == horizon_days, SectorOutcomeMetricModel.outcome_status == outcome_status)
            .order_by(SectorOutcomeMetricModel.forecast_start_date.asc())
        ).scalars().all()
        return [
            {
                "run_id": row.run_id,
                "model_version": row.model_version,
                "sector_symbol": row.sector_symbol,
                "forecast_start_date": row.forecast_start_date,
                "forecast_target_date": row.forecast_target_date,
                "predicted_rank_score": row.predicted_rank_score,
                "predicted_return_30d": row.predicted_return_30d,
                "actual_return_30d": row.actual_return_30d,
                "actual_rank": row.actual_rank,
                "top_bucket_hit_flag": row.top_bucket_hit_flag,
                "favor_hit_flag": row.favor_hit_flag,
                "rank_error": row.rank_error,
            }
            for row in rows
        ]

    def get_scheduler_run_by_key(self, idempotency_key: str) -> dict | None:
        row = self.session.execute(
            select(SchedulerRunModel).where(SchedulerRunModel.idempotency_key == idempotency_key)
        ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "scheduler_run_id": row.scheduler_run_id,
            "job_name": row.job_name,
            "scheduled_for": row.scheduled_for,
            "started_at": row.started_at,
            "completed_at": row.completed_at,
            "status": row.status,
            "target": row.target,
            "prediction_date": row.prediction_date,
            "run_id": row.run_id,
            "error_message": row.error_message,
            "attempt_count": row.attempt_count,
            "idempotency_key": row.idempotency_key,
        }

    def upsert_graph_project(self, record: GraphProjectRecord) -> None:
        payload = {
            "project_id": record.project_id,
            "run_id": record.run_id,
            "status": record.status,
            "graph_backend": record.graph_backend,
            "backend_graph_ref": record.backend_graph_ref,
            "source_artifact_path": record.source_artifact_path,
            "ontology_json": record.ontology_json,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        statement = insert(GraphProjectModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[GraphProjectModel.project_id],
                set_=payload,
            )
        )
        self.session.flush()

    def upsert_graph_build_task(self, record: GraphBuildTaskRecord) -> None:
        payload = {
            "task_id": record.task_id,
            "project_id": record.project_id,
            "status": record.status,
            "progress_stage": record.progress_stage,
            "progress_detail": record.progress_detail,
            "error_message": record.error_message,
            "started_at": record.started_at,
            "stage_started_at": record.stage_started_at,
            "last_progress_at": record.last_progress_at,
            "telemetry_json": record.telemetry_json,
            "completed_at": record.completed_at,
        }
        statement = insert(GraphBuildTaskModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[GraphBuildTaskModel.task_id],
                set_=payload,
            )
        )
        self.session.flush()

    def upsert_graph_snapshot(self, record: GraphSnapshotRecord) -> None:
        payload = {
            "snapshot_id": record.snapshot_id,
            "project_id": record.project_id,
            "run_id": record.run_id,
            "status": record.status,
            "node_count": record.node_count,
            "edge_count": record.edge_count,
            "graph_json": record.graph_json,
            "highlights_json": record.highlights_json,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }
        statement = insert(GraphSnapshotModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[GraphSnapshotModel.snapshot_id],
                set_=payload,
            )
        )
        self.session.flush()

    def upsert_analysis_report(self, record: AnalysisReportRecord) -> None:
        payload = {
            "report_id": record.report_id,
            "subject_type": record.subject_type,
            "subject_id": record.subject_id,
            "status": record.status,
            "source_artifact_path": record.source_artifact_path,
            "graph_project_id": record.graph_project_id,
            "artifact_dir": record.artifact_dir,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
            "completed_at": record.completed_at,
            "error_message": record.error_message,
        }
        statement = insert(AnalysisReportModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[AnalysisReportModel.report_id],
                set_=payload,
            )
        )
        self.session.flush()

    def upsert_analysis_report_task(self, record: AnalysisReportTaskRecord) -> None:
        payload = {
            "task_id": record.task_id,
            "report_id": record.report_id,
            "status": record.status,
            "progress_stage": record.progress_stage,
            "error_message": record.error_message,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
        }
        statement = insert(AnalysisReportTaskModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[AnalysisReportTaskModel.task_id],
                set_=payload,
            )
        )
        self.session.flush()

    def get_graph_project(self, *, run_id: str | None = None, project_id: str | None = None) -> dict | None:
        query = select(GraphProjectModel)
        if run_id is not None:
            query = query.where(GraphProjectModel.run_id == run_id)
        if project_id is not None:
            query = query.where(GraphProjectModel.project_id == project_id)
        row = self.session.execute(query).scalar_one_or_none()
        if row is None:
            return None
        return {
            "project_id": row.project_id,
            "run_id": row.run_id,
            "status": row.status,
            "graph_backend": row.graph_backend,
            "backend_graph_ref": row.backend_graph_ref,
            "source_artifact_path": row.source_artifact_path,
            "ontology_json": row.ontology_json,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def get_analysis_report(self, report_id: str) -> dict | None:
        row = self.session.get(AnalysisReportModel, report_id)
        if row is None:
            return None
        return {
            "report_id": row.report_id,
            "subject_type": row.subject_type,
            "subject_id": row.subject_id,
            "status": row.status,
            "source_artifact_path": row.source_artifact_path,
            "graph_project_id": row.graph_project_id,
            "artifact_dir": row.artifact_dir,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "completed_at": row.completed_at,
            "error_message": row.error_message,
        }

    def get_latest_analysis_report_for_subject(self, *, subject_type: str, subject_id: str) -> dict | None:
        row = self.session.execute(
            select(AnalysisReportModel)
            .where(
                AnalysisReportModel.subject_type == subject_type,
                AnalysisReportModel.subject_id == subject_id,
            )
            .order_by(AnalysisReportModel.updated_at.desc())
        ).scalars().first()
        if row is None:
            return None
        return self.get_analysis_report(row.report_id)

    def get_analysis_report_task(self, task_id: str) -> dict | None:
        row = self.session.get(AnalysisReportTaskModel, task_id)
        if row is None:
            return None
        return {
            "task_id": row.task_id,
            "report_id": row.report_id,
            "status": row.status,
            "progress_stage": row.progress_stage,
            "error_message": row.error_message,
            "started_at": row.started_at,
            "completed_at": row.completed_at,
        }

    def get_latest_analysis_report_task(self, report_id: str) -> dict | None:
        row = self.session.execute(
            select(AnalysisReportTaskModel)
            .where(AnalysisReportTaskModel.report_id == report_id)
            .order_by(AnalysisReportTaskModel.started_at.desc())
        ).scalars().first()
        if row is None:
            return None
        return self.get_analysis_report_task(row.task_id)

    def get_graph_task(self, task_id: str) -> dict | None:
        row = self.session.get(GraphBuildTaskModel, task_id)
        if row is None:
            return None
        return {
            "task_id": row.task_id,
            "project_id": row.project_id,
            "status": row.status,
            "progress_stage": row.progress_stage,
            "progress_detail": row.progress_detail,
            "error_message": row.error_message,
            "started_at": row.started_at,
            "stage_started_at": row.stage_started_at,
            "last_progress_at": row.last_progress_at,
            "telemetry": json.loads(row.telemetry_json) if row.telemetry_json else None,
            "completed_at": row.completed_at,
        }

    def get_graph_snapshot(self, *, run_id: str | None = None, project_id: str | None = None) -> dict | None:
        query = select(GraphSnapshotModel)
        if run_id is not None:
            query = query.where(GraphSnapshotModel.run_id == run_id)
        if project_id is not None:
            query = query.where(GraphSnapshotModel.project_id == project_id)
        query = query.order_by(GraphSnapshotModel.updated_at.desc())
        row = self.session.execute(query).scalars().first()
        if row is None:
            return None
        return {
            "snapshot_id": row.snapshot_id,
            "project_id": row.project_id,
            "run_id": row.run_id,
            "status": row.status,
            "node_count": row.node_count,
            "edge_count": row.edge_count,
            "graph_json": row.graph_json,
            "highlights_json": row.highlights_json,
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }

    def list_graph_projects(self) -> list[dict]:
        rows = self.session.execute(select(GraphProjectModel).order_by(GraphProjectModel.updated_at.desc())).scalars().all()
        return [
            {
                "project_id": row.project_id,
                "run_id": row.run_id,
                "status": row.status,
                "graph_backend": row.graph_backend,
                "backend_graph_ref": row.backend_graph_ref,
                "source_artifact_path": row.source_artifact_path,
                "ontology_json": row.ontology_json,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]

    def get_latest_graph_task(self, project_id: str) -> dict | None:
        row = self.session.execute(
            select(GraphBuildTaskModel)
            .where(GraphBuildTaskModel.project_id == project_id)
            .order_by(GraphBuildTaskModel.started_at.desc())
        ).scalars().first()
        if row is None:
            return None
        return {
            "task_id": row.task_id,
            "project_id": row.project_id,
            "status": row.status,
            "progress_stage": row.progress_stage,
            "progress_detail": row.progress_detail,
            "error_message": row.error_message,
            "started_at": row.started_at,
            "stage_started_at": row.stage_started_at,
            "last_progress_at": row.last_progress_at,
            "telemetry": json.loads(row.telemetry_json) if row.telemetry_json else None,
            "completed_at": row.completed_at,
        }

    def upsert_scheduler_run(self, record: SchedulerRunRecord) -> None:
        payload = {
            "scheduler_run_id": record.scheduler_run_id,
            "job_name": record.job_name,
            "scheduled_for": record.scheduled_for,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "status": record.status,
            "target": record.target,
            "prediction_date": record.prediction_date,
            "run_id": record.run_id,
            "error_message": record.error_message,
            "attempt_count": record.attempt_count,
            "idempotency_key": record.idempotency_key,
        }
        statement = insert(SchedulerRunModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[SchedulerRunModel.idempotency_key],
                set_=payload,
            )
        )
        self.session.flush()

    def get_retraining_event(self, *, model_name: str, scheduled_date: str) -> dict | None:
        row = self.session.execute(
            select(RetrainingEventModel).where(
                RetrainingEventModel.model_name == model_name,
                RetrainingEventModel.scheduled_date == scheduled_date,
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return {
            "retraining_event_id": row.retraining_event_id,
            "model_name": row.model_name,
            "model_version": row.model_version,
            "scheduled_date": row.scheduled_date,
            "started_at": row.started_at,
            "completed_at": row.completed_at,
            "status": row.status,
            "training_window_start": row.training_window_start,
            "training_window_end": row.training_window_end,
            "training_row_count": row.training_row_count,
            "notes": row.notes,
        }

    def upsert_retraining_event(self, record: RetrainingEventRecord) -> None:
        payload = {
            "retraining_event_id": record.retraining_event_id,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "scheduled_date": record.scheduled_date,
            "started_at": record.started_at,
            "completed_at": record.completed_at,
            "status": record.status,
            "training_window_start": record.training_window_start,
            "training_window_end": record.training_window_end,
            "training_row_count": record.training_row_count,
            "notes": record.notes,
        }
        statement = insert(RetrainingEventModel).values(**payload)
        self.session.execute(
            statement.on_conflict_do_update(
                index_elements=[RetrainingEventModel.model_name, RetrainingEventModel.scheduled_date],
                set_=payload,
            )
        )
        self.session.flush()

    def latest_successful_model_version(self, model_name: str) -> str | None:
        row = self.session.execute(
            select(RetrainingEventModel)
            .where(RetrainingEventModel.model_name == model_name, RetrainingEventModel.status == "complete")
            .order_by(RetrainingEventModel.scheduled_date.desc(), RetrainingEventModel.completed_at.desc())
        ).scalars().first()
        if row is None:
            return None
        return row.model_version

    def latest_model_metric(
        self,
        *,
        model_name: str,
        metric_name: str,
        horizon: int,
        exclude_model_version: str | None = None,
    ) -> dict | None:
        query = (
            select(ModelMetricModel)
            .where(
                ModelMetricModel.model_name == model_name,
                ModelMetricModel.metric_name == metric_name,
                ModelMetricModel.horizon == horizon,
            )
            .order_by(ModelMetricModel.evaluation_date.desc(), ModelMetricModel.id.desc())
        )
        rows = self.session.execute(query).scalars().all()
        for row in rows:
            if exclude_model_version and row.model_version == exclude_model_version:
                continue
            return {
                "model_name": row.model_name,
                "model_version": row.model_version,
                "metric_name": row.metric_name,
                "metric_value": row.metric_value,
                "evaluation_date": row.evaluation_date,
                "horizon": row.horizon,
            }
        return None
