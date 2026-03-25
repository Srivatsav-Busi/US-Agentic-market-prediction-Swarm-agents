from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class InstrumentModel(Base):
    __tablename__ = "instruments"

    instrument_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    asset_class: Mapped[str] = mapped_column(String(64), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    source_priority: Mapped[int] = mapped_column(Integer, nullable=False, default=100)
    proxy_symbol: Mapped[str | None] = mapped_column(String(64))
    active_flag: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class DailyPriceModel(Base):
    __tablename__ = "daily_prices"
    __table_args__ = (UniqueConstraint("trade_date", "instrument_id", name="uq_daily_prices_trade_date_instrument_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_date: Mapped[str] = mapped_column(String(32), nullable=False)
    instrument_id: Mapped[int] = mapped_column(ForeignKey("instruments.instrument_id"), nullable=False)
    open: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float, nullable=False)
    adjusted_close: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    proxy_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    ingestion_timestamp: Mapped[str] = mapped_column(String(64), nullable=False)


class MacroSeriesModel(Base):
    __tablename__ = "macro_series"
    __table_args__ = (UniqueConstraint("series_name", "observation_date", name="uq_macro_series_series_name_observation_date"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_name: Mapped[str] = mapped_column(String(64), nullable=False)
    observation_date: Mapped[str] = mapped_column(String(32), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)
    release_date: Mapped[str | None] = mapped_column(String(32))
    source: Mapped[str] = mapped_column(String(128), nullable=False)
    frequency: Mapped[str] = mapped_column(String(32), nullable=False)


class NewsEvidenceModel(Base):
    __tablename__ = "news_evidence"

    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_date: Mapped[str] = mapped_column(String(32), nullable=False)
    published_at: Mapped[str] = mapped_column(String(64), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False)
    category: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    url: Mapped[str | None] = mapped_column(Text)
    sentiment_score: Mapped[float | None] = mapped_column(Float)
    impact_score: Mapped[float | None] = mapped_column(Float)
    freshness_score: Mapped[float | None] = mapped_column(Float)
    credibility_score: Mapped[float | None] = mapped_column(Float)
    duplicate_cluster: Mapped[str | None] = mapped_column(String(128))


class FeatureSnapshotModel(Base):
    __tablename__ = "daily_feature_snapshot"
    __table_args__ = (UniqueConstraint("snapshot_date", "feature_name", "target_scope", name="uq_daily_feature_snapshot_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date: Mapped[str] = mapped_column(String(32), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_group: Mapped[str] = mapped_column(String(64), nullable=False)
    target_scope: Mapped[str] = mapped_column(String(128), nullable=False)
    generation_run_id: Mapped[str] = mapped_column(String(128), nullable=False)


class DailyPredictionRunModel(Base):
    __tablename__ = "daily_prediction_runs"

    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_date: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str] = mapped_column(String(128), nullable=False)
    prediction_label: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)
    run_health: Mapped[str | None] = mapped_column(String(32))
    expected_return: Mapped[float | None] = mapped_column(Float)
    expected_volatility: Mapped[float | None] = mapped_column(Float)
    posterior_up: Mapped[float | None] = mapped_column(Float)
    posterior_neutral: Mapped[float | None] = mapped_column(Float)
    posterior_down: Mapped[float | None] = mapped_column(Float)
    model_version: Mapped[str | None] = mapped_column(String(64))
    feature_snapshot_version: Mapped[str | None] = mapped_column(String(128))
    model_stack_version: Mapped[str | None] = mapped_column(String(128))
    calibration_version: Mapped[str | None] = mapped_column(String(128))
    regime_slice: Mapped[str | None] = mapped_column(String(64))
    agreement_score: Mapped[float | None] = mapped_column(Float)
    pipeline_stage_status: Mapped[str | None] = mapped_column(Text)
    stage_diagnostics: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)


class ProjectedPathModel(Base):
    __tablename__ = "projected_paths"
    __table_args__ = (UniqueConstraint("run_id", "forecast_date", "scenario_type", name="uq_projected_paths_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("daily_prediction_runs.run_id"), nullable=False)
    forecast_date: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_day: Mapped[int] = mapped_column(Integer, nullable=False)
    target_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    predicted_price: Mapped[float | None] = mapped_column(Float)
    predicted_return: Mapped[float | None] = mapped_column(Float)
    lower_band: Mapped[float | None] = mapped_column(Float)
    upper_band: Mapped[float | None] = mapped_column(Float)
    scenario_type: Mapped[str] = mapped_column(String(32), nullable=False)


class SectorOutlookModel(Base):
    __tablename__ = "sector_outlook"
    __table_args__ = (UniqueConstraint("run_id", "sector_symbol", name="uq_sector_outlook_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("daily_prediction_runs.run_id"), nullable=False)
    sector_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    sector_name: Mapped[str] = mapped_column(String(128), nullable=False)
    ranking_score: Mapped[float | None] = mapped_column(Float)
    expected_return_30d: Mapped[float | None] = mapped_column(Float)
    expected_volatility_30d: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)
    recommendation_label: Mapped[str | None] = mapped_column(String(32))
    rationale: Mapped[str | None] = mapped_column(Text)


class ModelMetricModel(Base):
    __tablename__ = "model_metrics"
    __table_args__ = (
        UniqueConstraint("model_name", "model_version", "evaluation_date", "horizon", "metric_name", name="uq_model_metrics_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluation_date: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon: Mapped[int] = mapped_column(Integer, nullable=False)
    metric_name: Mapped[str] = mapped_column(String(128), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    training_window_start: Mapped[str | None] = mapped_column(String(32))
    training_window_end: Mapped[str | None] = mapped_column(String(32))


class EvaluationRunModel(Base):
    __tablename__ = "evaluation_runs"

    evaluation_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    evaluation_date: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    eligible_run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_run_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    notes: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(64))


class ForecastOutcomeModel(Base):
    __tablename__ = "forecast_outcomes"
    __table_args__ = (UniqueConstraint("run_id", "horizon_days", name="uq_forecast_outcomes_run_horizon"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("daily_prediction_runs.run_id"), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    evaluation_run_id: Mapped[str | None] = mapped_column(ForeignKey("evaluation_runs.evaluation_run_id"))
    target: Mapped[str] = mapped_column(String(128), nullable=False)
    forecast_start_date: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_target_date: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    latest_price: Mapped[float | None] = mapped_column(Float)
    predicted_price: Mapped[float | None] = mapped_column(Float)
    actual_price: Mapped[float | None] = mapped_column(Float)
    predicted_return: Mapped[float | None] = mapped_column(Float)
    actual_return: Mapped[float | None] = mapped_column(Float)
    predicted_direction_label: Mapped[str | None] = mapped_column(String(32))
    actual_direction_label: Mapped[str | None] = mapped_column(String(32))
    prediction_error: Mapped[float | None] = mapped_column(Float)
    absolute_error: Mapped[float | None] = mapped_column(Float)
    band_hit_flag: Mapped[int | None] = mapped_column(Integer)
    run_health: Mapped[str | None] = mapped_column(String(32))
    outcome_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    evaluated_at: Mapped[str | None] = mapped_column(String(64))


class SectorOutcomeMetricModel(Base):
    __tablename__ = "sector_outcome_metrics"
    __table_args__ = (UniqueConstraint("run_id", "sector_symbol", name="uq_sector_outcome_metrics_run_sector"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("daily_prediction_runs.run_id"), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    evaluation_run_id: Mapped[str | None] = mapped_column(ForeignKey("evaluation_runs.evaluation_run_id"))
    sector_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    forecast_start_date: Mapped[str] = mapped_column(String(32), nullable=False)
    forecast_target_date: Mapped[str] = mapped_column(String(32), nullable=False)
    horizon_days: Mapped[int] = mapped_column(Integer, nullable=False)
    predicted_rank_score: Mapped[float | None] = mapped_column(Float)
    predicted_return_30d: Mapped[float | None] = mapped_column(Float)
    actual_return_30d: Mapped[float | None] = mapped_column(Float)
    actual_rank: Mapped[int | None] = mapped_column(Integer)
    top_bucket_hit_flag: Mapped[int | None] = mapped_column(Integer)
    favor_hit_flag: Mapped[int | None] = mapped_column(Integer)
    rank_error: Mapped[float | None] = mapped_column(Float)
    outcome_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    evaluated_at: Mapped[str | None] = mapped_column(String(64))


class SchedulerRunModel(Base):
    __tablename__ = "scheduler_runs"
    __table_args__ = (UniqueConstraint("idempotency_key", name="uq_scheduler_runs_idempotency_key"),)

    scheduler_run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    job_name: Mapped[str] = mapped_column(String(64), nullable=False)
    scheduled_for: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str | None] = mapped_column(String(128))
    prediction_date: Mapped[str | None] = mapped_column(String(32))
    run_id: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    idempotency_key: Mapped[str] = mapped_column(String(255), nullable=False)


class RetrainingEventModel(Base):
    __tablename__ = "retraining_events"
    __table_args__ = (UniqueConstraint("model_name", "scheduled_date", name="uq_retraining_events_model_schedule"),)

    retraining_event_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str | None] = mapped_column(String(64))
    scheduled_date: Mapped[str] = mapped_column(String(32), nullable=False)
    started_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    training_window_start: Mapped[str | None] = mapped_column(String(32))
    training_window_end: Mapped[str | None] = mapped_column(String(32))
    training_row_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[str | None] = mapped_column(Text)


class GraphProjectModel(Base):
    __tablename__ = "graph_projects"
    __table_args__ = (UniqueConstraint("run_id", name="uq_graph_projects_run_id"),)

    project_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    graph_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    backend_graph_ref: Mapped[str | None] = mapped_column(String(255))
    source_artifact_path: Mapped[str | None] = mapped_column(Text)
    ontology_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class GraphBuildTaskModel(Base):
    __tablename__ = "graph_build_tasks"

    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("graph_projects.project_id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    progress_stage: Mapped[str | None] = mapped_column(String(128))
    progress_detail: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[str] = mapped_column(String(64), nullable=False)
    stage_started_at: Mapped[str | None] = mapped_column(String(64))
    last_progress_at: Mapped[str | None] = mapped_column(String(64))
    telemetry_json: Mapped[str | None] = mapped_column(Text)
    completed_at: Mapped[str | None] = mapped_column(String(64))


class GraphSnapshotModel(Base):
    __tablename__ = "graph_snapshots"
    __table_args__ = (UniqueConstraint("project_id", name="uq_graph_snapshots_project_id"),)

    snapshot_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("graph_projects.project_id"), nullable=False)
    run_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    node_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    edge_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    graph_json: Mapped[str | None] = mapped_column(Text)
    highlights_json: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)


class DailyGraphSummaryModel(Base):
    __tablename__ = "daily_graph_summary"
    __table_args__ = (UniqueConstraint("prediction_date", "target", "feature_name", name="uq_daily_graph_summary_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_date: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_group: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generation_run_id: Mapped[str] = mapped_column(String(128), nullable=False)


class DailyGraphDeltaModel(Base):
    __tablename__ = "daily_graph_delta"
    __table_args__ = (UniqueConstraint("prediction_date", "target", "feature_name", name="uq_daily_graph_delta_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_date: Mapped[str] = mapped_column(String(32), nullable=False)
    target: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_name: Mapped[str] = mapped_column(String(255), nullable=False)
    feature_value: Mapped[float] = mapped_column(Float, nullable=False)
    feature_group: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    generation_run_id: Mapped[str] = mapped_column(String(128), nullable=False)


class AnalysisReportModel(Base):
    __tablename__ = "analysis_reports"

    report_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    subject_type: Mapped[str] = mapped_column(String(32), nullable=False)
    subject_id: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    source_artifact_path: Mapped[str | None] = mapped_column(Text)
    graph_project_id: Mapped[str | None] = mapped_column(String(128))
    artifact_dir: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(64), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)


class AnalysisReportTaskModel(Base):
    __tablename__ = "analysis_report_tasks"

    task_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    report_id: Mapped[str] = mapped_column(ForeignKey("analysis_reports.report_id"), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    progress_stage: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[str] = mapped_column(String(64), nullable=False)
    completed_at: Mapped[str | None] = mapped_column(String(64))


@dataclass(frozen=True)
class InstrumentRecord:
    symbol: str
    display_name: str
    asset_class: str
    category: str
    source_priority: int = 100
    proxy_symbol: str | None = None
    active_flag: int = 1


@dataclass(frozen=True)
class DailyPriceRecord:
    trade_date: str
    instrument_id: int
    close: float
    source: str
    ingestion_timestamp: str
    open: float | None = None
    high: float | None = None
    low: float | None = None
    adjusted_close: float | None = None
    volume: float | None = None
    proxy_used: int = 0


@dataclass(frozen=True)
class MacroObservationRecord:
    series_name: str
    observation_date: str
    value: float
    source: str
    frequency: str
    release_date: str | None = None


@dataclass(frozen=True)
class NewsEvidenceRecord:
    evidence_id: str
    run_date: str
    published_at: str
    source: str
    category: str
    title: str
    summary: str
    url: str
    sentiment_score: float
    impact_score: float
    freshness_score: float
    credibility_score: float
    duplicate_cluster: str


@dataclass(frozen=True)
class FeatureSnapshotRecord:
    snapshot_date: str
    feature_name: str
    feature_value: float
    feature_group: str
    target_scope: str
    generation_run_id: str


@dataclass(frozen=True)
class EvaluationRunRecord:
    evaluation_run_id: str
    evaluation_date: str
    horizon_days: int
    status: str
    created_at: str
    eligible_run_count: int = 0
    completed_run_count: int = 0
    notes: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class ForecastOutcomeRecord:
    run_id: str
    target: str
    forecast_start_date: str
    forecast_target_date: str
    horizon_days: int
    outcome_status: str
    model_version: str | None = None
    evaluation_run_id: str | None = None
    latest_price: float | None = None
    predicted_price: float | None = None
    actual_price: float | None = None
    predicted_return: float | None = None
    actual_return: float | None = None
    predicted_direction_label: str | None = None
    actual_direction_label: str | None = None
    prediction_error: float | None = None
    absolute_error: float | None = None
    band_hit_flag: int | None = None
    run_health: str | None = None
    evaluated_at: str | None = None


@dataclass(frozen=True)
class SectorOutcomeMetricRecord:
    run_id: str
    sector_symbol: str
    forecast_start_date: str
    forecast_target_date: str
    horizon_days: int
    outcome_status: str
    model_version: str | None = None
    evaluation_run_id: str | None = None
    predicted_rank_score: float | None = None
    predicted_return_30d: float | None = None
    actual_return_30d: float | None = None
    actual_rank: int | None = None
    top_bucket_hit_flag: int | None = None
    favor_hit_flag: int | None = None
    rank_error: float | None = None
    evaluated_at: str | None = None


@dataclass(frozen=True)
class ModelMetricRecord:
    model_name: str
    model_version: str
    evaluation_date: str
    horizon: int
    metric_name: str
    metric_value: float
    training_window_start: str | None = None
    training_window_end: str | None = None


@dataclass(frozen=True)
class SchedulerRunRecord:
    scheduler_run_id: str
    job_name: str
    started_at: str
    status: str
    idempotency_key: str
    scheduled_for: str | None = None
    completed_at: str | None = None
    target: str | None = None
    prediction_date: str | None = None
    run_id: str | None = None
    error_message: str | None = None
    attempt_count: int = 1


@dataclass(frozen=True)
class RetrainingEventRecord:
    retraining_event_id: str
    model_name: str
    scheduled_date: str
    started_at: str
    status: str
    model_version: str | None = None
    completed_at: str | None = None
    training_window_start: str | None = None
    training_window_end: str | None = None
    training_row_count: int = 0
    notes: str | None = None


@dataclass(frozen=True)
class GraphProjectRecord:
    project_id: str
    run_id: str
    status: str
    graph_backend: str
    created_at: str
    updated_at: str
    backend_graph_ref: str | None = None
    source_artifact_path: str | None = None
    ontology_json: str | None = None


@dataclass(frozen=True)
class GraphBuildTaskRecord:
    task_id: str
    project_id: str
    status: str
    started_at: str
    progress_stage: str | None = None
    progress_detail: str | None = None
    error_message: str | None = None
    stage_started_at: str | None = None
    last_progress_at: str | None = None
    telemetry_json: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class GraphSnapshotRecord:
    snapshot_id: str
    project_id: str
    run_id: str
    status: str
    node_count: int
    edge_count: int
    created_at: str
    updated_at: str
    graph_json: str | None = None
    highlights_json: str | None = None


@dataclass(frozen=True)
class DailyGraphSummaryRecord:
    prediction_date: str
    target: str
    feature_name: str
    feature_value: float
    feature_group: str
    schema_version: str
    generation_run_id: str


@dataclass(frozen=True)
class DailyGraphDeltaRecord:
    prediction_date: str
    target: str
    feature_name: str
    feature_value: float
    feature_group: str
    schema_version: str
    generation_run_id: str


@dataclass(frozen=True)
class AnalysisReportRecord:
    report_id: str
    subject_type: str
    subject_id: str
    status: str
    artifact_dir: str
    created_at: str
    updated_at: str
    source_artifact_path: str | None = None
    graph_project_id: str | None = None
    completed_at: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class AnalysisReportTaskRecord:
    task_id: str
    report_id: str
    status: str
    started_at: str
    progress_stage: str | None = None
    error_message: str | None = None
    completed_at: str | None = None
