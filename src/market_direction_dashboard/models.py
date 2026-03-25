from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FetchResult:
    fetch_group: str
    provider_name: str
    payload: dict[str, Any] | list[Any] | str | None
    fetch_timestamp: str
    latency_ms: int
    status: str
    freshness_seconds: int | None = None
    fallback_used: bool = False
    item_count: int = 0
    warning: str | None = None
    proxy_for: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceItem:
    title: str
    source: str
    category: str
    published_at: str
    url: str
    summary: str
    impact: str
    impact_score: float
    id: str = ""
    source_type: str = "news"
    fetched_at: str = ""
    raw_text: str = ""
    instrument: str = ""
    region: str = "US"
    direction: str = "neutral"
    confidence_hint: float = 0.5
    freshness_score: float = 0.5
    credibility_score: float = 0.5
    proxy_used: bool = False
    quality_score: float = 0.5
    base_quality_score: float = 0.5
    evidence_kind: str = "direct"
    duplicate_cluster: str = ""
    graph_quality_adjustment: float = 0.0
    graph_quality_reasons: list[str] = field(default_factory=list)
    data_quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SignalFeature:
    name: str
    direction: str
    strength: float
    supporting_evidence_ids: list[str] = field(default_factory=list)
    conflict_count: int = 0
    time_decay_weight: float = 1.0
    category: str = "market"
    summary: str = ""
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentReport:
    name: str
    category: str
    summary: str
    bullish_points: list[str] = field(default_factory=list)
    bearish_points: list[str] = field(default_factory=list)
    score: float = 0.0
    confidence: float = 0.0
    dominant_regime_label: str = "mixed"
    unresolved_conflicts: list[str] = field(default_factory=list)
    evidence_coverage_count: int = 0
    missing_data_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SourceAgentReport:
    source: str
    summary: str
    categories: list[str] = field(default_factory=list)
    bullish_points: list[str] = field(default_factory=list)
    bearish_points: list[str] = field(default_factory=list)
    neutral_points: list[str] = field(default_factory=list)
    score: float = 0.0
    source_reliability: float = 0.0
    freshness_assessment: str = "unknown"
    source_confidence: float = 0.0
    source_warnings: list[str] = field(default_factory=list)
    evidence_ids_used: list[str] = field(default_factory=list)
    source_regime_fit: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChallengeReport:
    summary: str
    overconfident_categories: list[str] = field(default_factory=list)
    duplicate_narratives: list[str] = field(default_factory=list)
    proxy_risks: list[str] = field(default_factory=list)
    weak_confirmation: list[str] = field(default_factory=list)
    graph_risks: list[str] = field(default_factory=list)
    recommended_label: str = "NEUTRAL"
    conviction_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfidenceBreakdown:
    signal_strength: float
    source_diversity: float
    freshness: float
    agreement: float
    market_data_completeness: float
    fallback_proxy_burden: float
    total_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualitySummary:
    valid_item_count: int
    rejected_item_count: int
    duplicate_item_count: int
    stale_item_count: int
    malformed_item_count: int
    proxy_item_count: int
    distinct_provider_count: int
    average_quality_score: float
    graph_adjusted_item_count: int = 0
    graph_quality_summary: dict[str, Any] = field(default_factory=dict)
    cluster_quality_summary: list[dict[str, Any]] = field(default_factory=list)
    gate_failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionTraceStep:
    stage: str
    summary: str
    references: list[str] = field(default_factory=list)
    value: float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PredictionArtifacts:
    prediction_date: str
    target: str
    prediction_label: str
    confidence: float
    summary: str
    bullish_factors: list[str]
    bearish_factors: list[str]
    economic_report: str
    political_report: str
    social_report: str
    market_context_report: str
    sources: list[dict]
    source_agent_reports: list[dict]
    market_snapshot: dict
    warnings: list[str]
    source_diagnostics: dict
    run_id: str = ""
    next_session_date: str = ""
    run_health: str = "DEGRADED"
    final_score: float = 0.0
    confidence_breakdown: dict[str, Any] = field(default_factory=dict)
    data_quality_summary: dict[str, Any] = field(default_factory=dict)
    category_weights: dict[str, float] = field(default_factory=dict)
    quality_penalties: dict[str, float] = field(default_factory=dict)
    used_proxies: list[str] = field(default_factory=list)
    unused_sources: list[str] = field(default_factory=list)
    challenge_agent_report: dict[str, Any] = field(default_factory=dict)
    decision_trace: list[dict[str, Any]] = field(default_factory=list)
    signal_features: list[dict[str, Any]] = field(default_factory=list)
    statistical_engine_status: str = "DEGRADED"
    graph_priors: dict[str, Any] = field(default_factory=dict)
    graph_feature_summary: dict[str, Any] = field(default_factory=dict)
    graph_evidence_adjustments: dict[str, Any] = field(default_factory=dict)
    graph_conflict_summary: dict[str, Any] = field(default_factory=dict)
    graph_quality_summary: dict[str, Any] = field(default_factory=dict)
    graph_delta_summary: dict[str, Any] = field(default_factory=dict)
    history_coverage: dict[str, Any] = field(default_factory=dict)
    regime_probabilities: dict[str, float] = field(default_factory=dict)
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    posterior_probabilities: dict[str, float] = field(default_factory=dict)
    neutral_band: dict[str, Any] = field(default_factory=dict)
    statistical_failures: list[str] = field(default_factory=list)
    forecast_horizon_days: int = 30
    market_projection: dict[str, Any] = field(default_factory=dict)
    sector_outlook: list[dict[str, Any]] = field(default_factory=list)
    forecast_summary: dict[str, Any] = field(default_factory=dict)
    top_drivers: list[dict[str, Any]] = field(default_factory=list)
    confidence_notes: list[str] = field(default_factory=list)
    ensemble_diagnostics: dict[str, Any] = field(default_factory=dict)
    backend_diagnostics: dict[str, Any] = field(default_factory=dict)
    swarm_summary: dict[str, Any] = field(default_factory=dict)
    swarm_reporting: dict[str, Any] = field(default_factory=dict)
    swarm_diagnostics: dict[str, Any] = field(default_factory=dict)
    swarm_setup: dict[str, Any] = field(default_factory=dict)
    swarm_rounds: list[dict[str, Any]] = field(default_factory=list)
    swarm_agents: list[dict[str, Any]] = field(default_factory=list)
    swarm_priors: dict[str, float] = field(default_factory=dict)
    simulation_environment_summary: dict[str, Any] = field(default_factory=dict)
    simulation_environment_path: str = ""
    simulation_id: str = ""
    pipeline_stage_status: dict[str, Any] = field(default_factory=dict)
    stage_diagnostics: dict[str, Any] = field(default_factory=dict)
    feature_snapshot_version: str = ""
    model_stack_version: str = ""
    calibration_version: str = ""
    agreement_features: dict[str, float] = field(default_factory=dict)
    regime_slice: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
