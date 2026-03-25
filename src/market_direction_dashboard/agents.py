from __future__ import annotations

import uuid
from datetime import date, timedelta

from .llm_clients import BaseLLMClient
from .forecasting import apply_hybrid_overlay, build_baseline_forecast, build_hybrid_ml_overlay
from .graph_features import GraphFeatureVector, GraphPredictionContext
from .models import (
    AgentReport,
    ChallengeReport,
    ConfidenceBreakdown,
    DataQualitySummary,
    DecisionTraceStep,
    PredictionArtifacts,
    SignalFeature,
    SourceAgentReport,
    SourceItem,
)
from .simulation_reporting import build_swarm_reporting_payload
from .statistical_engine import build_statistical_decision


AGENT_SPECS = [
    ("economic_research_agent", "economic"),
    ("political_policy_agent", "political"),
    ("social_sentiment_agent", "social"),
    ("market_microstructure_agent", "market"),
]

WEIGHTS = {
    "economic": 0.35,
    "political": 0.20,
    "social": 0.10,
    "market": 0.35,
}


def run_source_agents(items: list[SourceItem], features: list[SignalFeature], llm_client: BaseLLMClient, target: str) -> list[SourceAgentReport]:
    grouped: dict[str, list[SourceItem]] = {}
    for item in items:
        grouped.setdefault(item.source, []).append(item)

    reports: list[SourceAgentReport] = []
    for source_name, source_items in grouped.items():
        source_features = [feature for feature in features if set(feature.supporting_evidence_ids) & {item.id for item in source_items}]
        fallback = _build_source_agent_fallback(source_name, source_items, source_features, target)
        summary = llm_client.summarize(
            f"You are a source-specific research subagent for {source_name}. Summarize how this data source affects the next-day outlook for {target}.",
            _format_items(source_items, source_features),
            fallback["summary"],
        )
        reports.append(
            SourceAgentReport(
                source=source_name,
                summary=summary,
                categories=fallback["categories"],
                bullish_points=fallback["bullish_points"],
                bearish_points=fallback["bearish_points"],
                neutral_points=fallback["neutral_points"],
                score=fallback["score"],
                source_reliability=fallback["source_reliability"],
                freshness_assessment=fallback["freshness_assessment"],
                source_confidence=fallback["source_confidence"],
                source_warnings=fallback["source_warnings"],
                evidence_ids_used=fallback["evidence_ids_used"],
                source_regime_fit=fallback["source_regime_fit"],
            )
        )
    reports.sort(key=lambda report: (abs(report.score), report.source), reverse=True)
    return reports


def run_research_agents(
    items: list[SourceItem],
    source_reports: list[SourceAgentReport],
    features: list[SignalFeature],
    llm_client: BaseLLMClient,
    target: str,
) -> dict[str, AgentReport]:
    reports: dict[str, AgentReport] = {}
    for agent_name, category in AGENT_SPECS:
        category_items = [item for item in items if item.category == category]
        category_reports = [report for report in source_reports if category in report.categories]
        category_features = [feature for feature in features if feature.category == category]
        fallback = _build_agent_fallback(agent_name, category, category_items, category_reports, category_features, target)
        summary = llm_client.summarize(
            f"You are {agent_name}. Produce a concise next-day market-impact summary for the {target}.",
            _format_items(category_items, category_features),
            fallback["summary"],
        )
        reports[category] = AgentReport(
            name=agent_name,
            category=category,
            summary=summary,
            bullish_points=fallback["bullish_points"],
            bearish_points=fallback["bearish_points"],
            score=fallback["score"],
            confidence=fallback["confidence"],
            dominant_regime_label=fallback["dominant_regime_label"],
            unresolved_conflicts=fallback["unresolved_conflicts"],
            evidence_coverage_count=fallback["evidence_coverage_count"],
            missing_data_penalty=fallback["missing_data_penalty"],
        )
    return reports


def synthesize_prediction(
    prediction_date: str,
    target: str,
    config: dict,
    reports: dict[str, AgentReport],
    items: list[SourceItem],
    snapshot: dict,
    warnings: list[str],
    source_diagnostics: dict,
    source_agent_reports: list[SourceAgentReport],
    llm_client: BaseLLMClient,
    features: list[SignalFeature],
    challenge_llm_client: BaseLLMClient | None = None,
    final_summary_llm_client: BaseLLMClient | None = None,
    backend_diagnostics: dict | None = None,
    swarm_priors: dict | None = None,
    swarm_diagnostics: dict | None = None,
    swarm_summary: dict | None = None,
    swarm_reporting: dict | None = None,
    swarm_setup: dict | None = None,
    swarm_rounds: list[dict] | None = None,
    swarm_agents: list[dict] | None = None,
    simulation_environment_summary: dict | None = None,
    simulation_environment_path: str = "",
    simulation_id: str = "",
    graph_prediction_context: GraphPredictionContext | None = None,
    graph_feature_vector: GraphFeatureVector | dict | None = None,
    graph_delta_summary: dict | None = None,
) -> PredictionArtifacts:
    challenge_llm_client = challenge_llm_client or llm_client
    final_summary_llm_client = final_summary_llm_client or llm_client
    quality_summary = DataQualitySummary(**source_diagnostics.get("data_quality_summary", {}))
    agreement_features = _compute_agreement_features(items, features, source_agent_reports)
    challenge_report = run_challenge_agent(reports, items, source_agent_reports, features, quality_summary, challenge_llm_client, target)
    statistical_decision = build_statistical_decision(
        target=target,
        snapshot=snapshot,
        items=items,
        features=features,
        quality_summary=quality_summary,
        config=config,
        graph_prediction_context=graph_prediction_context,
        graph_delta_summary=graph_delta_summary,
    )
    baseline_forecast = build_baseline_forecast(target=target, snapshot=snapshot, config=config)
    ml_overlay = build_hybrid_ml_overlay(
        target=target,
        snapshot=snapshot,
        config=config,
        graph_feature_vector=graph_feature_vector,
    )
    evidence_bias = float(
        sum(
            feature.strength if feature.direction in {"positive", "bullish"} else -feature.strength
            for feature in features[:8]
        )
        / max(len(features[:8]) or 1, 1)
    )
    hybrid_forecast = apply_hybrid_overlay(
        baseline=baseline_forecast,
        ml_result=ml_overlay.to_dict(),
        evidence_bias=evidence_bias,
        regime_probabilities=statistical_decision.regime_probabilities,
    )
    penalties = _compute_quality_penalties(items, quality_summary)
    posterior_probabilities = dict(statistical_decision.posterior_probabilities)
    swarm_priors = swarm_priors or {}
    if swarm_priors:
        posterior_probabilities["UP"] += float(swarm_priors.get("swarm_up_bias", 0.0))
        posterior_probabilities["DOWN"] += float(swarm_priors.get("swarm_down_bias", 0.0))
        posterior_probabilities["NEUTRAL"] = max(
            1e-8,
            posterior_probabilities["NEUTRAL"] - min(
                posterior_probabilities["NEUTRAL"] * 0.35,
                abs(float(swarm_priors.get("swarm_confidence_modifier", 0.0))),
            ),
        )
        posterior_probabilities = _normalize_probabilities(posterior_probabilities)
    if challenge_report.conviction_penalty:
        posterior_probabilities["NEUTRAL"] += challenge_report.conviction_penalty * 0.7
        posterior_probabilities["UP"] *= max(0.4, 1.0 - challenge_report.conviction_penalty)
        posterior_probabilities["DOWN"] *= max(0.4, 1.0 - challenge_report.conviction_penalty)
        posterior_probabilities = _normalize_probabilities(posterior_probabilities)

    adjusted_score = posterior_probabilities["UP"] - posterior_probabilities["DOWN"]
    gate_failures = quality_summary.gate_failures
    label = statistical_decision.label
    if label != "NEUTRAL" and challenge_report.conviction_penalty >= 0.12:
        label = "NEUTRAL"
    if gate_failures:
        label = "NEUTRAL"
        adjusted_score = 0.0

    confidence_breakdown = _build_confidence_breakdown(
        items=items,
        reports=reports,
        quality_summary=quality_summary,
        gate_failures=gate_failures,
        statistical_decision=statistical_decision,
    )
    confidence = min(confidence_breakdown.total_confidence, statistical_decision.confidence)
    if swarm_priors and not gate_failures:
        confidence = min(92.0, max(25.0, confidence + float(swarm_priors.get("swarm_confidence_modifier", 0.0)) * 100.0))
    if gate_failures:
        confidence = min(confidence, 58.0)
    run_health = _classify_run_health(quality_summary, source_diagnostics)
    stage_diagnostics = {
        "ingestion": {
            "status": "complete",
            "market_series": len(snapshot.get("series", {})),
            "history_series": len(snapshot.get("history", {})),
            "source_attempts": len(source_diagnostics.get("fetch_results", [])),
        },
        "normalization": {
            "status": "complete",
            "valid_items": quality_summary.valid_item_count,
            "gate_failures": quality_summary.gate_failures,
            "avg_quality_score": quality_summary.average_quality_score,
        },
        "feature_extraction": {
            "status": "complete",
            "feature_count": len(features),
            "agreement_score": round(agreement_features["agreement_score"], 4),
            "conflict_score": round(agreement_features["conflict_score"], 4),
        },
        "statistical_decision": {
            "status": "complete" if statistical_decision.engine_status != "DEGRADED" else "degraded",
            "engine_status": statistical_decision.engine_status,
            "failures": statistical_decision.failures,
        },
        "challenge": {
            "status": "complete",
            "conviction_penalty": challenge_report.conviction_penalty,
            "recommended_label": challenge_report.recommended_label,
        },
        "graph_quality": {
            "status": "complete" if quality_summary.graph_quality_summary else "skipped",
            "graph_adjusted_item_count": quality_summary.graph_adjusted_item_count,
            "graph_quality_score": (quality_summary.graph_quality_summary or {}).get("graph_quality_score"),
            "severe_graph_risk": (quality_summary.graph_quality_summary or {}).get("severe_graph_risk"),
        },
        "swarm": {
            "status": "complete" if swarm_diagnostics else "skipped",
            "agent_count": len(swarm_agents or []),
            "round_count": len(swarm_rounds or []),
            "dominant_stance": (swarm_summary or {}).get("dominant_stance"),
        },
        "publish": {
            "status": "complete",
            "prediction_label": label,
            "run_health": run_health,
        },
    }
    pipeline_stage_status = {stage: payload["status"] for stage, payload in stage_diagnostics.items()}

    bullish_factors = _top_factor_titles(items, features, positive=True)
    bearish_factors = _top_factor_titles(items, features, positive=False)

    fallback_summary = (
        f"The next-trading-day outlook for the {target} is {label}. "
        f"Final statistical score is {adjusted_score:+.2f} with expected return {statistical_decision.expected_return:+.3%} "
        f"and expected volatility {statistical_decision.expected_volatility:.3%}. "
        f"Run health is {run_health}."
    )
    summary = final_summary_llm_client.summarize(
        "You are the direction_judge_agent. Produce a final market prediction summary.",
        "\n\n".join(
            [report.summary for report in reports.values()]
            + [challenge_report.summary]
            + [feature.summary for feature in features[:8]]
            + [
                f"Statistical engine={statistical_decision.engine_status}",
                f"Posterior probabilities={posterior_probabilities}",
            ]
        ),
        fallback_summary,
    )

    decision_trace = [
        DecisionTraceStep(
            stage="validation",
            summary=(
                f"Accepted {quality_summary.valid_item_count} evidence items, rejected {quality_summary.rejected_item_count}, "
                f"duplicate removals {quality_summary.duplicate_item_count}, stale removals {quality_summary.stale_item_count}."
            ),
            value=quality_summary.average_quality_score,
        ),
        DecisionTraceStep(
            stage="feature_extraction",
            summary=f"Derived {len(features)} structured signal features from normalized evidence.",
            references=[feature.name for feature in features[:8]],
            value=len(features),
        ),
        *[DecisionTraceStep(**step) if isinstance(step, dict) else step for step in statistical_decision.trace_steps],
        DecisionTraceStep(
            stage="challenge",
            summary=challenge_report.summary,
            references=challenge_report.overconfident_categories + challenge_report.duplicate_narratives + challenge_report.graph_risks,
            value=round(challenge_report.conviction_penalty, 3),
        ),
        DecisionTraceStep(
            stage="swarm",
            summary=(
                f"Swarm consensus {(swarm_summary or {}).get('dominant_stance', 'not available')} "
                f"with average consensus {(swarm_summary or {}).get('average_consensus', 0.0):.2f} "
                f"and conflict {(swarm_summary or {}).get('average_conflict', 0.0):.2f}."
                if swarm_summary
                else "Swarm layer skipped."
            ),
            references=[feature.name for feature in features if feature.name.startswith("swarm_")][:5],
            value=round(float(swarm_priors.get("swarm_confidence_modifier", 0.0)), 3) if swarm_priors else 0.0,
        ),
        DecisionTraceStep(
            stage="final_decision",
            summary=f"Applied Bayesian evidence update, challenge penalties, and trust gates to produce {label} with run health {run_health}.",
            references=gate_failures + statistical_decision.failures,
            value=round(adjusted_score, 3),
        ),
    ]

    next_session_date = _next_session_date(prediction_date)
    swarm_reporting = swarm_reporting or build_swarm_reporting_payload(
        profiles=swarm_agents or [],
        rounds=swarm_rounds or [],
        setup=swarm_setup or {},
        summary=swarm_summary or {},
    )
    return PredictionArtifacts(
        prediction_date=prediction_date,
        target=target,
        prediction_label=label,
        confidence=round(confidence, 1),
        summary=summary,
        bullish_factors=bullish_factors,
        bearish_factors=bearish_factors,
        economic_report=reports["economic"].summary,
        political_report=reports["political"].summary,
        social_report=reports["social"].summary,
        market_context_report=reports["market"].summary,
        sources=[item.to_dict() for item in items],
        source_agent_reports=[report.to_dict() for report in source_agent_reports],
        market_snapshot=snapshot,
        warnings=warnings,
        source_diagnostics=source_diagnostics,
        run_id=str(uuid.uuid4()),
        next_session_date=next_session_date,
        run_health=run_health,
        final_score=round(adjusted_score, 3),
        confidence_breakdown=confidence_breakdown.to_dict(),
        data_quality_summary=quality_summary.to_dict(),
        category_weights=WEIGHTS.copy(),
        quality_penalties={key: round(value, 3) for key, value in penalties.items()},
        used_proxies=source_diagnostics.get("used_proxies", []),
        unused_sources=source_diagnostics.get("unused_sources", []),
        challenge_agent_report=challenge_report.to_dict(),
        decision_trace=[step.to_dict() for step in decision_trace],
        signal_features=[feature.to_dict() for feature in features],
        statistical_engine_status=statistical_decision.engine_status,
        graph_priors=statistical_decision.graph_priors,
        graph_feature_summary=statistical_decision.graph_feature_summary,
        graph_evidence_adjustments=statistical_decision.graph_evidence_adjustments,
        graph_conflict_summary=statistical_decision.graph_conflict_summary,
        graph_quality_summary=quality_summary.graph_quality_summary,
        graph_delta_summary=statistical_decision.graph_delta_summary,
        history_coverage=statistical_decision.history_coverage,
        regime_probabilities=statistical_decision.regime_probabilities,
        expected_return=statistical_decision.expected_return,
        expected_volatility=statistical_decision.expected_volatility,
        posterior_probabilities={key: round(value, 4) for key, value in posterior_probabilities.items()},
        neutral_band=statistical_decision.neutral_band,
        statistical_failures=statistical_decision.failures,
        forecast_horizon_days=hybrid_forecast.horizon_days,
        market_projection=hybrid_forecast.projection,
        sector_outlook=hybrid_forecast.sectors,
        forecast_summary={
            "horizon_days": hybrid_forecast.horizon_days,
            "regime_label": hybrid_forecast.regime_label,
            "expected_return_30d": round(hybrid_forecast.expected_return_30d, 6),
            "expected_volatility_30d": round(hybrid_forecast.expected_volatility_30d, 6),
        },
        top_drivers=hybrid_forecast.top_drivers + [
            {
                "name": "ML overlay",
                "direction": "positive" if ml_overlay.predicted_return_30d >= 0 else "negative",
                "value": round(ml_overlay.predicted_return_30d, 6),
                "summary": "Gradient-boosted 30-day return estimate blended into the baseline projection.",
            }
        ],
        confidence_notes=hybrid_forecast.confidence_notes,
        ensemble_diagnostics={
            "mode": "hybrid_v1",
            "weights": {
                "baseline": 0.45,
                "ml": 0.35,
                "regime": 0.10,
                "live_evidence": 0.10,
            },
            "ml_overlay": ml_overlay.to_dict(),
            "evidence_bias": round(evidence_bias, 6),
            "regime_probabilities": statistical_decision.regime_probabilities,
        },
        backend_diagnostics=backend_diagnostics or {},
        swarm_summary=swarm_summary or {},
        swarm_reporting=swarm_reporting,
        swarm_diagnostics=swarm_diagnostics or {},
        swarm_setup=swarm_setup or {},
        swarm_rounds=swarm_rounds or [],
        swarm_agents=swarm_agents or [],
        swarm_priors={key: round(float(value), 4) for key, value in (swarm_priors or {}).items()},
        simulation_environment_summary=simulation_environment_summary or {},
        simulation_environment_path=simulation_environment_path,
        simulation_id=simulation_id,
        pipeline_stage_status=pipeline_stage_status,
        stage_diagnostics=stage_diagnostics,
        feature_snapshot_version="daily_feature_snapshot:v2",
        model_stack_version="next_session_stack:v2",
        calibration_version="confidence_calibration:v2",
        agreement_features={key: round(value, 4) for key, value in agreement_features.items()},
        regime_slice=max(statistical_decision.regime_probabilities, key=statistical_decision.regime_probabilities.get, default="mixed"),
    )


def run_challenge_agent(
    reports: dict[str, AgentReport],
    items: list[SourceItem],
    source_reports: list[SourceAgentReport],
    features: list[SignalFeature],
    quality_summary: DataQualitySummary,
    llm_client: BaseLLMClient,
    target: str,
) -> ChallengeReport:
    overconfident = [category for category, report in reports.items() if abs(report.score) > 0.5 and report.confidence > 0.75]
    duplicate_clusters = {item.duplicate_cluster for item in items if item.duplicate_cluster}
    duplicate_narratives = [f"{len(items) - len(duplicate_clusters)} duplicate narratives compressed"] if len(items) > len(duplicate_clusters) else []
    proxy_risks = [report.source for report in source_reports if "proxy_used" in report.source_warnings]
    weak_confirmation = [feature.name for feature in features if feature.conflict_count > 0]
    graph_quality = quality_summary.graph_quality_summary or {}
    graph_risks = []
    if graph_quality.get("severe_graph_risk"):
        graph_risks.append("severe_graph_risk")
    if float(graph_quality.get("source_monoculture_penalty", 0.0)) >= 0.08:
        graph_risks.append("source_monoculture")
    if float(graph_quality.get("contradiction_penalty", 0.0)) >= 0.08:
        graph_risks.append("graph_contradiction")
    if float(graph_quality.get("stale_cluster_penalty", 0.0)) >= 0.08:
        graph_risks.append("stale_clusters")
    conviction_penalty = min(
        0.32,
        0.05 * len(overconfident)
        + 0.03 * len(proxy_risks)
        + 0.03 * len(weak_confirmation)
        + 0.04 * len(graph_risks)
        + float(graph_quality.get("cluster_redundancy_penalty", 0.0)) * 0.35
        + float(graph_quality.get("contradiction_penalty", 0.0)) * 0.45
        + float(graph_quality.get("source_monoculture_penalty", 0.0)) * 0.35,
    )
    recommended = "NEUTRAL" if conviction_penalty >= 0.12 else "UP" if sum(report.score for report in reports.values()) > 0 else "DOWN"
    fallback_summary = (
        f"Challenge review for {target}: "
        f"overconfident categories={', '.join(overconfident) or 'none'}, "
        f"proxy risks={', '.join(proxy_risks) or 'none'}, "
        f"mixed-signal features={', '.join(weak_confirmation[:4]) or 'none'}, "
        f"graph risks={', '.join(graph_risks) or 'none'}."
    )
    summary = llm_client.summarize(
        "You are the challenge_agent. Attack the current conclusion and identify overconfidence, duplicate narratives, and weak confirmation.",
        "\n".join(
            [f"{category}: score={report.score:+.2f}, confidence={report.confidence:.2f}" for category, report in reports.items()]
            + [feature.summary for feature in features[:8]]
        ),
        fallback_summary,
    )
    return ChallengeReport(
        summary=summary,
        overconfident_categories=overconfident,
        duplicate_narratives=duplicate_narratives,
        proxy_risks=proxy_risks,
        weak_confirmation=weak_confirmation[:5],
        graph_risks=graph_risks[:5],
        recommended_label=recommended,
        conviction_penalty=round(conviction_penalty, 3),
    )


def _compute_agreement_features(
    items: list[SourceItem],
    features: list[SignalFeature],
    source_reports: list[SourceAgentReport],
) -> dict[str, float]:
    bullish_items = sum(1 for item in items if item.direction == "bullish")
    bearish_items = sum(1 for item in items if item.direction == "bearish")
    directional_count = max(bullish_items + bearish_items, 1)
    evidence_alignment = abs(bullish_items - bearish_items) / directional_count
    feature_conflicts = sum(1 for feature in features if feature.conflict_count > 0)
    feature_conflict_ratio = feature_conflicts / max(len(features), 1)
    proxy_risk_ratio = sum(1 for report in source_reports if "proxy_used" in report.source_warnings) / max(len(source_reports), 1)
    agreement_score = max(0.0, min(1.0, evidence_alignment * 0.7 + (1.0 - feature_conflict_ratio) * 0.3))
    conflict_score = max(0.0, min(1.0, feature_conflict_ratio * 0.65 + proxy_risk_ratio * 0.35))
    return {
        "agreement_score": agreement_score,
        "conflict_score": conflict_score,
        "bullish_item_share": bullish_items / directional_count,
        "bearish_item_share": bearish_items / directional_count,
        "proxy_risk_ratio": proxy_risk_ratio,
    }


def _build_agent_fallback(
    agent_name: str,
    category: str,
    items: list[SourceItem],
    source_reports: list[SourceAgentReport],
    features: list[SignalFeature],
    target: str,
) -> dict:
    if not items:
        return {
            "summary": f"{agent_name} found no usable {category} inputs for the {target}; this category remains neutral.",
            "bullish_points": [],
            "bearish_points": [],
            "score": 0.0,
            "confidence": 0.15,
            "dominant_regime_label": "insufficient_data",
            "unresolved_conflicts": ["No evidence available."],
            "evidence_coverage_count": 0,
            "missing_data_penalty": 0.15,
        }

    bullish = [item.title for item in items if item.impact_score > 0.15][:3]
    bearish = [item.title for item in items if item.impact_score < -0.15][:3]
    weighted_quality = sum(item.impact_score * max(0.2, item.quality_score) for item in items)
    quality_total = sum(max(0.2, item.quality_score) for item in items) or 1.0
    score = weighted_quality / quality_total
    feature_bias = sum((_feature_sign(feature) * feature.strength * feature.time_decay_weight) for feature in features)
    score = max(-1.0, min(1.0, score * 0.7 + feature_bias * 0.3))
    missing_data_penalty = 0.0 if len(items) >= 2 else 0.12
    confidence = max(
        0.15,
        min(
            0.95,
            0.3
            + min(0.3, len(items) * 0.08)
            + min(0.2, len(source_reports) * 0.05)
            + abs(score) * 0.15
            - missing_data_penalty,
        ),
    )
    unresolved = [feature.name for feature in features if feature.conflict_count > 0]
    regime = _dominant_regime_label(category, score, features)
    summary = (
        f"{agent_name} reviewed {len(items)} {category} inputs for the {target}. "
        f"Calibrated score is {score:+.2f} with confidence {confidence:.2f}. "
        f"Bullish drivers: {', '.join(bullish) if bullish else 'none'}. "
        f"Bearish drivers: {', '.join(bearish) if bearish else 'none'}. "
        f"Dominant regime: {regime}."
    )
    return {
        "summary": summary,
        "bullish_points": bullish,
        "bearish_points": bearish,
        "score": round(score, 3),
        "confidence": round(confidence, 3),
        "dominant_regime_label": regime,
        "unresolved_conflicts": unresolved[:3],
        "evidence_coverage_count": len(items),
        "missing_data_penalty": round(missing_data_penalty, 3),
    }


def _build_source_agent_fallback(
    source_name: str,
    items: list[SourceItem],
    features: list[SignalFeature],
    target: str,
) -> dict:
    categories = sorted({item.category for item in items})
    bullish = [item.title for item in items if item.impact_score > 0.15][:3]
    bearish = [item.title for item in items if item.impact_score < -0.15][:3]
    neutral = [item.title for item in items if -0.15 <= item.impact_score <= 0.15][:2]
    quality_weighted = sum(item.impact_score * max(0.2, item.quality_score) for item in items)
    quality_total = sum(max(0.2, item.quality_score) for item in items) or 1.0
    average_score = quality_weighted / quality_total
    reliability = sum(item.credibility_score for item in items) / len(items) if items else 0.0
    freshness_score = sum(item.freshness_score for item in items) / len(items) if items else 0.0
    source_confidence = max(0.1, min(0.95, 0.4 * reliability + 0.4 * freshness_score + 0.2 * abs(average_score)))
    warnings = []
    if any(item.proxy_used for item in items):
        warnings.append("proxy_used")
    if freshness_score < 0.4:
        warnings.append("aging_evidence")
    if reliability < 0.5:
        warnings.append("low_reliability")
    regime_fit = _source_regime_fit(categories, features)
    summary = (
        f"The {source_name} subagent reviewed {len(items)} items for the {target}. "
        f"Covered categories: {', '.join(categories) if categories else 'none'}. "
        f"Calibrated score is {average_score:+.2f} with confidence {source_confidence:.2f}. "
        f"Freshness is {'fresh' if freshness_score >= 0.55 else 'mixed' if freshness_score >= 0.35 else 'stale-leaning'}. "
        f"Regime fit: {regime_fit}."
    )
    return {
        "summary": summary,
        "categories": categories,
        "bullish_points": bullish,
        "bearish_points": bearish,
        "neutral_points": neutral,
        "score": round(average_score, 3),
        "source_reliability": round(reliability, 3),
        "freshness_assessment": "fresh" if freshness_score >= 0.55 else "mixed" if freshness_score >= 0.35 else "stale",
        "source_confidence": round(source_confidence, 3),
        "source_warnings": warnings,
        "evidence_ids_used": [item.id for item in items],
        "source_regime_fit": regime_fit,
    }


def _build_confidence_breakdown(
    items: list[SourceItem],
    reports: dict[str, AgentReport],
    quality_summary: DataQualitySummary,
    gate_failures: list[str],
    statistical_decision,
) -> ConfidenceBreakdown:
    signal_strength = min(1.0, abs(statistical_decision.final_score) + statistical_decision.posterior_probabilities.get("NEUTRAL", 0.0) * 0.2)
    source_diversity = min(1.0, quality_summary.distinct_provider_count / 5.0)
    freshness = sum(item.freshness_score for item in items) / len(items) if items else 0.0
    agreement = min(1.0, max(0.0, 1.0 - (1.0 - max(statistical_decision.regime_probabilities.values(), default=0.33))))
    market_data_completeness = min(1.0, len([item for item in items if item.category == "market"]) / 6.0)
    fallback_proxy_burden = max(0.0, 1.0 - (quality_summary.proxy_item_count / max(1, quality_summary.valid_item_count)))
    total = (
        signal_strength * 0.26
        + source_diversity * 0.14
        + freshness * 0.18
        + agreement * 0.16
        + market_data_completeness * 0.16
        + fallback_proxy_burden * 0.10
    ) * 100.0
    if gate_failures:
        total *= 0.78
    return ConfidenceBreakdown(
        signal_strength=round(signal_strength * 100.0, 1),
        source_diversity=round(source_diversity * 100.0, 1),
        freshness=round(freshness * 100.0, 1),
        agreement=round(agreement * 100.0, 1),
        market_data_completeness=round(market_data_completeness * 100.0, 1),
        fallback_proxy_burden=round(fallback_proxy_burden * 100.0, 1),
        total_confidence=round(max(25.0, min(92.0, total)), 1),
    )


def _compute_quality_penalties(items: list[SourceItem], quality_summary: DataQualitySummary) -> dict[str, float]:
    total_items = max(1, quality_summary.valid_item_count)
    duplicate_penalty = min(0.12, quality_summary.duplicate_item_count / max(1, total_items) * 0.08)
    stale_penalty = min(0.14, quality_summary.stale_item_count / max(1, total_items) * 0.14)
    proxy_penalty = min(0.16, quality_summary.proxy_item_count / max(1, total_items) * 0.16)
    missing_penalty = min(0.16, len(quality_summary.gate_failures) * 0.04)
    has_bullish = any(item.impact == "bullish" for item in items)
    has_bearish = any(item.impact == "bearish" for item in items)
    conflict_penalty = 0.08 if has_bullish and has_bearish else 0.0
    graph_quality = quality_summary.graph_quality_summary or {}
    return {
        "missing_data_penalty": float(missing_penalty),
        "stale_data_penalty": float(stale_penalty),
        "proxy_overuse_penalty": float(proxy_penalty),
        "duplicate_headline_penalty": float(duplicate_penalty),
        "conflict_penalty": float(conflict_penalty),
        "graph_cluster_redundancy_penalty": float(graph_quality.get("cluster_redundancy_penalty", 0.0)),
        "graph_stale_cluster_penalty": float(graph_quality.get("stale_cluster_penalty", 0.0)),
        "graph_source_monoculture_penalty": float(graph_quality.get("source_monoculture_penalty", 0.0)),
        "graph_independent_corroboration_boost": float(graph_quality.get("independent_corroboration_boost", 0.0)),
    }


def _classify_run_health(quality_summary: DataQualitySummary, source_diagnostics: dict) -> str:
    graph_quality = quality_summary.graph_quality_summary or {}
    if graph_quality.get("severe_graph_risk"):
        return "LOW_TRUST"
    if len(quality_summary.gate_failures) >= 2:
        return "LOW_TRUST"
    if quality_summary.gate_failures or quality_summary.proxy_item_count >= 2:
        return "PARTIAL"
    if source_diagnostics.get("available_history_labels") and len(source_diagnostics.get("available_history_labels", [])) < 5:
        return "DEGRADED"
    if source_diagnostics.get("fallback_source_count", 0) > 0 or source_diagnostics.get("network_error"):
        return "DEGRADED"
    return "HEALTHY"


def _top_factor_titles(items: list[SourceItem], features: list[SignalFeature], positive: bool) -> list[str]:
    ranked_items = sorted(items, key=lambda item: item.impact_score, reverse=positive)
    selected = [item.title for item in ranked_items if (item.impact_score > 0.15 if positive else item.impact_score < -0.15)][:3]
    feature_selected = [
        feature.name.replace("_", " ")
        for feature in sorted(features, key=lambda feature: feature.strength, reverse=True)
        if (feature.direction == "bullish") == positive
    ][:2]
    combined = selected + feature_selected
    return combined[:5] or ["No strong bullish factors detected" if positive else "No strong bearish factors detected"]


def _format_items(items: list[SourceItem], features: list[SignalFeature]) -> str:
    if not items and not features:
        return "No items available."
    lines = [
        f"- {item.title} | {item.source} | {item.published_at} | {item.impact} | q={item.quality_score:.2f} | {item.summary}"
        for item in items[:8]
    ]
    lines.extend(
        f"- feature: {feature.name} | {feature.direction} | strength={feature.strength:.2f} | {feature.summary}"
        for feature in features[:5]
    )
    return "\n".join(lines)


def _source_regime_fit(categories: list[str], features: list[SignalFeature]) -> str:
    feature_names = {feature.name for feature in features}
    if "market" in categories and {"yield_rising", "vix_spike"} & feature_names:
        return "macro_shock_day"
    if "social" in categories and "retail_euphoric_tone" in feature_names:
        return "retail_momentum_day"
    return "standard"


def _dominant_regime_label(category: str, score: float, features: list[SignalFeature]) -> str:
    feature_names = {feature.name for feature in features}
    if category == "market" and {"yield_rising", "vix_spike"} & feature_names:
        return "macro_shock"
    if category == "political" and "policy_uncertainty_increase" in feature_names:
        return "policy_uncertainty"
    if category == "social" and "retail_euphoric_tone" in feature_names:
        return "retail_momentum"
    if category == "economic" and "growth_positive_macro_surprise" in feature_names:
        return "growth_relief"
    return "risk_on" if score > 0.15 else "risk_off" if score < -0.15 else "mixed"


def _feature_sign(feature: SignalFeature) -> int:
    if feature.direction == "bullish":
        return 1
    if feature.direction == "bearish":
        return -1
    return 0


def _score_stddev(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return variance ** 0.5


def _normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 1e-8) for value in values.values()) or 1.0
    return {key: max(value, 1e-8) / total for key, value in values.items()}


def _next_session_date(prediction_date: str) -> str:
    candidate = date.fromisoformat(prediction_date) + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate.isoformat()


def run_sector_agents(sector_data: dict, llm_client: BaseLLMClient) -> list[dict]:
    if not sector_data:
        return []
    
    analyzed_sectors = []
    for sector, data in sector_data.items():
        if "error" in data:
            continue
        ticker = data.get("ticker", "")
        price = data.get("price", 0.0)
        change_pct = data.get("change_pct", 0.0)
        
        fallback_summary = f"{sector} ({ticker}) changed {change_pct}% today. Baseline outlook is neutral."
        summary = llm_client.summarize(
            f"You are a sector-specific investment analyst for {sector}. Given the ETF {ticker} price is {price} with a daily change of {change_pct}%, provide a very concise bullish or bearish prediction and state if it is a recommended investment today.",
            f"Sector: {sector}, Ticker: {ticker}, Price: {price}, Change: {change_pct}%",
            fallback_summary
        )
        
        direction = "bullish" if change_pct > 0 else "bearish" if change_pct < 0 else "neutral"
        analyzed_sectors.append({
            "sector": sector,
            "ticker": ticker,
            "prediction": direction,
            "summary": summary,
            "recommended": change_pct > 0.5
        })
        
    return analyzed_sectors
