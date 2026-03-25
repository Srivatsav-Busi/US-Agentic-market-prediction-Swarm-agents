from datetime import date, timedelta

from market_direction_dashboard.core.domain import Entity, KnowledgeGraph, Relationship
from market_direction_dashboard.graph_features import build_graph_delta_summary, build_graph_feature_vector, build_graph_prediction_context
from market_direction_dashboard.models import DataQualitySummary, SignalFeature, SourceItem
from market_direction_dashboard.statistical_engine import build_statistical_decision


def _history_series(start_value: float, step: float, days: int = 90) -> list[dict[str, float | str]]:
    start = date(2025, 12, 1)
    return [
        {"date": (start + timedelta(days=index)).isoformat(), "value": start_value + index * step}
        for index in range(days)
    ]


def test_build_graph_prediction_context_sparse_graph_collapses_influence() -> None:
    graph = KnowledgeGraph(
        graph_id="graph-1",
        entities=[
            Entity(
                entity_id="evidence:1",
                entity_type="Evidence",
                name="Single evidence item",
                properties={
                    "direction": "bullish",
                    "impact_score": 0.4,
                    "credibility_score": 0.7,
                    "quality_score": 0.7,
                    "freshness_score": 0.8,
                    "category": "market",
                    "source": "Reuters",
                },
            ),
            Entity(entity_id="provider:reuters", entity_type="EvidenceSource", name="Reuters", properties={"provider": "Reuters"}),
        ],
        relationships=[
            Relationship(
                relationship_id="provider:reuters:evidence:1",
                relationship_type="PUBLISHED",
                source_entity_id="provider:reuters",
                target_entity_id="evidence:1",
            )
        ],
    )

    context = build_graph_prediction_context(graph)

    assert context.priors.sparse_graph is True
    assert context.priors.influence_weight <= 0.05
    assert context.feature_summary["evidence_node_count"] == 1


def test_build_statistical_decision_emits_graph_diagnostics() -> None:
    graph = KnowledgeGraph(
        graph_id="graph-2",
        entities=[
            Entity(
                entity_id="evidence:bull",
                entity_type="Evidence",
                name="Bullish breadth",
                properties={
                    "direction": "bullish",
                    "impact_score": 0.8,
                    "credibility_score": 0.9,
                    "quality_score": 0.85,
                    "freshness_score": 0.9,
                    "category": "market",
                    "source": "Reuters",
                    "instrument": "S&P 500",
                },
            ),
            Entity(
                entity_id="evidence:bear",
                entity_type="Evidence",
                name="Bearish macro",
                properties={
                    "direction": "bearish",
                    "impact_score": 0.3,
                    "credibility_score": 0.8,
                    "quality_score": 0.8,
                    "freshness_score": 0.75,
                    "category": "economic",
                    "source": "Bloomberg",
                    "instrument": "UST10Y",
                },
            ),
            Entity(
                entity_id="evidence:confirm",
                entity_type="Evidence",
                name="Independent confirmation",
                properties={
                    "direction": "bullish",
                    "impact_score": 0.4,
                    "credibility_score": 0.85,
                    "quality_score": 0.8,
                    "freshness_score": 0.8,
                    "category": "economic",
                    "source": "WSJ",
                    "instrument": "DXY",
                },
            ),
            Entity(
                entity_id="feature:momentum",
                entity_type="SignalFeature",
                name="momentum",
                properties={"direction": "bullish", "strength": 0.6, "category": "market"},
            ),
            Entity(entity_id="provider:reuters", entity_type="EvidenceSource", name="Reuters", properties={"provider": "Reuters"}),
            Entity(entity_id="provider:bloomberg", entity_type="EvidenceSource", name="Bloomberg", properties={"provider": "Bloomberg"}),
            Entity(entity_id="provider:wsj", entity_type="EvidenceSource", name="WSJ", properties={"provider": "WSJ"}),
            Entity(entity_id="run:test", entity_type="PredictionRun", name="S&P 500"),
        ],
        relationships=[
            Relationship("r1", "PUBLISHED", "provider:reuters", "evidence:bull"),
            Relationship("r2", "PUBLISHED", "provider:bloomberg", "evidence:bear"),
            Relationship("r3", "PUBLISHED", "provider:wsj", "evidence:confirm"),
            Relationship("r4", "GENERATED_FEATURE", "evidence:bull", "feature:momentum"),
            Relationship("r5", "GENERATED_FEATURE", "evidence:confirm", "feature:momentum"),
            Relationship("r6", "INFLUENCES_RUN", "feature:momentum", "run:test"),
        ],
    )
    context = build_graph_prediction_context(graph)
    snapshot = {
        "history": {
            "S&P 500": _history_series(100, 0.8),
            "VIX": _history_series(20, 0.05),
            "UST10Y": _history_series(4, 0.01),
            "DXY": _history_series(100, 0.03),
        }
    }
    items = [
        SourceItem(
            title="Bullish breadth",
            source="Reuters",
            category="market",
            published_at="2026-03-18",
            url="",
            summary="",
            impact="high",
            impact_score=0.7,
            direction="bullish",
            freshness_score=0.9,
            credibility_score=0.85,
            quality_score=0.9,
        ),
        SourceItem(
            title="Bearish macro",
            source="Bloomberg",
            category="economic",
            published_at="2026-03-18",
            url="",
            summary="",
            impact="medium",
            impact_score=-0.2,
            direction="bearish",
            freshness_score=0.8,
            credibility_score=0.8,
            quality_score=0.8,
        ),
        SourceItem(
            title="Independent confirmation",
            source="WSJ",
            category="economic",
            published_at="2026-03-18",
            url="",
            summary="",
            impact="medium",
            impact_score=0.35,
            direction="bullish",
            freshness_score=0.8,
            credibility_score=0.82,
            quality_score=0.8,
        ),
    ]
    features = [SignalFeature(name="momentum", direction="bullish", strength=0.5, category="market")]
    quality = DataQualitySummary(
        valid_item_count=3,
        rejected_item_count=0,
        duplicate_item_count=0,
        stale_item_count=0,
        malformed_item_count=0,
        proxy_item_count=0,
        distinct_provider_count=3,
        average_quality_score=0.8,
    )

    decision = build_statistical_decision(
        target="S&P 500",
        snapshot=snapshot,
        items=items,
        features=features,
        quality_summary=quality,
        config={"minimum_history_rows": 45},
        graph_prediction_context=context,
    )

    assert decision.graph_priors["influence_weight"] >= 0.0
    assert "duplicate_penalty" in decision.graph_evidence_adjustments
    assert "contradiction_score" in decision.graph_conflict_summary
    assert any(step["stage"] == "graph_prior_integration" for step in decision.trace_steps)


def test_build_graph_delta_summary_detects_reversal_and_acceleration() -> None:
    previous_features = {
        "graph__evidence_nodes_total": 4.0,
        "graph__provider_concentration_index": 0.45,
        "graph__source_redundancy_ratio": 0.1,
        "graph__contradiction_density": 0.15,
        "graph__disconnected_corroboration_score": 0.35,
        "graph__bullish_path_strength": 0.75,
        "graph__bearish_path_strength": 0.2,
        "graph__market_macro_connectivity": 0.2,
        "graph__freshness_weighted_evidence_mass": 1.2,
        "graph__regime_cluster_intensity_market": 0.6,
        "graph__regime_cluster_intensity_economic": 0.2,
        "graph__regime_cluster_intensity_political": 0.1,
        "graph__regime_cluster_intensity_social": 0.1,
    }
    current_features = {
        "graph__evidence_nodes_total": 6.0,
        "graph__provider_concentration_index": 0.3,
        "graph__source_redundancy_ratio": 0.05,
        "graph__contradiction_density": 0.28,
        "graph__disconnected_corroboration_score": 0.5,
        "graph__bullish_path_strength": 0.25,
        "graph__bearish_path_strength": 0.68,
        "graph__market_macro_connectivity": 0.38,
        "graph__freshness_weighted_evidence_mass": 1.8,
        "graph__regime_cluster_intensity_market": 0.25,
        "graph__regime_cluster_intensity_economic": 0.5,
        "graph__regime_cluster_intensity_political": 0.15,
        "graph__regime_cluster_intensity_social": 0.1,
    }

    delta = build_graph_delta_summary(
        prediction_date="2026-03-18",
        target="S&P 500",
        current_features=current_features,
        previous_summary={"prediction_date": "2026-03-17", "features": previous_features},
    )

    assert delta.delta_available is True
    assert delta.previous_prediction_date == "2026-03-17"
    assert delta.narrative_reversal_flag is True
    assert delta.theme_acceleration > 0.0
    assert delta.features["graph_delta__available"] == 1.0


def test_graph_delta_features_flow_into_statistical_decision() -> None:
    graph = KnowledgeGraph(
        graph_id="graph-3",
        entities=[
            Entity(
                entity_id="evidence:1",
                entity_type="Evidence",
                name="Macro stress",
                properties={
                    "direction": "bearish",
                    "impact_score": 0.6,
                    "credibility_score": 0.85,
                    "quality_score": 0.8,
                    "freshness_score": 0.85,
                    "category": "economic",
                    "source": "Reuters",
                },
            ),
            Entity(
                entity_id="evidence:2",
                entity_type="Evidence",
                name="Credit spread widening",
                properties={
                    "direction": "bearish",
                    "impact_score": 0.5,
                    "credibility_score": 0.8,
                    "quality_score": 0.78,
                    "freshness_score": 0.8,
                    "category": "market",
                    "source": "Bloomberg",
                },
            ),
            Entity(entity_id="provider:reuters", entity_type="EvidenceSource", name="Reuters", properties={"provider": "Reuters"}),
            Entity(entity_id="provider:bloomberg", entity_type="EvidenceSource", name="Bloomberg", properties={"provider": "Bloomberg"}),
            Entity(entity_id="run:test", entity_type="PredictionRun", name="S&P 500"),
        ],
        relationships=[
            Relationship("r1", "PUBLISHED", "provider:reuters", "evidence:1"),
            Relationship("r2", "PUBLISHED", "provider:bloomberg", "evidence:2"),
            Relationship("r3", "USES_EVIDENCE", "run:test", "evidence:1"),
            Relationship("r4", "USES_EVIDENCE", "run:test", "evidence:2"),
        ],
    )
    context = build_graph_prediction_context(graph)
    vector = build_graph_feature_vector(graph, prediction_date="2026-03-18", target="S&P 500")
    delta = build_graph_delta_summary(
        prediction_date="2026-03-18",
        target="S&P 500",
        current_features=vector.features,
        previous_summary={
            "prediction_date": "2026-03-17",
            "features": {
                **vector.features,
                "graph__bullish_path_strength": 0.8,
                "graph__bearish_path_strength": 0.1,
                "graph__regime_cluster_intensity_market": 0.7,
                "graph__regime_cluster_intensity_economic": 0.1,
            },
        },
    )
    snapshot = {
        "history": {
            "S&P 500": _history_series(100, 0.4),
            "VIX": _history_series(20, 0.03),
            "UST10Y": _history_series(4, 0.005),
            "DXY": _history_series(100, 0.02),
        }
    }
    items = [
        SourceItem(
            title="Macro stress",
            source="Reuters",
            category="economic",
            published_at="2026-03-18",
            url="",
            summary="",
            impact="high",
            impact_score=-0.45,
            direction="bearish",
            freshness_score=0.85,
            credibility_score=0.85,
            quality_score=0.8,
        ),
        SourceItem(
            title="Credit spread widening",
            source="Bloomberg",
            category="market",
            published_at="2026-03-18",
            url="",
            summary="",
            impact="medium",
            impact_score=-0.35,
            direction="bearish",
            freshness_score=0.8,
            credibility_score=0.8,
            quality_score=0.78,
        ),
    ]
    quality = DataQualitySummary(
        valid_item_count=2,
        rejected_item_count=0,
        duplicate_item_count=0,
        stale_item_count=0,
        malformed_item_count=0,
        proxy_item_count=0,
        distinct_provider_count=2,
        average_quality_score=0.8,
    )

    decision = build_statistical_decision(
        target="S&P 500",
        snapshot=snapshot,
        items=items,
        features=[],
        quality_summary=quality,
        config={"minimum_history_rows": 45},
        graph_prediction_context=context,
        graph_delta_summary=delta.to_dict(),
    )

    assert decision.graph_delta_summary["delta_available"] is True
    assert any(step["stage"] == "graph_delta_integration" for step in decision.trace_steps)
