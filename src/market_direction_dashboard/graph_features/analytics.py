from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.domain import KnowledgeGraph, knowledge_graph_from_serialized


@dataclass(frozen=True)
class GraphPredictionPriors:
    bullish_path_strength: float = 0.5
    bearish_path_strength: float = 0.5
    consensus_score: float = 0.0
    contradiction_score: float = 0.0
    contagion_score: float = 0.0
    credibility_weighted_pressure: float = 0.0
    cross_category_connectivity: float = 0.0
    freshness_propagation_score: float = 0.0
    influence_weight: float = 0.0
    sparse_graph: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphQualityAdjustments:
    cluster_redundancy_penalty: float = 0.0
    independent_corroboration_boost: float = 0.0
    contradiction_penalty: float = 0.0
    stale_cluster_penalty: float = 0.0
    source_monoculture_penalty: float = 0.0
    graph_quality_score: float = 0.0
    severe_graph_risk: bool = False
    item_adjustments: dict[str, dict[str, Any]] = field(default_factory=dict)
    cluster_summaries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphPredictionContext:
    priors: GraphPredictionPriors
    evidence_adjustments: dict[str, float] = field(default_factory=dict)
    conflict_summary: dict[str, Any] = field(default_factory=dict)
    feature_summary: dict[str, Any] = field(default_factory=dict)
    quality_adjustments: GraphQualityAdjustments = field(default_factory=GraphQualityAdjustments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_priors": self.priors.to_dict(),
            "graph_evidence_adjustments": dict(self.evidence_adjustments),
            "graph_conflict_summary": dict(self.conflict_summary),
            "graph_feature_summary": dict(self.feature_summary),
            "graph_quality_summary": self.quality_adjustments.to_dict(),
        }


def build_graph_prediction_context(graph: KnowledgeGraph | dict[str, Any] | None) -> GraphPredictionContext:
    typed_graph = _coerce_graph(graph)
    if typed_graph is None:
        return GraphPredictionContext(priors=GraphPredictionPriors())

    entity_by_id = {entity.entity_id: entity for entity in typed_graph.entities}
    evidence_entities = [entity for entity in typed_graph.entities if entity.entity_type == "Evidence"]
    feature_entities = [entity for entity in typed_graph.entities if entity.entity_type == "SignalFeature"]
    provider_entities = [entity for entity in typed_graph.entities if entity.entity_type == "EvidenceSource"]
    relationships = typed_graph.relationships

    directional_masses = Counter({"bullish": 0.0, "bearish": 0.0, "neutral": 0.0})
    categories: Counter[str] = Counter()
    providers: Counter[str] = Counter()
    duplicate_clusters: Counter[str] = Counter()
    unique_instruments: set[str] = set()
    freshness_total = 0.0
    credibility_total = 0.0
    quality_total = 0.0

    cluster_entities: defaultdict[str, list[Any]] = defaultdict(list)
    for entity in evidence_entities:
        properties = entity.properties or {}
        direction = _normalize_direction(properties.get("direction"))
        impact = abs(float(properties.get("impact_score") or 0.0))
        credibility = _clamp(float(properties.get("credibility_score") or 0.5))
        quality = _clamp(float(properties.get("quality_score") or 0.5))
        freshness = _clamp(float(properties.get("freshness_score") or 0.5))
        weight = max(0.05, impact) * max(0.2, credibility) * max(0.2, quality)
        directional_masses[direction] += weight
        category = str(properties.get("category") or "unknown")
        categories[category] += 1
        provider = str(properties.get("source") or "unknown")
        providers[provider] += 1
        duplicate_cluster = str(properties.get("duplicate_cluster") or "").strip()
        if duplicate_cluster:
            duplicate_clusters[duplicate_cluster] += 1
        cluster_entities[duplicate_cluster or entity.entity_id].append(entity)
        instrument = str(properties.get("instrument") or "").strip()
        if instrument:
            unique_instruments.add(instrument)
        freshness_total += freshness
        credibility_total += credibility
        quality_total += quality

    for entity in feature_entities:
        properties = entity.properties or {}
        direction = _normalize_direction(properties.get("direction"))
        strength = max(0.0, float(properties.get("strength") or 0.0))
        directional_masses[direction] += strength * 0.35
        category = str(properties.get("category") or "unknown")
        categories[category] += 1

    evidence_count = len(evidence_entities)
    feature_count = len(feature_entities)
    provider_count = max(len(providers), len(provider_entities))
    relationship_count = len(relationships)
    category_count = len([name for name in categories if name and name != "unknown"])
    total_directional = directional_masses["bullish"] + directional_masses["bearish"] + directional_masses["neutral"]
    bullish_strength = directional_masses["bullish"] / total_directional if total_directional else 0.5
    bearish_strength = directional_masses["bearish"] / total_directional if total_directional else 0.5
    directional_balance = abs(directional_masses["bullish"] - directional_masses["bearish"]) / max(total_directional, 1e-8)
    contradiction_score = min(
        1.0,
        (2.0 * min(directional_masses["bullish"], directional_masses["bearish"]) / max(total_directional, 1e-8))
        + max(0.0, 0.08 * (sum(1 for cluster_size in duplicate_clusters.values() if cluster_size > 1))),
    )
    consensus_score = max(0.0, min(1.0, directional_balance * 0.75 + min(provider_count, 6) / 12.0))
    contagion_score = max(
        0.0,
        min(
            1.0,
            relationship_count / max(3.0 * max(len(typed_graph.entities), 1), 1.0),
        ),
    )
    cross_category_connectivity = max(
        0.0,
        min(
            1.0,
            (category_count / 4.0) * 0.65 + min(len(unique_instruments), 5) / 10.0 + contagion_score * 0.25,
        ),
    )
    avg_freshness = freshness_total / evidence_count if evidence_count else 0.0
    avg_credibility = credibility_total / evidence_count if evidence_count else 0.0
    avg_quality = quality_total / evidence_count if evidence_count else 0.0
    pressure_raw = bullish_strength - bearish_strength
    credibility_weighted_pressure = max(
        -1.0,
        min(1.0, pressure_raw * (0.35 + avg_credibility * 0.35 + avg_quality * 0.30)),
    )
    freshness_propagation_score = max(
        0.0,
        min(1.0, avg_freshness * 0.6 + contagion_score * 0.2 + cross_category_connectivity * 0.2),
    )
    sparse_graph = evidence_count < 3 or relationship_count < 4
    influence_weight = max(
        0.0,
        min(
            0.18,
            (
                0.05
                + min(evidence_count, 8) * 0.01
                + min(provider_count, 5) * 0.008
                + cross_category_connectivity * 0.04
                + avg_quality * 0.03
                - contradiction_score * 0.08
            ),
        ),
    )
    if sparse_graph or avg_quality < 0.25:
        influence_weight *= 0.25

    provider_concentration = 0.0
    if providers:
        provider_concentration = sum((count / evidence_count) ** 2 for count in providers.values())
    redundancy_ratio = 0.0
    if evidence_count:
        repeated_provider_items = sum(max(0, count - 1) for count in providers.values())
        duplicate_items = sum(max(0, count - 1) for count in duplicate_clusters.values())
        redundancy_ratio = min(1.0, (repeated_provider_items + duplicate_items) / max(evidence_count, 1))

    duplicate_penalty = min(0.16, redundancy_ratio * 0.12 + provider_concentration * 0.06)
    independent_corroboration_boost = min(
        0.12,
        max(0.0, (1.0 - provider_concentration) * 0.05 + consensus_score * 0.04 + cross_category_connectivity * 0.03),
    )
    contradiction_penalty = min(0.16, contradiction_score * 0.14)
    cluster_summaries, item_adjustments, stale_cluster_penalty, monoculture_penalty, graph_quality_score, severe_graph_risk = _build_quality_adjustments(
        cluster_entities=cluster_entities,
        provider_concentration=provider_concentration,
        contradiction_score=contradiction_score,
        redundancy_ratio=redundancy_ratio,
    )

    priors = GraphPredictionPriors(
        bullish_path_strength=round(bullish_strength, 4),
        bearish_path_strength=round(bearish_strength, 4),
        consensus_score=round(consensus_score, 4),
        contradiction_score=round(contradiction_score, 4),
        contagion_score=round(contagion_score, 4),
        credibility_weighted_pressure=round(credibility_weighted_pressure, 4),
        cross_category_connectivity=round(cross_category_connectivity, 4),
        freshness_propagation_score=round(freshness_propagation_score, 4),
        influence_weight=round(influence_weight, 4),
        sparse_graph=sparse_graph,
    )
    evidence_adjustments = {
        "duplicate_penalty": round(duplicate_penalty, 4),
        "independent_corroboration_boost": round(independent_corroboration_boost, 4),
        "contradiction_penalty": round(contradiction_penalty, 4),
        "stale_cluster_penalty": round(stale_cluster_penalty, 4),
        "source_monoculture_penalty": round(monoculture_penalty, 4),
        "provider_concentration_index": round(provider_concentration, 4),
        "source_redundancy_ratio": round(redundancy_ratio, 4),
    }
    conflict_summary = {
        "bullish_evidence_mass": round(directional_masses["bullish"], 4),
        "bearish_evidence_mass": round(directional_masses["bearish"], 4),
        "neutral_evidence_mass": round(directional_masses["neutral"], 4),
        "contradiction_score": round(contradiction_score, 4),
        "consensus_score": round(consensus_score, 4),
        "duplicate_cluster_count": sum(1 for count in duplicate_clusters.values() if count > 1),
        "dominant_direction": "bullish" if directional_masses["bullish"] >= directional_masses["bearish"] else "bearish",
    }
    feature_summary = {
        "entity_count": len(typed_graph.entities),
        "relationship_count": relationship_count,
        "evidence_node_count": evidence_count,
        "feature_node_count": feature_count,
        "provider_count": provider_count,
        "category_count": category_count,
        "instrument_count": len(unique_instruments),
        "sparse_graph": sparse_graph,
        "graph_density": round(relationship_count / max(len(typed_graph.entities), 1), 4),
        "avg_evidence_quality": round(avg_quality, 4),
    }
    return GraphPredictionContext(
        priors=priors,
        evidence_adjustments=evidence_adjustments,
        conflict_summary=conflict_summary,
        feature_summary=feature_summary,
        quality_adjustments=GraphQualityAdjustments(
            cluster_redundancy_penalty=round(duplicate_penalty, 4),
            independent_corroboration_boost=round(independent_corroboration_boost, 4),
            contradiction_penalty=round(contradiction_penalty, 4),
            stale_cluster_penalty=round(stale_cluster_penalty, 4),
            source_monoculture_penalty=round(monoculture_penalty, 4),
            graph_quality_score=round(graph_quality_score, 4),
            severe_graph_risk=severe_graph_risk,
            item_adjustments=item_adjustments,
            cluster_summaries=cluster_summaries,
        ),
    )


def _build_quality_adjustments(
    *,
    cluster_entities: dict[str, list[Any]],
    provider_concentration: float,
    contradiction_score: float,
    redundancy_ratio: float,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], float, float, float, bool]:
    cluster_summaries: list[dict[str, Any]] = []
    item_adjustments: dict[str, dict[str, Any]] = {}
    stale_penalty_total = 0.0
    monoculture_penalty = min(0.18, max(0.0, provider_concentration - 0.35) * 0.28 + redundancy_ratio * 0.08)
    for cluster_id, entities in cluster_entities.items():
        providers = {str((entity.properties or {}).get("source") or "unknown").strip().lower() for entity in entities}
        directions = Counter(_normalize_direction((entity.properties or {}).get("direction")) for entity in entities)
        avg_freshness = (
            sum(_clamp(float((entity.properties or {}).get("freshness_score") or 0.5)) for entity in entities) / max(len(entities), 1)
        )
        duplicate_pressure = max(0.0, (len(entities) - len(providers)) / max(len(entities), 1))
        contradiction_density = (2.0 * min(directions["bullish"], directions["bearish"])) / max(sum(directions.values()), 1)
        stale_penalty = max(0.0, (0.45 - avg_freshness) * 0.28) if avg_freshness < 0.45 else 0.0
        corroboration_boost = 0.0
        if len(entities) >= 2 and len(providers) >= 2 and contradiction_density < 0.25 and avg_freshness >= 0.35:
            corroboration_boost = min(0.14, 0.05 + (len(providers) - 1) * 0.025 + avg_freshness * 0.04)
        redundancy_penalty = min(0.18, duplicate_pressure * 0.18 + max(0.0, (1 - len(providers) / max(len(entities), 1))) * 0.06)
        contradiction_penalty = min(0.18, contradiction_density * 0.18 + contradiction_score * 0.06)
        net_adjustment = max(-0.35, min(0.18, corroboration_boost - redundancy_penalty - contradiction_penalty - stale_penalty - monoculture_penalty * 0.7))
        summary = {
            "cluster_id": cluster_id,
            "item_count": len(entities),
            "provider_count": len(providers),
            "avg_freshness": round(avg_freshness, 4),
            "duplicate_pressure": round(duplicate_pressure, 4),
            "contradiction_density": round(contradiction_density, 4),
            "redundancy_penalty": round(redundancy_penalty, 4),
            "corroboration_boost": round(corroboration_boost, 4),
            "stale_penalty": round(stale_penalty, 4),
            "net_adjustment": round(net_adjustment, 4),
        }
        cluster_summaries.append(summary)
        stale_penalty_total += stale_penalty
        for entity in entities:
            reasons: list[str] = []
            if redundancy_penalty > 0.03:
                reasons.append("cluster_redundancy")
            if corroboration_boost > 0.03:
                reasons.append("independent_corroboration")
            if contradiction_penalty > 0.03:
                reasons.append("cluster_contradiction")
            if stale_penalty > 0.03:
                reasons.append("stale_cluster")
            if monoculture_penalty > 0.03:
                reasons.append("source_monoculture")
            item_adjustments[entity.entity_id] = {
                "adjustment": round(net_adjustment, 4),
                "quality_multiplier": round(max(0.65, min(1.15, 1.0 + net_adjustment)), 4),
                "cluster_id": cluster_id,
                "reasons": reasons,
            }
    graph_quality_score = max(
        0.0,
        min(
            1.0,
            0.72
            + sum(cluster.get("corroboration_boost", 0.0) for cluster in cluster_summaries) * 0.35
            - sum(cluster.get("redundancy_penalty", 0.0) for cluster in cluster_summaries) * 0.28
            - sum(cluster.get("stale_penalty", 0.0) for cluster in cluster_summaries) * 0.28
            - contradiction_score * 0.25
            - monoculture_penalty * 0.35,
        ),
    )
    severe_graph_risk = contradiction_score >= 0.65 or monoculture_penalty >= 0.14 or stale_penalty_total >= 0.18
    cluster_summaries.sort(key=lambda item: abs(float(item.get("net_adjustment", 0.0))), reverse=True)
    return cluster_summaries[:12], item_adjustments, min(0.18, stale_penalty_total), monoculture_penalty, graph_quality_score, severe_graph_risk


def _coerce_graph(graph: KnowledgeGraph | dict[str, Any] | None) -> KnowledgeGraph | None:
    if graph is None:
        return None
    if isinstance(graph, KnowledgeGraph):
        return graph
    if isinstance(graph, dict):
        return knowledge_graph_from_serialized(graph)
    return None


def _normalize_direction(value: Any) -> str:
    direction = str(value or "").strip().lower()
    if direction in {"bullish", "positive", "up"}:
        return "bullish"
    if direction in {"bearish", "negative", "down"}:
        return "bearish"
    return "neutral"


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
