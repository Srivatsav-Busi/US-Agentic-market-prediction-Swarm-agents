from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from ..core.domain import KnowledgeGraph, knowledge_graph_from_serialized
from .analytics import build_graph_prediction_context
from .deltas import GRAPH_DELTA_COLUMNS, GraphDeltaSummary

GRAPH_FEATURE_SCHEMA_VERSION = "graph_feature_vector:v2"
GRAPH_FEATURE_COLUMNS = (
    "graph__feature_available",
    "graph__evidence_nodes_total",
    "graph__evidence_nodes_market",
    "graph__evidence_nodes_economic",
    "graph__evidence_nodes_political",
    "graph__evidence_nodes_social",
    "graph__evidence_nodes_other",
    "graph__provider_concentration_index",
    "graph__source_redundancy_ratio",
    "graph__contradiction_density",
    "graph__disconnected_corroboration_score",
    "graph__bullish_path_strength",
    "graph__bearish_path_strength",
    "graph__market_macro_connectivity",
    "graph__inflation_yields_equities_chain_strength",
    "graph__freshness_weighted_evidence_mass",
    "graph__regime_cluster_intensity_market",
    "graph__regime_cluster_intensity_economic",
    "graph__regime_cluster_intensity_political",
    "graph__regime_cluster_intensity_social",
) + GRAPH_DELTA_COLUMNS
STANDARD_GRAPH_CATEGORIES = ("market", "economic", "political", "social")
MACRO_TOKENS = ("macro", "economic", "inflation", "yield", "rates", "policy", "bond", "treasury")
MARKET_TOKENS = ("equity", "market", "stock", "index", "s&p", "nasdaq", "dow", "russell")
INFLATION_TOKENS = ("inflation", "cpi", "pce", "prices")
YIELD_TOKENS = ("yield", "treasury", "rates", "bond")
EQUITY_TOKENS = ("equity", "stock", "s&p", "nasdaq", "dow", "russell", "market")


@dataclass(frozen=True)
class GraphFeatureVector:
    prediction_date: str
    target: str
    schema_version: str
    features: dict[str, float]
    feature_groups: dict[str, str]
    sparse_graph: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sparse_graph": self.sparse_graph,
            "feature_count": len(self.features),
            "features": dict(self.features),
        }

    def with_delta_summary(self, delta_summary: GraphDeltaSummary | dict[str, Any] | None) -> GraphFeatureVector:
        if delta_summary is None:
            return self
        if isinstance(delta_summary, GraphDeltaSummary):
            delta_features = dict(delta_summary.features)
            delta_groups = dict(delta_summary.feature_groups)
        else:
            delta_features = {str(key): float(value) for key, value in (delta_summary.get("features") or {}).items()}
            delta_groups = {str(key): str(value) for key, value in (delta_summary.get("feature_groups") or {}).items()}
        return GraphFeatureVector(
            prediction_date=self.prediction_date,
            target=self.target,
            schema_version=self.schema_version,
            features={**self.features, **delta_features},
            feature_groups={**self.feature_groups, **delta_groups},
            sparse_graph=self.sparse_graph,
        )


def build_graph_feature_vector(
    graph: KnowledgeGraph | dict[str, Any] | None,
    *,
    prediction_date: str,
    target: str,
) -> GraphFeatureVector:
    typed_graph = _coerce_graph(graph)
    default_features = _default_feature_map()
    feature_groups = {name: _infer_feature_group(name) for name in GRAPH_FEATURE_COLUMNS}
    if typed_graph is None:
        return GraphFeatureVector(
            prediction_date=prediction_date,
            target=target,
            schema_version=GRAPH_FEATURE_SCHEMA_VERSION,
            features=default_features,
            feature_groups=feature_groups,
            sparse_graph=True,
        )

    graph_prediction_context = build_graph_prediction_context(typed_graph)
    entity_by_id = {entity.entity_id: entity for entity in typed_graph.entities}
    evidence_entities = [entity for entity in typed_graph.entities if entity.entity_type == "Evidence"]
    category_counts: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    duplicate_clusters: Counter[str] = Counter()
    freshness_weighted_evidence_mass = 0.0
    evidence_nodes_by_id = {entity.entity_id for entity in evidence_entities}
    incoming_evidence_links: Counter[str] = Counter()
    regime_mass_by_category: defaultdict[str, float] = defaultdict(float)

    for entity in evidence_entities:
        properties = entity.properties or {}
        category = _normalize_category(properties.get("category"))
        category_counts[category] += 1
        provider = str(properties.get("source") or "unknown").strip().lower() or "unknown"
        provider_counts[provider] += 1
        duplicate_cluster = str(properties.get("duplicate_cluster") or "").strip()
        if duplicate_cluster:
            duplicate_clusters[duplicate_cluster] += 1
        impact = abs(float(properties.get("impact_score") or 0.0))
        freshness = _clamp(float(properties.get("freshness_score") or 0.5))
        credibility = _clamp(float(properties.get("credibility_score") or 0.5))
        quality = _clamp(float(properties.get("quality_score") or 0.5))
        freshness_weighted_evidence_mass += max(0.05, impact) * freshness * (0.5 + 0.25 * credibility + 0.25 * quality)
        regime_mass_by_category[category] += max(0.05, impact) * max(0.2, credibility)

    market_macro_edges = 0
    inflation_edges = 0
    yield_edges = 0
    equities_edges = 0
    chain_hits = 0

    for relationship in typed_graph.relationships:
        if relationship.source_entity_id in evidence_nodes_by_id:
            incoming_evidence_links[relationship.source_entity_id] += 1
        if relationship.target_entity_id in evidence_nodes_by_id:
            incoming_evidence_links[relationship.target_entity_id] += 1
        source_text = _entity_search_blob(entity_by_id.get(relationship.source_entity_id))
        target_text = _entity_search_blob(entity_by_id.get(relationship.target_entity_id))
        pair_text = f"{source_text} {target_text}".strip()
        if _matches_any(source_text, MARKET_TOKENS) and _matches_any(target_text, MACRO_TOKENS):
            market_macro_edges += 1
        elif _matches_any(source_text, MACRO_TOKENS) and _matches_any(target_text, MARKET_TOKENS):
            market_macro_edges += 1
        if _matches_any(pair_text, INFLATION_TOKENS):
            inflation_edges += 1
        if _matches_any(pair_text, YIELD_TOKENS):
            yield_edges += 1
        if _matches_any(pair_text, EQUITY_TOKENS):
            equities_edges += 1
        if (
            (_matches_any(source_text, INFLATION_TOKENS) and _matches_any(target_text, YIELD_TOKENS))
            or (_matches_any(source_text, YIELD_TOKENS) and _matches_any(target_text, EQUITY_TOKENS))
            or (_matches_any(source_text, INFLATION_TOKENS) and _matches_any(target_text, EQUITY_TOKENS))
        ):
            chain_hits += 1

    evidence_count = len(evidence_entities)
    linked_evidence_count = sum(1 for entity_id in evidence_nodes_by_id if incoming_evidence_links.get(entity_id, 0) > 0)
    disconnected_corroboration = 0.0
    if evidence_count:
        disconnected_corroboration = max(
            0.0,
            min(
                1.0,
                (linked_evidence_count / evidence_count) * (1.0 - float(graph_prediction_context.evidence_adjustments.get("provider_concentration_index", 0.0))),
            ),
        )

    provider_concentration = 0.0
    if evidence_count and provider_counts:
        provider_concentration = sum((count / evidence_count) ** 2 for count in provider_counts.values())

    redundancy_ratio = 0.0
    if evidence_count:
        repeated_provider_items = sum(max(0, count - 1) for count in provider_counts.values())
        duplicate_items = sum(max(0, count - 1) for count in duplicate_clusters.values())
        redundancy_ratio = min(1.0, (repeated_provider_items + duplicate_items) / evidence_count)

    contradiction_density = min(
        1.0,
        float(graph_prediction_context.priors.contradiction_score) * (0.6 + min(len(typed_graph.relationships), 20) / 50.0),
    )
    market_macro_connectivity = min(1.0, market_macro_edges / max(len(typed_graph.relationships), 1))
    inflation_yields_equities_chain_strength = min(
        1.0,
        (min(inflation_edges, yield_edges, equities_edges) + chain_hits) / max(len(typed_graph.relationships), 1),
    )
    regime_intensity_denominator = max(sum(regime_mass_by_category.values()), 1.0)

    features = {
        "graph__feature_available": 1.0,
        "graph__evidence_nodes_total": float(evidence_count),
        "graph__evidence_nodes_market": float(category_counts.get("market", 0)),
        "graph__evidence_nodes_economic": float(category_counts.get("economic", 0)),
        "graph__evidence_nodes_political": float(category_counts.get("political", 0)),
        "graph__evidence_nodes_social": float(category_counts.get("social", 0)),
        "graph__evidence_nodes_other": float(sum(count for category, count in category_counts.items() if category not in STANDARD_GRAPH_CATEGORIES)),
        "graph__provider_concentration_index": round(provider_concentration, 6),
        "graph__source_redundancy_ratio": round(redundancy_ratio, 6),
        "graph__contradiction_density": round(contradiction_density, 6),
        "graph__disconnected_corroboration_score": round(disconnected_corroboration, 6),
        "graph__bullish_path_strength": float(graph_prediction_context.priors.bullish_path_strength),
        "graph__bearish_path_strength": float(graph_prediction_context.priors.bearish_path_strength),
        "graph__market_macro_connectivity": round(market_macro_connectivity, 6),
        "graph__inflation_yields_equities_chain_strength": round(inflation_yields_equities_chain_strength, 6),
        "graph__freshness_weighted_evidence_mass": round(freshness_weighted_evidence_mass, 6),
        "graph__regime_cluster_intensity_market": round(regime_mass_by_category.get("market", 0.0) / regime_intensity_denominator, 6),
        "graph__regime_cluster_intensity_economic": round(regime_mass_by_category.get("economic", 0.0) / regime_intensity_denominator, 6),
        "graph__regime_cluster_intensity_political": round(regime_mass_by_category.get("political", 0.0) / regime_intensity_denominator, 6),
        "graph__regime_cluster_intensity_social": round(regime_mass_by_category.get("social", 0.0) / regime_intensity_denominator, 6),
    }
    for column in GRAPH_FEATURE_COLUMNS:
        features.setdefault(column, 0.0)

    return GraphFeatureVector(
        prediction_date=prediction_date,
        target=target,
        schema_version=GRAPH_FEATURE_SCHEMA_VERSION,
        features=features,
        feature_groups=feature_groups,
        sparse_graph=graph_prediction_context.priors.sparse_graph,
    )


def _coerce_graph(graph: KnowledgeGraph | dict[str, Any] | None) -> KnowledgeGraph | None:
    if graph is None:
        return None
    if isinstance(graph, KnowledgeGraph):
        return graph
    if isinstance(graph, dict):
        try:
            return knowledge_graph_from_serialized(graph)
        except Exception:
            return None
    return None


def _default_feature_map() -> dict[str, float]:
    return {name: 0.0 for name in GRAPH_FEATURE_COLUMNS}


def _infer_feature_group(feature_name: str) -> str:
    if feature_name.startswith("graph_delta__"):
        return "delta"
    if "evidence_nodes" in feature_name:
        return "counts"
    if "intensity" in feature_name:
        return "regime"
    if "strength" in feature_name or "density" in feature_name or "connectivity" in feature_name:
        return "structure"
    if "freshness" in feature_name:
        return "quality"
    return "availability"


def _normalize_category(value: Any) -> str:
    category = str(value or "other").strip().lower()
    if category in STANDARD_GRAPH_CATEGORIES:
        return category
    if category in {"macro", "economy", "economic"}:
        return "economic"
    return "other"


def _entity_search_blob(entity: Any) -> str:
    if entity is None:
        return ""
    properties = getattr(entity, "properties", {}) or {}
    parts = [
        str(getattr(entity, "entity_type", "") or ""),
        str(getattr(entity, "name", "") or ""),
        str(getattr(entity, "canonical_name", "") or ""),
        str(getattr(entity, "summary", "") or ""),
        str(properties.get("category") or ""),
        str(properties.get("instrument") or ""),
        str(properties.get("source") or ""),
    ]
    return " ".join(part.strip().lower() for part in parts if part)


def _matches_any(text: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text for token in tokens)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
