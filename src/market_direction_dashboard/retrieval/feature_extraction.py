from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ..core.domain import Entity, KnowledgeGraph
from ..models import SignalFeature, SourceItem
from .graph_retriever import InMemoryGraphRetriever


EQUITY_INSTRUMENTS = {"S&P 500", "NASDAQ 100", "DOW JONES", "RUSSELL 2000"}
YIELD_INSTRUMENT_TERMS = ("10 YR", "10YR", "TREASURY", "YIELD", "BOND")
INFLATION_TERMS = ("inflation", "cpi", "pce", "prices", "sticky")
POLICY_TERMS = ("policy", "tariff", "sanction", "geopolit", "war", "shutdown", "election", "uncertainty")
CREDIT_TERMS = ("credit", "spread", "default", "bank", "funding", "liquidity", "debt")
GROWTH_TERMS = ("growth", "recession", "slowdown", "demand", "manufacturing", "consumer")
MACRO_TERMS = INFLATION_TERMS + ("labor", "employment", "fed", "rates", "macro", "economy")


def build_retrieval_assisted_features(
    *,
    graph: KnowledgeGraph | None,
    items: list[SourceItem],
    snapshot: dict[str, Any],
    base_features: list[SignalFeature],
) -> list[SignalFeature]:
    if graph is None or not graph.entities:
        return []

    evidence_entities = [entity for entity in graph.entities if entity.entity_type == "Evidence"]
    if not evidence_entities:
        return []

    retriever = InMemoryGraphRetriever(graph)
    relationships_by_id = {relationship.relationship_id: relationship for relationship in graph.relationships}
    items_by_id = {item.id: item for item in items if item.id}
    provider_by_evidence_id: dict[str, str] = {}
    for relationship in graph.relationships:
        if relationship.relationship_type == "PUBLISHED":
            provider_by_evidence_id[relationship.target_entity_id] = relationship.source_entity_id

    created: list[SignalFeature] = []
    seen_names = {feature.name for feature in base_features}

    def add_feature(
        *,
        name: str,
        direction: str,
        strength: float,
        evidence_entities_used: list[Entity],
        relationship_ids: list[str],
        category: str,
        summary: str,
        retrieval_kind: str,
        supporting_evidence_ids: list[str] | None = None,
        conflict_count: int = 0,
    ) -> None:
        if name in seen_names or strength <= 0.0:
            return
        evidence_ids = supporting_evidence_ids or [
            str(entity.properties.get("document_id") or "").strip()
            for entity in evidence_entities_used
            if str(entity.properties.get("document_id") or "").strip()
        ]
        evidence_ids = [evidence_id for evidence_id in evidence_ids if evidence_id]
        entity_ids = [entity.entity_id for entity in evidence_entities_used]
        edge_types = sorted(
            {
                relationships_by_id[relationship_id].relationship_type
                for relationship_id in relationship_ids
                if relationship_id in relationships_by_id
            }
        )
        created.append(
            SignalFeature(
                name=name,
                direction=direction,
                strength=round(min(1.0, max(0.0, strength)), 3),
                supporting_evidence_ids=evidence_ids,
                conflict_count=conflict_count,
                category=category,
                summary=summary,
                provenance={
                    "feature_pass": "pass_2",
                    "retrieval_kind": retrieval_kind,
                    "graph_entity_ids": entity_ids,
                    "graph_relationship_ids": relationship_ids,
                    "graph_edge_types": edge_types,
                },
            )
        )
        seen_names.add(name)

    category_lookup = _category_neighborhood_lookup(retriever, evidence_entities)
    contradiction_clusters = _contradiction_clusters(evidence_entities)

    inflation_entities = _match_entities(evidence_entities, items_by_id, terms=INFLATION_TERMS, category="economic")
    yield_entities = _match_entities(evidence_entities, items_by_id, instrument_terms=YIELD_INSTRUMENT_TERMS)
    equity_entities = _match_entities(evidence_entities, items_by_id, instruments=EQUITY_INSTRUMENTS, category="market")
    if inflation_entities and yield_entities and (equity_entities or _equity_snapshot_is_soft(snapshot)):
        combined = _unique_entities(inflation_entities + yield_entities + equity_entities)
        add_feature(
            name="inflation_yields_equity_pressure",
            direction="bearish",
            strength=min(1.0, 0.35 + 0.18 * len(combined)),
            evidence_entities_used=combined,
            relationship_ids=_collect_relationship_ids(retriever, combined),
            category="economic",
            summary="Graph retrieval linked inflation evidence, yield pressure, and equity-sensitive nodes into a single bearish macro chain.",
            retrieval_kind="source_to_theme_path_lookup",
        )

    policy_entities = _match_entities(evidence_entities, items_by_id, terms=POLICY_TERMS, category="political")
    credit_entities = _match_entities(evidence_entities, items_by_id, terms=CREDIT_TERMS)
    if policy_entities and credit_entities:
        combined = _unique_entities(policy_entities + credit_entities)
        add_feature(
            name="policy_uncertainty_credit_stress",
            direction="bearish",
            strength=min(1.0, 0.3 + 0.2 * min(len(policy_entities), 2) + 0.15 * min(len(credit_entities), 2)),
            evidence_entities_used=combined,
            relationship_ids=_collect_relationship_ids(retriever, combined),
            category="political",
            summary="Political uncertainty evidence connected to credit-stress nodes across the retrieval neighborhood.",
            retrieval_kind="cross_category_motif_detection",
        )

    oil_entities = _match_entities(evidence_entities, items_by_id, instruments={"WTI CRUDE OIL"}, terms=("oil", "energy"))
    usd_entities = _match_entities(evidence_entities, items_by_id, instruments={"DXY"}, terms=("usd", "dollar"))
    growth_entities = _match_entities(evidence_entities, items_by_id, terms=GROWTH_TERMS)
    if oil_entities and usd_entities and growth_entities:
        growth_conflict_count = _count_directional_conflicts(growth_entities)
        if growth_conflict_count or _growth_snapshot_is_soft(snapshot):
            combined = _unique_entities(oil_entities + usd_entities + growth_entities)
            add_feature(
                name="oil_usd_growth_conflict",
                direction="bearish",
                strength=min(1.0, 0.28 + 0.12 * len(combined) + 0.08 * growth_conflict_count),
                evidence_entities_used=combined,
                relationship_ids=_collect_relationship_ids(retriever, combined),
                category="market",
                summary="Oil, dollar, and growth evidence formed a conflicting macro motif that weakens pro-risk conviction.",
                retrieval_kind="cross_category_motif_detection",
                conflict_count=growth_conflict_count,
            )

    macro_entities = _match_entities(evidence_entities, items_by_id, terms=MACRO_TERMS, category="economic")
    macro_direction, provider_count, macro_docs = _independent_macro_confirmation(macro_entities, provider_by_evidence_id)
    if macro_direction and provider_count >= 2:
        neighborhood_support = len(category_lookup.get("economic", []))
        add_feature(
            name="independent_macro_confirmation",
            direction=macro_direction,
            strength=min(1.0, 0.24 + provider_count * 0.16 + len(macro_docs) * 0.05 + neighborhood_support * 0.01),
            evidence_entities_used=macro_docs,
            relationship_ids=_collect_relationship_ids(retriever, macro_docs),
            category="economic",
            summary=f"Independent macro evidence from {provider_count} provider clusters corroborated the {macro_direction} thesis.",
            retrieval_kind="category_neighborhood_lookup",
        )

    social_cluster_entities = _crowded_headline_entities(evidence_entities, items_by_id)
    market_confirmation = _market_confirmation_strength(base_features, equity_entities)
    if social_cluster_entities and market_confirmation < 0.35:
        add_feature(
            name="headline_crowding_without_market_confirmation",
            direction="bearish",
            strength=min(1.0, 0.3 + 0.1 * len(social_cluster_entities) + (0.35 - market_confirmation)),
            evidence_entities_used=social_cluster_entities,
            relationship_ids=_collect_relationship_ids(retriever, social_cluster_entities),
            category="social",
            summary="Crowded headline clusters were not confirmed by market-linked evidence in the graph neighborhood.",
            retrieval_kind="contradiction_cluster_detection",
            conflict_count=len(contradiction_clusters),
        )

    return created


def _category_neighborhood_lookup(
    retriever: InMemoryGraphRetriever,
    evidence_entities: list[Entity],
) -> dict[str, list[str]]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for entity in evidence_entities:
        category = str(entity.properties.get("category") or "unknown").strip().lower()
        if not category or category == "unknown":
            continue
        neighborhood = retriever.neighborhood_lookup(entity_id=entity.entity_id, hops=1)
        related_evidence_ids = sorted(
            candidate.entity_id for candidate in neighborhood.entities if candidate.entity_type == "Evidence"
        )
        by_category[category].extend(related_evidence_ids)
    return {category: sorted(set(entity_ids)) for category, entity_ids in by_category.items()}


def _collect_relationship_ids(retriever: InMemoryGraphRetriever, entities: list[Entity]) -> list[str]:
    relationship_ids: set[str] = set()
    entity_ids = [entity.entity_id for entity in entities]
    for entity_id in entity_ids:
        neighborhood = retriever.neighborhood_lookup(entity_id=entity_id, hops=1)
        relationship_ids.update(relationship.relationship_id for relationship in neighborhood.relationships)
    for index, source_id in enumerate(entity_ids):
        for target_id in entity_ids[index + 1 :]:
            relationship_ids.update(
                relationship.relationship_id
                for relationship in retriever.path_lookup(source_entity_id=source_id, target_entity_id=target_id, max_hops=3)
            )
    return sorted(relationship_ids)


def _match_entities(
    evidence_entities: list[Entity],
    items_by_id: dict[str, SourceItem],
    *,
    terms: tuple[str, ...] = (),
    instruments: set[str] | None = None,
    instrument_terms: tuple[str, ...] = (),
    category: str | None = None,
) -> list[Entity]:
    matches: list[Entity] = []
    for entity in evidence_entities:
        properties = entity.properties or {}
        document_id = str(properties.get("document_id") or "").strip()
        item = items_by_id.get(document_id)
        text = " ".join(
            [
                entity.name,
                entity.summary,
                str(properties.get("category") or ""),
                str(properties.get("instrument") or ""),
                item.title if item else "",
                item.summary if item else "",
                item.raw_text if item else "",
            ]
        ).lower()
        instrument_value = properties.get("instrument")
        instrument = str(instrument_value or (item.instrument if item else "")).strip()
        category_value = properties.get("category")
        item_category = item.category if item else ""
        if category and str(category_value or item_category).strip().lower() != category:
            continue
        if instruments and instrument in instruments:
            matches.append(entity)
            continue
        if instrument_terms and any(term in instrument.upper() for term in instrument_terms):
            matches.append(entity)
            continue
        if terms and any(term in text for term in terms):
            matches.append(entity)
    return _unique_entities(matches)


def _independent_macro_confirmation(
    macro_entities: list[Entity],
    provider_by_evidence_id: dict[str, str],
) -> tuple[str | None, int, list[Entity]]:
    by_direction: dict[str, list[Entity]] = defaultdict(list)
    providers_by_direction: dict[str, set[str]] = defaultdict(set)
    for entity in macro_entities:
        direction = _normalize_direction(entity.properties.get("direction"))
        if direction not in {"bullish", "bearish"}:
            continue
        by_direction[direction].append(entity)
        provider = provider_by_evidence_id.get(entity.entity_id)
        if provider:
            providers_by_direction[direction].add(provider)
    if not providers_by_direction:
        return None, 0, []
    direction = max(
        providers_by_direction,
        key=lambda candidate: (len(providers_by_direction[candidate]), len(by_direction[candidate])),
    )
    provider_count = len(providers_by_direction[direction])
    return direction, provider_count, _unique_entities(by_direction[direction])


def _crowded_headline_entities(evidence_entities: list[Entity], items_by_id: dict[str, SourceItem]) -> list[Entity]:
    duplicate_clusters = Counter()
    clustered_entities: list[Entity] = []
    for entity in evidence_entities:
        cluster = _duplicate_cluster_for_entity(entity, items_by_id)
        if not cluster:
            continue
        duplicate_clusters[cluster] += 1
        clustered_entities.append(entity)
    crowded_clusters = {cluster for cluster, count in duplicate_clusters.items() if count >= 2}
    if not crowded_clusters:
        return []
    return _unique_entities(
        [
            entity
            for entity in clustered_entities
            if _duplicate_cluster_for_entity(entity, items_by_id) in crowded_clusters
        ]
    )


def _duplicate_cluster_for_entity(entity: Entity, items_by_id: dict[str, SourceItem]) -> str:
    cluster = str(entity.properties.get("duplicate_cluster") or "").strip()
    if cluster:
        return cluster
    document_id = str(entity.properties.get("document_id") or "").strip()
    item = items_by_id.get(document_id)
    return str(item.duplicate_cluster if item else "").strip()


def _contradiction_clusters(evidence_entities: list[Entity]) -> list[list[str]]:
    by_cluster: dict[str, list[str]] = defaultdict(list)
    directions_by_cluster: dict[str, set[str]] = defaultdict(set)
    for entity in evidence_entities:
        cluster = str(entity.properties.get("duplicate_cluster") or "").strip()
        if not cluster:
            continue
        direction = _normalize_direction(entity.properties.get("direction"))
        by_cluster[cluster].append(entity.entity_id)
        directions_by_cluster[cluster].add(direction)
    return [
        sorted(entity_ids)
        for cluster, entity_ids in by_cluster.items()
        if {"bullish", "bearish"} & directions_by_cluster[cluster] and len(directions_by_cluster[cluster]) >= 2
    ]


def _market_confirmation_strength(base_features: list[SignalFeature], equity_entities: list[Entity]) -> float:
    strengths = [
        feature.strength
        for feature in base_features
        if feature.category == "market" and feature.direction == "bullish"
    ]
    if equity_entities:
        strengths.append(min(1.0, len(equity_entities) * 0.15))
    return max(strengths) if strengths else 0.0


def _equity_snapshot_is_soft(snapshot: dict[str, Any]) -> bool:
    series = snapshot.get("series") or {}
    declines = 0
    for instrument in EQUITY_INSTRUMENTS:
        if float((series.get(instrument) or {}).get("pct_change") or 0.0) < 0:
            declines += 1
    return declines >= 2


def _growth_snapshot_is_soft(snapshot: dict[str, Any]) -> bool:
    series = snapshot.get("series") or {}
    oil_change = float((series.get("WTI CRUDE OIL") or {}).get("pct_change") or 0.0)
    usd_change = float((series.get("DXY") or {}).get("pct_change") or 0.0)
    return oil_change > 0 and usd_change > 0


def _count_directional_conflicts(entities: list[Entity]) -> int:
    directions = Counter(_normalize_direction(entity.properties.get("direction")) for entity in entities)
    return min(directions.get("bullish", 0), directions.get("bearish", 0))


def _unique_entities(entities: list[Entity]) -> list[Entity]:
    seen: set[str] = set()
    unique: list[Entity] = []
    for entity in entities:
        if entity.entity_id in seen:
            continue
        seen.add(entity.entity_id)
        unique.append(entity)
    return unique


def _normalize_direction(value: Any) -> str:
    direction = str(value or "").strip().lower()
    if direction in {"bullish", "positive", "up"}:
        return "bullish"
    if direction in {"bearish", "negative", "down"}:
        return "bearish"
    return "neutral"
