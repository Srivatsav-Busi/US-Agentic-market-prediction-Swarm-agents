from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..core.domain import Document, Entity, KnowledgeGraph, Relationship


def extract_document_entities(
    *,
    run_id: str,
    target: str,
    prediction_date: str,
    documents: list[Document],
    snapshot: dict[str, Any],
) -> tuple[list[Entity], list[Relationship], dict[str, int], set[str]]:
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    provider_counts: dict[str, int] = defaultdict(int)
    source_classes: set[str] = set()

    run_entity_id = f"run:{run_id}"
    entities.append(
        Entity(
            entity_id=run_entity_id,
            entity_type="PredictionRun",
            name=target,
            canonical_name=target,
            properties={
                "run_id": run_id,
                "prediction_date": prediction_date,
                "target": target,
            },
        )
    )

    for document in documents:
        source_classes.add(document.source_type)
        evidence_entity_id = f"evidence:{document.document_id}"
        entities.append(
            Entity(
                entity_id=evidence_entity_id,
                entity_type="Evidence",
                name=document.title,
                canonical_name=document.title,
                summary=document.summary,
                properties={
                    "document_id": document.document_id,
                    "source": document.source,
                    "source_type": document.source_type,
                    "source_class": document.source_type,
                    "published_at": document.published_at,
                    "url": document.url,
                    **document.metadata,
                },
                source_document_ids=[document.document_id],
                metadata={"published_at": document.published_at},
            )
        )
        relationships.append(
            Relationship(
                relationship_id=f"{run_entity_id}:evidence:{document.document_id}",
                relationship_type="USES_EVIDENCE",
                source_entity_id=run_entity_id,
                target_entity_id=evidence_entity_id,
                evidence_document_ids=[document.document_id],
            )
        )

        provider = str(document.source or "unknown")
        provider_counts[provider] += 1
        provider_entity_id = f"provider:{_slug(provider)}"
        entities.append(
            Entity(
                entity_id=provider_entity_id,
                entity_type="EvidenceSource",
                name=provider,
                canonical_name=provider,
                properties={"provider": provider},
            )
        )
        relationships.append(
            Relationship(
                relationship_id=f"{provider_entity_id}:evidence:{document.document_id}",
                relationship_type="PUBLISHED",
                source_entity_id=provider_entity_id,
                target_entity_id=evidence_entity_id,
                evidence_document_ids=[document.document_id],
            )
        )

        instrument = str(document.metadata.get("instrument") or "").strip()
        if instrument:
            instrument_entity_id = f"instrument:{instrument}"
            entities.append(
                Entity(
                    entity_id=instrument_entity_id,
                    entity_type="Instrument",
                    name=instrument,
                    canonical_name=instrument,
                    properties={"instrument": instrument},
                )
            )
            relationships.append(
                Relationship(
                    relationship_id=f"{evidence_entity_id}:instrument:{instrument}",
                    relationship_type="REFERENCES_INSTRUMENT",
                    source_entity_id=evidence_entity_id,
                    target_entity_id=instrument_entity_id,
                    evidence_document_ids=[document.document_id],
                )
            )

    for instrument_name, series in (snapshot.get("series") or {}).items():
        instrument_entity_id = f"instrument:{instrument_name}"
        entities.append(
            Entity(
                entity_id=instrument_entity_id,
                entity_type="Instrument",
                name=str(instrument_name),
                canonical_name=str(instrument_name),
                properties={
                    "instrument": instrument_name,
                    "latest_value": series.get("latest"),
                    "pct_change": series.get("pct_change"),
                    "provider": series.get("provider"),
                    "fetched_at": series.get("fetched_at"),
                },
            )
        )
        relationships.append(
            Relationship(
                relationship_id=f"{run_entity_id}:instrument:{instrument_name}",
                relationship_type="REFERENCES_INSTRUMENT",
                source_entity_id=run_entity_id,
                target_entity_id=instrument_entity_id,
            )
        )

    return entities, relationships, dict(provider_counts), source_classes


def extract_artifact_entities(
    *,
    graph: KnowledgeGraph,
    artifact: dict[str, Any],
) -> list[Entity]:
    entities = list(graph.entities)
    seen_entities = {entity.entity_id for entity in entities}
    entity_ids_from_documents = {f"evidence:{document.document_id}" for document in graph.documents}

    def add_entity(entity: Entity) -> None:
        if entity.entity_id in seen_entities:
            return
        seen_entities.add(entity.entity_id)
        entities.append(entity)

    run_id = str(artifact.get("run_id") or graph.metadata.get("run_id") or artifact.get("target") or "run")
    run_entity_id = f"run:{run_id}"
    if run_entity_id not in seen_entities:
        add_entity(
            Entity(
                entity_id=run_entity_id,
                entity_type="PredictionRun",
                name=str(artifact.get("target") or "Market Run"),
                canonical_name=str(artifact.get("target") or "Market Run"),
                summary=str(artifact.get("summary") or ""),
                properties={
                    "run_id": artifact.get("run_id"),
                    "prediction_date": artifact.get("prediction_date"),
                    "prediction_label": artifact.get("prediction_label"),
                    "run_health": artifact.get("run_health"),
                    "target": artifact.get("target"),
                },
            )
        )

    for feature in artifact.get("signal_features", []) or []:
        if not isinstance(feature, dict):
            continue
        feature_name = str(feature.get("name") or "")
        if not feature_name:
            continue
        add_entity(
            Entity(
                entity_id=f"feature:{feature_name}",
                entity_type="SignalFeature",
                name=feature_name,
                canonical_name=feature_name,
                summary=str(feature.get("summary") or ""),
                properties={
                    "direction": feature.get("direction"),
                    "strength": feature.get("strength"),
                    "category": feature.get("category"),
                    "provenance": dict(feature.get("provenance") or {}),
                },
            )
        )

    for report in artifact.get("source_agent_reports", []) or []:
        if not isinstance(report, dict):
            continue
        source_name = str(report.get("source") or "unknown-source")
        add_entity(
            Entity(
                entity_id=f"source-report:{source_name}",
                entity_type="SourceReport",
                name=source_name,
                canonical_name=source_name,
                summary=str(report.get("summary") or ""),
                properties={
                    "score": report.get("score"),
                    "source_confidence": report.get("source_confidence"),
                    "source_regime_fit": report.get("source_regime_fit"),
                },
            )
        )

    for category in ("economic", "political", "social", "market"):
        summary = artifact.get(f"{category}_report")
        if not summary:
            continue
        add_entity(
            Entity(
                entity_id=f"category-report:{category}",
                entity_type="CategoryReport",
                name=category.title(),
                canonical_name=category,
                summary=str(summary),
                properties={"category": category},
            )
        )

    for persona in artifact.get("swarm_agents", []) or []:
        if not isinstance(persona, dict):
            continue
        agent_id = str(persona.get("agent_id") or "")
        if not agent_id:
            continue
        add_entity(
            Entity(
                entity_id=f"persona:{agent_id}",
                entity_type="SwarmPersona",
                name=str(persona.get("name") or agent_id),
                canonical_name=agent_id,
                summary=str(persona.get("bio") or persona.get("persona") or ""),
                properties={
                    "archetype": persona.get("archetype"),
                    "stance_bias": persona.get("stance_bias"),
                    "influence_weight": persona.get("influence_weight"),
                },
            )
        )

    for round_item in artifact.get("swarm_rounds", []) or []:
        if not isinstance(round_item, dict):
            continue
        round_index = round_item.get("round_index", 0)
        for index, action in enumerate(round_item.get("actions", []) or []):
            if not isinstance(action, dict):
                continue
            action_id = f"{action.get('agent_id', 'agent')}:{round_index}:{index}"
            add_entity(
                Entity(
                    entity_id=f"action:{action_id}",
                    entity_type="SwarmAction",
                    name=str(action.get("action_type") or "action"),
                    canonical_name=action_id,
                    summary=str(action.get("content") or ""),
                    properties={
                        "action_id": action_id,
                        "action_type": action.get("action_type"),
                        "direction": action.get("direction"),
                        "strength": action.get("strength"),
                        "round_index": action.get("round_index"),
                    },
                )
            )

    for scenario_type, points in (artifact.get("market_projection") or {}).items():
        if scenario_type == "confidence_band" or not isinstance(points, list):
            continue
        for point in points[:30]:
            if not isinstance(point, dict):
                continue
            point_entity_id = f"projection:{scenario_type}:{point.get('forecast_date')}:{point.get('horizon_day')}"
            add_entity(
                Entity(
                    entity_id=point_entity_id,
                    entity_type="ProjectionPoint",
                    name=f"{scenario_type} {point.get('horizon_day')}",
                    canonical_name=point_entity_id,
                    properties={
                        "forecast_date": point.get("forecast_date"),
                        "horizon_day": point.get("horizon_day"),
                        "scenario_type": scenario_type,
                        "predicted_price": point.get("predicted_price"),
                        "predicted_return": point.get("predicted_return"),
                    },
                )
            )
            target_symbol = point.get("target_symbol")
            if target_symbol:
                add_entity(
                    Entity(
                        entity_id=f"instrument:{target_symbol}",
                        entity_type="Instrument",
                        name=str(target_symbol),
                        canonical_name=str(target_symbol),
                        properties={"instrument": target_symbol},
                    )
                )

    for item in artifact.get("sector_outlook", []) or []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("sector_symbol") or item.get("ticker") or "")
        if not symbol:
            continue
        add_entity(
            Entity(
                entity_id=f"sector:{symbol}",
                entity_type="SectorOutlook",
                name=str(item.get("sector_name") or symbol),
                canonical_name=symbol,
                summary=str(item.get("rationale") or ""),
                properties={
                    "sector_symbol": symbol,
                    "recommendation_label": item.get("recommendation_label") or item.get("prediction"),
                    "ranking_score": item.get("ranking_score"),
                    "expected_return_30d": item.get("expected_return_30d"),
                },
            )
        )

    return entities


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"
