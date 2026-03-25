from __future__ import annotations

from typing import Any

from ..core.domain import Document, Entity, KnowledgeGraph, Relationship


def extract_document_relationships(
    *,
    run_id: str,
    documents: list[Document],
    snapshot: dict[str, Any],
) -> list[Relationship]:
    relationships: list[Relationship] = []
    run_entity_id = f"run:{run_id}"

    for document in documents:
        relationships.append(
            Relationship(
                relationship_id=f"{run_entity_id}:evidence:{document.document_id}",
                relationship_type="USES_EVIDENCE",
                source_entity_id=run_entity_id,
                target_entity_id=f"evidence:{document.document_id}",
                evidence_document_ids=[document.document_id],
            )
        )
        relationships.append(
            Relationship(
                relationship_id=f"provider:{_slug(str(document.source or 'unknown'))}:evidence:{document.document_id}",
                relationship_type="PUBLISHED",
                source_entity_id=f"provider:{_slug(str(document.source or 'unknown'))}",
                target_entity_id=f"evidence:{document.document_id}",
                evidence_document_ids=[document.document_id],
            )
        )
        instrument = str(document.metadata.get("instrument") or "").strip()
        if instrument:
            relationships.append(
                Relationship(
                    relationship_id=f"evidence:{document.document_id}:instrument:{instrument}",
                    relationship_type="REFERENCES_INSTRUMENT",
                    source_entity_id=f"evidence:{document.document_id}",
                    target_entity_id=f"instrument:{instrument}",
                    evidence_document_ids=[document.document_id],
                )
            )

    for instrument_name in (snapshot.get("series") or {}).keys():
        relationships.append(
            Relationship(
                relationship_id=f"{run_entity_id}:instrument:{instrument_name}",
                relationship_type="REFERENCES_INSTRUMENT",
                source_entity_id=run_entity_id,
                target_entity_id=f"instrument:{instrument_name}",
            )
        )

    return relationships


def extract_artifact_relationships(
    *,
    graph: KnowledgeGraph,
    artifact: dict[str, Any],
) -> list[Relationship]:
    relationships: list[Relationship] = list(graph.relationships)
    seen_relationships = {relationship.relationship_id for relationship in relationships}
    seen_entities = {entity.entity_id for entity in graph.entities}
    entity_ids_from_documents = {f"evidence:{document.document_id}" for document in graph.documents}

    def add_relationship(relationship: Relationship) -> None:
        if relationship.relationship_id in seen_relationships:
            return
        if relationship.source_entity_id not in seen_entities or relationship.target_entity_id not in seen_entities:
            return
        seen_relationships.add(relationship.relationship_id)
        relationships.append(relationship)

    run_id = str(artifact.get("run_id") or graph.metadata.get("run_id") or artifact.get("target") or "run")
    run_entity_id = f"run:{run_id}"

    for feature in artifact.get("signal_features", []) or []:
        if not isinstance(feature, dict):
            continue
        feature_name = str(feature.get("name") or "")
        if not feature_name:
            continue
        add_relationship(
            Relationship(
                relationship_id=f"feature:{feature_name}:run",
                relationship_type="INFLUENCES_RUN",
                source_entity_id=f"feature:{feature_name}",
                target_entity_id=run_entity_id,
                weight=float(feature.get("strength") or 1.0),
                evidence_document_ids=[str(evidence_id) for evidence_id in feature.get("supporting_evidence_ids", []) if evidence_id],
                metadata={"provenance": dict(feature.get("provenance") or {})},
            )
        )
        for evidence_id in feature.get("supporting_evidence_ids", []) or []:
            evidence_entity_id = f"evidence:{evidence_id}"
            if evidence_entity_id in entity_ids_from_documents:
                add_relationship(
                    Relationship(
                        relationship_id=f"{evidence_entity_id}:feature:{feature_name}",
                        relationship_type="GENERATED_FEATURE",
                        source_entity_id=evidence_entity_id,
                        target_entity_id=f"feature:{feature_name}",
                        weight=float(feature.get("strength") or 1.0),
                        evidence_document_ids=[str(evidence_id)],
                        metadata={"provenance": dict(feature.get("provenance") or {})},
                    )
                )

    for report in artifact.get("source_agent_reports", []) or []:
        if not isinstance(report, dict):
            continue
        source_name = str(report.get("source") or "unknown-source")
        add_relationship(
            Relationship(
                relationship_id=f"source-report:{source_name}:run",
                relationship_type="INFLUENCES_RUN",
                source_entity_id=f"source-report:{source_name}",
                target_entity_id=run_entity_id,
                weight=abs(float(report.get("score") or 0.0)),
            )
        )
        for evidence_id in report.get("evidence_ids_used", []) or []:
            evidence_entity_id = f"evidence:{evidence_id}"
            if evidence_entity_id in entity_ids_from_documents:
                add_relationship(
                    Relationship(
                        relationship_id=f"source-report:{source_name}:evidence:{evidence_id}",
                        relationship_type="USES_EVIDENCE",
                        source_entity_id=f"source-report:{source_name}",
                        target_entity_id=evidence_entity_id,
                        evidence_document_ids=[str(evidence_id)],
                    )
                )

    for category in ("economic", "political", "social", "market"):
        if not artifact.get(f"{category}_report"):
            continue
        add_relationship(
            Relationship(
                relationship_id=f"category-report:{category}:run",
                relationship_type="INFLUENCES_RUN",
                source_entity_id=f"category-report:{category}",
                target_entity_id=run_entity_id,
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
            if action.get("agent_id"):
                add_relationship(
                    Relationship(
                        relationship_id=f"persona:{action['agent_id']}:action:{action_id}",
                        relationship_type="AUTHORED",
                        source_entity_id=f"persona:{action['agent_id']}",
                        target_entity_id=f"action:{action_id}",
                    )
                )
            for feature_name in action.get("referenced_feature_names", []) or []:
                add_relationship(
                    Relationship(
                        relationship_id=f"action:{action_id}:feature:{feature_name}",
                        relationship_type="MENTIONS",
                        source_entity_id=f"action:{action_id}",
                        target_entity_id=f"feature:{feature_name}",
                    )
                )
            for evidence_id in action.get("referenced_evidence_ids", []) or []:
                evidence_entity_id = f"evidence:{evidence_id}"
                if evidence_entity_id in entity_ids_from_documents:
                    add_relationship(
                        Relationship(
                            relationship_id=f"action:{action_id}:evidence:{evidence_id}",
                            relationship_type="MENTIONS",
                            source_entity_id=f"action:{action_id}",
                            target_entity_id=evidence_entity_id,
                            evidence_document_ids=[str(evidence_id)],
                        )
                    )

    for scenario_type, points in (artifact.get("market_projection") or {}).items():
        if scenario_type == "confidence_band" or not isinstance(points, list):
            continue
        for point in points[:30]:
            if not isinstance(point, dict):
                continue
            point_entity_id = f"projection:{scenario_type}:{point.get('forecast_date')}:{point.get('horizon_day')}"
            add_relationship(
                Relationship(
                    relationship_id=f"{run_entity_id}:{point_entity_id}",
                    relationship_type="PROJECTS_TO",
                    source_entity_id=run_entity_id,
                    target_entity_id=point_entity_id,
                )
            )
            target_symbol = point.get("target_symbol")
            if target_symbol:
                add_relationship(
                    Relationship(
                        relationship_id=f"{point_entity_id}:instrument:{target_symbol}",
                        relationship_type="REFERENCES_INSTRUMENT",
                        source_entity_id=point_entity_id,
                        target_entity_id=f"instrument:{target_symbol}",
                    )
                )

    for item in artifact.get("sector_outlook", []) or []:
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("sector_symbol") or item.get("ticker") or "")
        if not symbol:
            continue
        add_relationship(
            Relationship(
                relationship_id=f"{run_entity_id}:sector:{symbol}",
                relationship_type="PROJECTS_TO",
                source_entity_id=run_entity_id,
                target_entity_id=f"sector:{symbol}",
            )
        )

    return relationships


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "-" for char in value).strip("-") or "unknown"
