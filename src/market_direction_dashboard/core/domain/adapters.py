from __future__ import annotations

from typing import Any

from .models import Document, Entity, KnowledgeGraph, Relationship, ReportContext, SimulationState


def document_from_source_item(item: dict[str, Any]) -> Document:
    document_id = str(item.get("id") or item.get("document_id") or item.get("title") or "document")
    return Document(
        document_id=document_id,
        source_type=str(item.get("source_type") or "news"),
        title=str(item.get("title") or document_id),
        source=str(item.get("source") or "unknown"),
        published_at=str(item.get("published_at") or ""),
        fetched_at=str(item.get("fetched_at") or ""),
        url=str(item.get("url") or ""),
        summary=str(item.get("summary") or ""),
        raw_text=str(item.get("raw_text") or ""),
        metadata={
            "category": item.get("category"),
            "direction": item.get("direction"),
            "impact_score": item.get("impact_score"),
            "credibility_score": item.get("credibility_score"),
            "freshness_score": item.get("freshness_score"),
            "instrument": item.get("instrument"),
            "region": item.get("region"),
            "quality_score": item.get("quality_score"),
            "evidence_kind": item.get("evidence_kind"),
            "duplicate_cluster": item.get("duplicate_cluster"),
        },
    )


def knowledge_graph_from_serialized(payload: dict[str, Any]) -> KnowledgeGraph:
    return KnowledgeGraph(
        graph_id=str(payload.get("graph_id") or "knowledge-graph"),
        documents=[Document(**document) for document in payload.get("documents", []) if isinstance(document, dict)],
        entities=[Entity(**entity) for entity in payload.get("entities", []) if isinstance(entity, dict)],
        relationships=[
            Relationship(**relationship)
            for relationship in payload.get("relationships", [])
            if isinstance(relationship, dict)
        ],
        metadata=dict(payload.get("metadata") or {}),
    )


def _append_entity(entities: list[Entity], seen_entities: set[str], entity: Entity) -> None:
    if entity.entity_id in seen_entities:
        return
    seen_entities.add(entity.entity_id)
    entities.append(entity)


def _append_relationship(
    relationships: list[Relationship],
    seen_relationships: set[str],
    relationship: Relationship,
) -> None:
    if relationship.relationship_id in seen_relationships:
        return
    seen_relationships.add(relationship.relationship_id)
    relationships.append(relationship)


def knowledge_graph_from_artifact(artifact: dict[str, Any]) -> KnowledgeGraph:
    existing = artifact.get("knowledge_graph")
    if isinstance(existing, dict) and existing.get("entities") is not None:
        return knowledge_graph_from_serialized(existing)

    documents = [document_from_source_item(item) for item in artifact.get("sources", []) if isinstance(item, dict)]
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    seen_entities: set[str] = set()
    seen_relationships: set[str] = set()

    run_id = str(artifact.get("run_id") or artifact.get("simulation_id") or artifact.get("target") or "run")
    run_entity_id = f"run:{run_id}"
    _append_entity(
        entities,
        seen_entities,
        Entity(
            entity_id=run_entity_id,
            entity_type="PredictionRun",
            name=str(artifact.get("target") or "Market Run"),
            canonical_name=str(artifact.get("target") or "Market Run"),
            summary=str(artifact.get("summary") or ""),
            properties={
                "run_id": artifact.get("run_id"),
                "simulation_id": artifact.get("simulation_id"),
                "prediction_date": artifact.get("prediction_date"),
                "prediction_label": artifact.get("prediction_label"),
                "run_health": artifact.get("run_health"),
            },
        ),
    )

    source_items = [item for item in artifact.get("sources", []) if isinstance(item, dict)]
    source_items_by_id = {
        str(item.get("id") or item.get("document_id") or item.get("title") or ""): item for item in source_items
    }
    for document in documents:
        document_item = source_items_by_id.get(document.document_id, {})
        evidence_entity_id = f"evidence:{document.document_id}"
        _append_entity(
            entities,
            seen_entities,
            Entity(
                entity_id=evidence_entity_id,
                entity_type="Evidence",
                name=document.title,
                canonical_name=document.title,
                summary=document.summary,
                properties={
                    "source": document.source,
                    "source_type": document.source_type,
                    "published_at": document.published_at,
                    "url": document.url,
                    **document.metadata,
                },
                source_document_ids=[document.document_id],
                metadata={"published_at": document.published_at},
            ),
        )
        _append_relationship(
            relationships,
            seen_relationships,
            Relationship(
                relationship_id=f"{run_entity_id}:uses:{document.document_id}",
                relationship_type="USES_EVIDENCE",
                source_entity_id=run_entity_id,
                target_entity_id=evidence_entity_id,
                evidence_document_ids=[document.document_id],
            ),
        )
        instrument = document_item.get("instrument")
        if instrument:
            instrument_entity_id = f"instrument:{instrument}"
            _append_entity(
                entities,
                seen_entities,
                Entity(
                    entity_id=instrument_entity_id,
                    entity_type="Instrument",
                    name=str(instrument),
                    canonical_name=str(instrument),
                    properties={"instrument": instrument},
                ),
            )
            _append_relationship(
                relationships,
                seen_relationships,
                Relationship(
                    relationship_id=f"{evidence_entity_id}:instrument:{instrument}",
                    relationship_type="REFERENCES_INSTRUMENT",
                    source_entity_id=evidence_entity_id,
                    target_entity_id=instrument_entity_id,
                    evidence_document_ids=[document.document_id],
                ),
            )

    target = artifact.get("target")
    if target:
        target_entity_id = f"instrument:{target}"
        target_series = artifact.get("market_snapshot", {}).get("series", {}).get(target, {})
        _append_entity(
            entities,
            seen_entities,
            Entity(
                entity_id=target_entity_id,
                entity_type="Instrument",
                name=str(target),
                canonical_name=str(target),
                properties={
                    "instrument": target,
                    "latest_value": target_series.get("latest"),
                    "pct_change": target_series.get("pct_change"),
                },
            ),
        )
        _append_relationship(
            relationships,
            seen_relationships,
            Relationship(
                relationship_id=f"{run_entity_id}:target:{target}",
                relationship_type="REFERENCES_INSTRUMENT",
                source_entity_id=run_entity_id,
                target_entity_id=target_entity_id,
            ),
        )

    for feature in artifact.get("signal_features", []):
        if not isinstance(feature, dict):
            continue
        feature_name = str(feature.get("name") or "")
        if not feature_name:
            continue
        feature_entity_id = f"feature:{feature_name}"
        _append_entity(
            entities,
            seen_entities,
            Entity(
                entity_id=feature_entity_id,
                entity_type="SignalFeature",
                name=feature_name,
                canonical_name=feature_name,
                summary=str(feature.get("summary") or ""),
                properties={
                    "direction": feature.get("direction"),
                    "strength": feature.get("strength"),
                    "category": feature.get("category"),
                },
            ),
        )
        _append_relationship(
            relationships,
            seen_relationships,
            Relationship(
                relationship_id=f"{feature_entity_id}:run",
                relationship_type="INFLUENCES_RUN",
                source_entity_id=feature_entity_id,
                target_entity_id=run_entity_id,
                weight=float(feature.get("strength") or 1.0),
                evidence_document_ids=[str(evidence_id) for evidence_id in feature.get("supporting_evidence_ids", []) if evidence_id],
            ),
        )
        for evidence_id in feature.get("supporting_evidence_ids", []) or []:
            _append_relationship(
                relationships,
                seen_relationships,
                Relationship(
                    relationship_id=f"evidence:{evidence_id}:feature:{feature_name}",
                    relationship_type="GENERATED_FEATURE",
                    source_entity_id=f"evidence:{evidence_id}",
                    target_entity_id=feature_entity_id,
                    weight=float(feature.get("strength") or 1.0),
                    evidence_document_ids=[str(evidence_id)],
                ),
            )

    for report in artifact.get("source_agent_reports", []):
        if not isinstance(report, dict):
            continue
        source_name = str(report.get("source") or "unknown-source")
        report_entity_id = f"source-report:{source_name}"
        _append_entity(
            entities,
            seen_entities,
            Entity(
                entity_id=report_entity_id,
                entity_type="SourceReport",
                name=source_name,
                canonical_name=source_name,
                summary=str(report.get("summary") or ""),
                properties={
                    "score": report.get("score"),
                    "source_confidence": report.get("source_confidence"),
                    "source_regime_fit": report.get("source_regime_fit"),
                },
            ),
        )
        _append_relationship(
            relationships,
            seen_relationships,
            Relationship(
                relationship_id=f"{report_entity_id}:run",
                relationship_type="INFLUENCES_RUN",
                source_entity_id=report_entity_id,
                target_entity_id=run_entity_id,
                weight=abs(float(report.get("score") or 0.0)),
            ),
        )
        for evidence_id in report.get("evidence_ids_used", []) or []:
            _append_relationship(
                relationships,
                seen_relationships,
                Relationship(
                    relationship_id=f"{report_entity_id}:evidence:{evidence_id}",
                    relationship_type="USES_EVIDENCE",
                    source_entity_id=report_entity_id,
                    target_entity_id=f"evidence:{evidence_id}",
                    evidence_document_ids=[str(evidence_id)],
                ),
            )

    for category in ("economic", "political", "social", "market"):
        summary = artifact.get(f"{category}_report")
        if not summary:
            continue
        category_entity_id = f"category-report:{category}"
        _append_entity(
            entities,
            seen_entities,
            Entity(
                entity_id=category_entity_id,
                entity_type="CategoryReport",
                name=category.title(),
                canonical_name=category,
                summary=str(summary),
                properties={"category": category},
            ),
        )
        _append_relationship(
            relationships,
            seen_relationships,
            Relationship(
                relationship_id=f"{category_entity_id}:run",
                relationship_type="INFLUENCES_RUN",
                source_entity_id=category_entity_id,
                target_entity_id=run_entity_id,
            ),
        )

    for persona in artifact.get("swarm_agents", []):
        if not isinstance(persona, dict):
            continue
        agent_id = str(persona.get("agent_id") or "")
        if not agent_id:
            continue
        _append_entity(
            entities,
            seen_entities,
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
            ),
        )

    for round_item in artifact.get("swarm_rounds", []):
        if not isinstance(round_item, dict):
            continue
        round_index = round_item.get("round_index", 0)
        for index, action in enumerate(round_item.get("actions", []) or []):
            if not isinstance(action, dict):
                continue
            action_id = f"{action.get('agent_id', 'agent')}:{round_index}:{index}"
            action_entity_id = f"action:{action_id}"
            _append_entity(
                entities,
                seen_entities,
                Entity(
                    entity_id=action_entity_id,
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
                ),
            )
            agent_id = action.get("agent_id")
            if agent_id:
                _append_relationship(
                    relationships,
                    seen_relationships,
                    Relationship(
                        relationship_id=f"persona:{agent_id}:action:{action_id}",
                        relationship_type="AUTHORED",
                        source_entity_id=f"persona:{agent_id}",
                        target_entity_id=action_entity_id,
                    ),
                )
            for feature_name in action.get("referenced_feature_names", []) or []:
                _append_relationship(
                    relationships,
                    seen_relationships,
                    Relationship(
                        relationship_id=f"{action_entity_id}:feature:{feature_name}",
                        relationship_type="MENTIONS",
                        source_entity_id=action_entity_id,
                        target_entity_id=f"feature:{feature_name}",
                    ),
                )
            for evidence_id in action.get("referenced_evidence_ids", []) or []:
                _append_relationship(
                    relationships,
                    seen_relationships,
                    Relationship(
                        relationship_id=f"{action_entity_id}:evidence:{evidence_id}",
                        relationship_type="MENTIONS",
                        source_entity_id=action_entity_id,
                        target_entity_id=f"evidence:{evidence_id}",
                        evidence_document_ids=[str(evidence_id)],
                    ),
                )

    for scenario_type, points in (artifact.get("market_projection") or {}).items():
        if scenario_type == "confidence_band" or not isinstance(points, list):
            continue
        for point in points[:30]:
            if not isinstance(point, dict):
                continue
            point_entity_id = f"projection:{scenario_type}:{point.get('forecast_date')}:{point.get('horizon_day')}"
            _append_entity(
                entities,
                seen_entities,
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
                ),
            )
            _append_relationship(
                relationships,
                seen_relationships,
                Relationship(
                    relationship_id=f"{run_entity_id}:{point_entity_id}",
                    relationship_type="PROJECTS_TO",
                    source_entity_id=run_entity_id,
                    target_entity_id=point_entity_id,
                ),
            )

    for item in artifact.get("sector_outlook", []):
        if not isinstance(item, dict):
            continue
        symbol = str(item.get("sector_symbol") or item.get("ticker") or "")
        if not symbol:
            continue
        sector_entity_id = f"sector:{symbol}"
        _append_entity(
            entities,
            seen_entities,
            Entity(
                entity_id=sector_entity_id,
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
            ),
        )
        _append_relationship(
            relationships,
            seen_relationships,
            Relationship(
                relationship_id=f"{run_entity_id}:{sector_entity_id}",
                relationship_type="PROJECTS_TO",
                source_entity_id=run_entity_id,
                target_entity_id=sector_entity_id,
            ),
        )

    return KnowledgeGraph(
        graph_id=f"kg:{run_id}",
        documents=documents,
        entities=entities,
        relationships=relationships,
        metadata={
            "system_of_record": "neo4j_aura",
            "target": artifact.get("target"),
            "prediction_date": artifact.get("prediction_date"),
            "prediction_label": artifact.get("prediction_label"),
        },
    )


def simulation_state_from_environment(
    environment_payload: dict[str, Any],
    *,
    graph: KnowledgeGraph,
    snapshot: dict[str, Any] | None = None,
) -> SimulationState:
    profiles = environment_payload.get("profiles", []) if isinstance(environment_payload, dict) else []
    return SimulationState(
        state_id=str(environment_payload.get("environment_id") or environment_payload.get("simulation_id") or "simulation-state"),
        world_state={
            "target": environment_payload.get("target"),
            "prediction_date": environment_payload.get("prediction_date"),
            "snapshot": snapshot or {},
        },
        agent_state={
            "profiles": profiles,
            "activity_configs": environment_payload.get("activity_configs", []),
            "time_config": environment_payload.get("time_config", {}),
            "social_edges": environment_payload.get("social_edges", []),
        },
        memory_state={
            "seed_posts": environment_payload.get("seed_posts", []),
            "memory_snapshot": environment_payload.get("memory_snapshot", {}),
            "memory_version": "phase1_canonical_memory_state",
        },
        event_history=[
            {
                "event": "environment_initialized",
                "environment_id": environment_payload.get("environment_id"),
                "created_at": environment_payload.get("created_at"),
                "profile_count": len(profiles),
            }
        ],
        graph=graph,
        metadata={
            "mode": environment_payload.get("mode"),
            "base_run_stem": environment_payload.get("base_run_stem"),
            "environment_version": environment_payload.get("environment_version"),
        },
    )


def report_context_from_payload(
    *,
    subject_type: str,
    subject_id: str,
    artifact: dict[str, Any],
    graph_payload: dict[str, Any] | None = None,
    simulation_payload: dict[str, Any] | None = None,
) -> ReportContext:
    knowledge_graph = (
        knowledge_graph_from_serialized(graph_payload)
        if isinstance(graph_payload, dict) and graph_payload.get("entities") is not None
        else knowledge_graph_from_artifact(artifact)
    )
    simulation_state = None
    if isinstance(simulation_payload, dict) and simulation_payload:
        simulation_state = simulation_state_from_environment(simulation_payload, graph=knowledge_graph)
    return ReportContext(
        context_id=f"{subject_type}:{subject_id}",
        subject_type=subject_type,
        subject_id=subject_id,
        artifact=artifact,
        graph=knowledge_graph,
        simulation_state=simulation_state,
        metadata={
            "graph_entity_count": len(knowledge_graph.entities),
            "graph_relationship_count": len(knowledge_graph.relationships),
        },
    )


def old_pipeline_output_to_typed_state(payload: dict[str, Any]) -> ReportContext:
    subject_id = str(payload.get("simulation_id") or payload.get("run_id") or payload.get("target") or "artifact")
    subject_type = "simulation" if payload.get("simulation_id") else "prediction_run"
    return report_context_from_payload(
        subject_type=subject_type,
        subject_id=subject_id,
        artifact=payload,
        graph_payload=payload.get("knowledge_graph"),
        simulation_payload=payload.get("simulation_state") or payload.get("simulation_environment"),
    )
