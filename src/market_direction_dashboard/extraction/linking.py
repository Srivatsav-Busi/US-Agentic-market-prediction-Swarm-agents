from __future__ import annotations

from typing import Iterable

from ..core.domain import Entity, Relationship


def dedupe_knowledge_graph_components(
    entities: Iterable[Entity],
    relationships: Iterable[Relationship],
) -> tuple[list[Entity], list[Relationship]]:
    deduped_entities: list[Entity] = []
    deduped_relationships: list[Relationship] = []
    seen_entities: set[str] = set()
    seen_relationships: set[str] = set()

    for entity in entities:
        if entity.entity_id in seen_entities:
            continue
        seen_entities.add(entity.entity_id)
        deduped_entities.append(entity)

    for relationship in relationships:
        if relationship.relationship_id in seen_relationships:
            continue
        if relationship.source_entity_id not in seen_entities or relationship.target_entity_id not in seen_entities:
            continue
        seen_relationships.add(relationship.relationship_id)
        deduped_relationships.append(relationship)

    return deduped_entities, deduped_relationships
