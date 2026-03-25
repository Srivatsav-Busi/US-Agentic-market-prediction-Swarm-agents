from __future__ import annotations

import json
from typing import Any

from ..core.domain import Document, Entity, KnowledgeGraph, Relationship


def _relationship_property_map(relationship: Relationship) -> dict[str, Any]:
    return {
        "relationship_id": relationship.relationship_id,
        "relationship_type": relationship.relationship_type,
        "weight": relationship.weight,
        "evidence_document_ids": relationship.evidence_document_ids,
        **relationship.properties,
        **relationship.metadata,
    }


def _neo4j_property_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        if all(item is None or isinstance(item, (str, int, float, bool)) for item in value):
            return value
        return json.dumps(value, sort_keys=True)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _sanitize_neo4j_properties(properties: dict[str, Any]) -> dict[str, Any]:
    return {key: _neo4j_property_value(value) for key, value in properties.items() if value is not None}


class CypherQueryService:
    DOCUMENT_QUERY = (
        "MERGE (d:Document {document_id: $document_id}) "
        "SET d += $properties"
    )
    ENTITY_QUERY = (
        "MERGE (e:Entity {entity_id: $entity_id}) "
        "SET e += $properties "
        "SET e:`GraphNode` "
    )
    DOCUMENT_LINK_QUERY = (
        "MATCH (e:Entity {entity_id: $entity_id}) "
        "MATCH (d:Document {document_id: $document_id}) "
        "MERGE (e)-[r:MENTIONED_IN {relationship_id: $relationship_id}]->(d) "
        "SET r += $properties"
    )
    RELATIONSHIP_QUERY = (
        "MATCH (source:Entity {entity_id: $source_entity_id}) "
        "MATCH (target:Entity {entity_id: $target_entity_id}) "
        "MERGE (source)-[r:RELATED_TO {relationship_id: $relationship_id}]->(target) "
        "SET r += $properties"
    )
    PROJECT_NODE_QUERY = (
        "MERGE (n:GraphEntity {project_id: $project_id, node_id: $node_id}) "
        "SET n += $properties"
    )
    PROJECT_EDGE_QUERY = (
        "MATCH (source:GraphEntity {project_id: $project_id, node_id: $source_id}) "
        "MATCH (target:GraphEntity {project_id: $project_id, node_id: $target_id}) "
        "MERGE (source)-[r:RELATED {project_id: $project_id, edge_id: $edge_id}]->(target) "
        "SET r += $properties"
    )

    def upsert_document(self, tx: Any, document: Document) -> None:
        tx.run(
            self.DOCUMENT_QUERY,
            document_id=document.document_id,
            properties=_sanitize_neo4j_properties(
                {
                    "document_id": document.document_id,
                    "source_type": document.source_type,
                    "title": document.title,
                    "source": document.source,
                    "published_at": document.published_at,
                    "fetched_at": document.fetched_at,
                    "url": document.url,
                    "summary": document.summary,
                    "raw_text": document.raw_text,
                    **document.metadata,
                }
            ),
        )

    def upsert_entity(self, tx: Any, entity: Entity) -> None:
        tx.run(
            self.ENTITY_QUERY,
            entity_id=entity.entity_id,
            properties=_sanitize_neo4j_properties(
                {
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type,
                    "name": entity.name,
                    "canonical_name": entity.canonical_name,
                    "summary": entity.summary,
                    "source_document_ids": entity.source_document_ids,
                    **entity.properties,
                    **entity.metadata,
                }
            ),
        )
        tx.run(f"MATCH (e:Entity {{entity_id: $entity_id}}) SET e:`{entity.entity_type}`", entity_id=entity.entity_id)

    def upsert_document_links(self, tx: Any, entity: Entity) -> None:
        for document_id in entity.source_document_ids:
            tx.run(
                self.DOCUMENT_LINK_QUERY,
                entity_id=entity.entity_id,
                document_id=document_id,
                relationship_id=f"{entity.entity_id}:document:{document_id}",
                properties={"relationship_type": "MENTIONED_IN"},
            )

    def upsert_relationship(self, tx: Any, relationship: Relationship) -> None:
        tx.run(
            self.RELATIONSHIP_QUERY,
            source_entity_id=relationship.source_entity_id,
            target_entity_id=relationship.target_entity_id,
            relationship_id=relationship.relationship_id,
            properties=_sanitize_neo4j_properties(_relationship_property_map(relationship)),
        )

    def replace_projected_graph(self, tx: Any, project_id: str, graph_payload: dict[str, Any]) -> None:
        tx.run("MATCH (n:GraphEntity {project_id: $project_id}) DETACH DELETE n", project_id=project_id)
        for node in graph_payload.get("nodes", []):
            properties = _sanitize_neo4j_properties(
                {
                "project_id": project_id,
                "node_id": node["id"],
                "type": node["type"],
                "label": node["label"],
                "summary": node.get("summary"),
                **{key: value for key, value in node.get("properties", {}).items() if value is not None},
                }
            )
            tx.run(
                self.PROJECT_NODE_QUERY,
                project_id=project_id,
                node_id=node["id"],
                properties=properties,
            )
        for edge in graph_payload.get("edges", []):
            tx.run(
                self.PROJECT_EDGE_QUERY,
                project_id=project_id,
                source_id=edge["source"],
                target_id=edge["target"],
                edge_id=edge["id"],
                properties=_sanitize_neo4j_properties(
                    {
                        "project_id": project_id,
                        "edge_id": edge["id"],
                        "relation": edge["relation"],
                        "weight": edge.get("weight"),
                        **{key: value for key, value in edge.get("properties", {}).items() if value is not None},
                    }
                ),
            )

    def serialize_graph(self, graph: KnowledgeGraph) -> dict[str, Any]:
        return graph.to_dict()
