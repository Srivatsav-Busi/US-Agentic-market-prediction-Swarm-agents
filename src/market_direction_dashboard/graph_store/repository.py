from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.domain import KnowledgeGraph, knowledge_graph_from_serialized
from .cypher import CypherQueryService, _sanitize_neo4j_properties
from .neo4j_store import Neo4jGraphStore
from .schema import GraphSchemaManager


@dataclass(frozen=True)
class GraphWriteResult:
    graph_id: str
    document_count: int
    entity_count: int
    relationship_count: int


class Neo4jGraphRepository:
    def __init__(
        self,
        store: Neo4jGraphStore,
        *,
        schema_manager: GraphSchemaManager | None = None,
        query_service: CypherQueryService | None = None,
    ) -> None:
        self.store = store
        self.schema_manager = schema_manager or GraphSchemaManager()
        self.query_service = query_service or CypherQueryService()

    def ensure_schema(self) -> None:
        self.store.execute_write(self.schema_manager.ensure)

    def upsert_knowledge_graph(self, graph: KnowledgeGraph | dict[str, Any]) -> GraphWriteResult:
        typed_graph = knowledge_graph_from_serialized(graph) if isinstance(graph, dict) else graph

        def _write(tx: Any) -> GraphWriteResult:
            for document in typed_graph.documents:
                self.query_service.upsert_document(tx, document)
            for entity in typed_graph.entities:
                self.query_service.upsert_entity(tx, entity)
                self.query_service.upsert_document_links(tx, entity)
            for relationship in typed_graph.relationships:
                self.query_service.upsert_relationship(tx, relationship)
            tx.run(
                "MERGE (g:KnowledgeGraph {graph_id: $graph_id}) "
                "SET g += $properties",
                graph_id=typed_graph.graph_id,
                properties=_sanitize_neo4j_properties(
                    {
                        "graph_id": typed_graph.graph_id,
                        **typed_graph.metadata,
                        "document_count": len(typed_graph.documents),
                        "entity_count": len(typed_graph.entities),
                        "relationship_count": len(typed_graph.relationships),
                    }
                ),
            )
            return GraphWriteResult(
                graph_id=typed_graph.graph_id,
                document_count=len(typed_graph.documents),
                entity_count=len(typed_graph.entities),
                relationship_count=len(typed_graph.relationships),
            )

        return self.store.execute_write(_write)

    def replace_projected_graph(self, *, project_id: str, graph_payload: dict[str, Any]) -> dict[str, Any]:
        def _write(tx: Any) -> dict[str, Any]:
            self.query_service.replace_projected_graph(tx, project_id, graph_payload)
            return {"backend_graph_ref": project_id}

        return self.store.execute_write(_write)
