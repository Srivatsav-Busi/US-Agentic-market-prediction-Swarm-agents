from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SchemaStatement:
    query: str


class GraphSchemaManager:
    def __init__(self) -> None:
        self._queries = [
            "CREATE CONSTRAINT knowledge_graph_id_unique IF NOT EXISTS FOR (g:KnowledgeGraph) REQUIRE g.graph_id IS UNIQUE",
            "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE",
            "CREATE CONSTRAINT graph_entity_unique IF NOT EXISTS FOR (n:GraphEntity) REQUIRE (n.project_id, n.node_id) IS UNIQUE",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX document_source_type_idx IF NOT EXISTS FOR (d:Document) ON (d.source_type)",
            "CREATE INDEX document_published_at_idx IF NOT EXISTS FOR (d:Document) ON (d.published_at)",
        ]

    @property
    def queries(self) -> list[str]:
        return list(self._queries)

    def statements(self) -> list[SchemaStatement]:
        return [SchemaStatement(query=query) for query in self._queries]

    def ensure(self, target: Any) -> list[dict[str, str]]:
        applied: list[dict[str, str]] = []
        if hasattr(target, "execute_write") and not hasattr(target, "run"):
            for statement in self.statements():
                target.execute_write(statement.query)
                applied.append({"query": statement.query})
            return applied

        for query in self._queries:
            target.run(query)
            applied.append({"query": query})
        return applied
