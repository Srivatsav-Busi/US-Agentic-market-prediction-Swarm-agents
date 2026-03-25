from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

from ..core.domain import Document, Entity, GraphRetriever, KnowledgeGraph, Relationship
from ..graph_store import Neo4jGraphStore


@dataclass(frozen=True)
class GraphRetrievalPreview:
    neighborhood_lookup: dict[str, Any]
    path_traversal: dict[str, Any]
    time_filtered_subgraph: dict[str, Any]
    prompt_ranking: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entity_from_row(row: dict[str, Any]) -> Entity:
    return Entity(
        entity_id=str(row.get("entity_id") or ""),
        entity_type=str(row.get("entity_type") or "Entity"),
        name=str(row.get("name") or row.get("canonical_name") or row.get("entity_id") or ""),
        canonical_name=str(row.get("canonical_name") or row.get("name") or row.get("entity_id") or ""),
        summary=str(row.get("summary") or ""),
        properties=dict(row.get("properties") or {}),
        source_document_ids=list(row.get("source_document_ids") or []),
        metadata=dict(row.get("metadata") or {}),
    )


def _relationship_from_row(row: dict[str, Any]) -> Relationship:
    return Relationship(
        relationship_id=str(row.get("relationship_id") or ""),
        relationship_type=str(row.get("relationship_type") or "RELATED_TO"),
        source_entity_id=str(row.get("source_entity_id") or ""),
        target_entity_id=str(row.get("target_entity_id") or ""),
        weight=float(row.get("weight") or 1.0),
        evidence_document_ids=list(row.get("evidence_document_ids") or []),
        properties=dict(row.get("properties") or {}),
        metadata=dict(row.get("metadata") or {}),
    )


class InMemoryGraphRetriever(GraphRetriever):
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self.entities_by_id = {entity.entity_id: entity for entity in graph.entities}
        self.documents_by_id = {document.document_id: document for document in graph.documents}
        self.outbound: dict[str, list[Relationship]] = defaultdict(list)
        self.inbound: dict[str, list[Relationship]] = defaultdict(list)
        for relationship in graph.relationships:
            self.outbound[relationship.source_entity_id].append(relationship)
            self.inbound[relationship.target_entity_id].append(relationship)

    def neighborhood_lookup(self, *, entity_id: str, hops: int = 1) -> KnowledgeGraph:
        if entity_id not in self.entities_by_id:
            return KnowledgeGraph(graph_id=f"{self.graph.graph_id}:neighborhood:{entity_id}", metadata={"scope": "neighborhood"})

        visited = {entity_id}
        frontier = {entity_id}
        relationship_ids: set[str] = set()

        for _ in range(max(1, hops)):
            next_frontier: set[str] = set()
            for current in frontier:
                for relationship in self.outbound.get(current, []) + self.inbound.get(current, []):
                    relationship_ids.add(relationship.relationship_id)
                    next_frontier.add(relationship.source_entity_id)
                    next_frontier.add(relationship.target_entity_id)
            next_frontier -= visited
            visited.update(next_frontier)
            frontier = next_frontier
            if not frontier:
                break

        return _subgraph(
            graph=self.graph,
            entity_ids=visited,
            relationship_ids=relationship_ids,
            scope="neighborhood",
            params={"entity_id": entity_id, "hops": hops},
        )

    def path_lookup(self, *, source_entity_id: str, target_entity_id: str, max_hops: int = 3) -> list[Relationship]:
        if source_entity_id not in self.entities_by_id or target_entity_id not in self.entities_by_id:
            return []

        queue = deque([(source_entity_id, [])])
        seen = {source_entity_id}
        while queue:
            current, path = queue.popleft()
            if len(path) >= max_hops:
                continue
            for relationship in self.outbound.get(current, []):
                next_entity = relationship.target_entity_id
                if next_entity in seen:
                    continue
                next_path = path + [relationship]
                if next_entity == target_entity_id:
                    return next_path
                seen.add(next_entity)
                queue.append((next_entity, next_path))
        return []

    def time_filtered_subgraph(self, *, start_at: str | None = None, end_at: str | None = None) -> KnowledgeGraph:
        entity_ids: set[str] = set()
        relationship_ids: set[str] = set()
        for entity in self.graph.entities:
            published_at = str(entity.properties.get("published_at") or entity.metadata.get("published_at") or "")
            if not published_at:
                continue
            if start_at and published_at < start_at:
                continue
            if end_at and published_at > end_at:
                continue
            entity_ids.add(entity.entity_id)
        for entity_id in list(entity_ids):
            for relationship in self.outbound.get(entity_id, []) + self.inbound.get(entity_id, []):
                relationship_ids.add(relationship.relationship_id)
                entity_ids.add(relationship.source_entity_id)
                entity_ids.add(relationship.target_entity_id)
        return _subgraph(
            graph=self.graph,
            entity_ids=entity_ids,
            relationship_ids=relationship_ids,
            scope="time_filtered",
            params={"start_at": start_at, "end_at": end_at},
        )

    def rank_for_prompt(self, *, query: str, limit: int = 10) -> list[Entity]:
        terms = [term for term in query.lower().split() if term]
        scored: list[tuple[int, Entity]] = []
        for entity in self.graph.entities:
            haystack = " ".join(
                [
                    entity.name,
                    entity.canonical_name,
                    entity.summary,
                    " ".join(str(value) for value in entity.properties.values()),
                    " ".join(str(value) for value in entity.metadata.values()),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, entity))
        scored.sort(key=lambda item: (-item[0], item[1].entity_id))
        return [entity for _, entity in scored[:limit]]


class AuraGraphRetriever(GraphRetriever):
    def __init__(self, store: Neo4jGraphStore) -> None:
        self.store = store

    def neighborhood_lookup(self, *, entity_id: str, hops: int = 1) -> KnowledgeGraph:
        entity_rows = self.store.execute_read(
            """
            MATCH (root:Entity {entity_id: $entity_id})
            OPTIONAL MATCH (root)-[*0..$hops]-(related:Entity)
            RETURN collect(DISTINCT related.entity_id) AS entity_ids
            """,
            {"entity_id": entity_id, "hops": max(1, int(hops))},
        )
        entity_ids = _flatten_str_lists(entity_rows, "entity_ids", fallback=[entity_id])
        return self._load_subgraph(entity_ids=entity_ids, graph_id=f"kg:neighborhood:{entity_id}", scope="neighborhood", params={"entity_id": entity_id, "hops": hops})

    def path_lookup(self, *, source_entity_id: str, target_entity_id: str, max_hops: int = 3) -> list[Relationship]:
        relationship_rows = self.store.execute_read(
            """
            MATCH p = shortestPath((source:Entity {entity_id: $source_entity_id})-[*1..$max_hops]-(target:Entity {entity_id: $target_entity_id}))
            RETURN [r IN relationships(p) | r.relationship_id] AS relationship_ids
            """,
            {
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "max_hops": max(1, int(max_hops)),
            },
        )
        relationship_ids = _flatten_str_lists(relationship_rows, "relationship_ids")
        if not relationship_ids:
            return []
        rows = self.store.execute_read(
            """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE r.relationship_id IN $relationship_ids
            RETURN r.relationship_id AS relationship_id, type(r) AS relationship_type,
                   source.entity_id AS source_entity_id, target.entity_id AS target_entity_id,
                   r.weight AS weight, r.evidence_document_ids AS evidence_document_ids,
                   r.properties AS properties, r.metadata AS metadata
            """,
            {"relationship_ids": relationship_ids},
        )
        return [_relationship_from_row(row) for row in rows]

    def time_filtered_subgraph(self, *, start_at: str | None = None, end_at: str | None = None) -> KnowledgeGraph:
        document_rows = self.store.execute_read(
            """
            MATCH (d:Document)
            WHERE ($start_at IS NULL OR d.published_at >= $start_at)
              AND ($end_at IS NULL OR d.published_at <= $end_at)
            RETURN d.document_id AS document_id
            """,
            {"start_at": start_at, "end_at": end_at},
        )
        document_ids = _flatten_str_lists(document_rows, "document_id")
        entity_rows = self.store.execute_read(
            """
            MATCH (e:Entity)
            WHERE ($start_at IS NULL OR coalesce(e.published_at, e.created_at, e.updated_at) >= $start_at)
              AND ($end_at IS NULL OR coalesce(e.published_at, e.created_at, e.updated_at) <= $end_at)
            RETURN e.entity_id AS entity_id
            """,
            {"start_at": start_at, "end_at": end_at},
        )
        entity_ids = _flatten_str_lists(entity_rows, "entity_id")
        if document_ids:
            entity_rows = self.store.execute_read(
                """
                MATCH (e:Entity)
                WHERE any(document_id IN coalesce(e.source_document_ids, []) WHERE document_id IN $document_ids)
                RETURN e.entity_id AS entity_id
                """,
                {"document_ids": document_ids},
            )
            entity_ids.extend(_flatten_str_lists(entity_rows, "entity_id"))
        entity_ids = _unique_preserve_order(entity_ids)
        return self._load_subgraph(entity_ids=entity_ids, graph_id="kg:time_filtered", scope="time_filtered", params={"start_at": start_at, "end_at": end_at})

    def rank_for_prompt(self, *, query: str, limit: int = 10) -> list[Entity]:
        rows = self.store.execute_read(
            """
            MATCH (e:Entity)
            RETURN e.entity_id AS entity_id, e.entity_type AS entity_type, e.name AS name, e.canonical_name AS canonical_name,
                   e.summary AS summary, e.properties AS properties, e.source_document_ids AS source_document_ids, e.metadata AS metadata
            """,
        )
        terms = [term for term in query.lower().split() if term]
        scored: list[tuple[int, Entity]] = []
        for row in rows:
            entity = _entity_from_row(row)
            haystack = " ".join(
                [
                    entity.name,
                    entity.canonical_name,
                    entity.summary,
                    " ".join(str(value) for value in entity.properties.values()),
                    " ".join(str(value) for value in entity.metadata.values()),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score:
                scored.append((score, entity))
        scored.sort(key=lambda item: (-item[0], item[1].entity_id))
        return [entity for _, entity in scored[:limit]]

    def _load_subgraph(
        self,
        *,
        entity_ids: list[str],
        graph_id: str,
        scope: str,
        params: dict[str, Any],
    ) -> KnowledgeGraph:
        if not entity_ids:
            return KnowledgeGraph(graph_id=graph_id, metadata={**params, "retrieval_scope": scope})
        entity_rows = self.store.execute_read(
            """
            MATCH (e:Entity)
            WHERE e.entity_id IN $entity_ids
            RETURN e.entity_id AS entity_id, e.entity_type AS entity_type, e.name AS name, e.canonical_name AS canonical_name,
                   e.summary AS summary, e.properties AS properties, e.source_document_ids AS source_document_ids, e.metadata AS metadata
            """,
            {"entity_ids": entity_ids},
        )
        document_ids = _unique_preserve_order(
            [doc_id for row in entity_rows for doc_id in (row.get("source_document_ids") or []) if doc_id]
        )
        document_rows = self.store.execute_read(
            """
            MATCH (d:Document)
            WHERE d.document_id IN $document_ids
            RETURN d.document_id AS document_id, d.source_type AS source_type, d.title AS title, d.source AS source,
                   d.published_at AS published_at, d.fetched_at AS fetched_at, d.url AS url, d.summary AS summary,
                   d.raw_text AS raw_text, d.metadata AS metadata
            """,
            {"document_ids": document_ids},
        )
        relationship_rows = self.store.execute_read(
            """
            MATCH (source:Entity)-[r]->(target:Entity)
            WHERE source.entity_id IN $entity_ids AND target.entity_id IN $entity_ids
            RETURN r.relationship_id AS relationship_id, type(r) AS relationship_type,
                   source.entity_id AS source_entity_id, target.entity_id AS target_entity_id,
                   r.weight AS weight, r.evidence_document_ids AS evidence_document_ids,
                   r.properties AS properties, r.metadata AS metadata
            """,
            {"entity_ids": entity_ids},
        )
        return KnowledgeGraph(
            graph_id=graph_id,
            documents=[
                Document(
                    document_id=str(row.get("document_id") or ""),
                    source_type=str(row.get("source_type") or "news"),
                    title=str(row.get("title") or row.get("document_id") or ""),
                    source=str(row.get("source") or "unknown"),
                    published_at=str(row.get("published_at") or ""),
                    fetched_at=str(row.get("fetched_at") or ""),
                    url=str(row.get("url") or ""),
                    summary=str(row.get("summary") or ""),
                    raw_text=str(row.get("raw_text") or ""),
                    metadata=dict(row.get("metadata") or {}),
                )
                for row in document_rows
            ],
            entities=[_entity_from_row(row) for row in entity_rows],
            relationships=[_relationship_from_row(row) for row in relationship_rows],
            metadata={**params, "retrieval_scope": scope},
        )


def build_graph_retrieval_preview(
    *,
    graph: KnowledgeGraph,
    target: str,
    prediction_date: str,
    retriever: GraphRetriever | None = None,
) -> GraphRetrievalPreview:
    retriever = retriever or InMemoryGraphRetriever(graph)
    target_entity_id = f"instrument:{target}"
    neighborhood = retriever.neighborhood_lookup(entity_id=target_entity_id, hops=2)
    path = retriever.path_lookup(
        source_entity_id=f"run:{graph.metadata.get('run_id') or graph.graph_id.removeprefix('kg:')}",
        target_entity_id=target_entity_id,
        max_hops=2,
    )
    time_slice = retriever.time_filtered_subgraph(start_at=f"{prediction_date}T00:00:00", end_at=f"{prediction_date}T23:59:59")
    ranked = retriever.rank_for_prompt(query=f"{target} market policy research sentiment", limit=5)
    return GraphRetrievalPreview(
        neighborhood_lookup={
            "entity_id": target_entity_id,
            "entity_count": len(neighborhood.entities),
            "relationship_count": len(neighborhood.relationships),
        },
        path_traversal={
            "source_entity_id": f"run:{graph.metadata.get('run_id') or graph.graph_id.removeprefix('kg:')}",
            "target_entity_id": target_entity_id,
            "path_length": len(path),
        },
        time_filtered_subgraph={
            "entity_count": len(time_slice.entities),
            "relationship_count": len(time_slice.relationships),
        },
        prompt_ranking=[entity.entity_id for entity in ranked],
    )


def _flatten_str_lists(rows: list[dict[str, Any]], key: str, fallback: list[str] | None = None) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    if not values and fallback:
        return list(fallback)
    return _unique_preserve_order(values)


def _unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _subgraph(
    *,
    graph: KnowledgeGraph,
    entity_ids: set[str],
    relationship_ids: set[str],
    scope: str,
    params: dict[str, Any],
) -> KnowledgeGraph:
    return KnowledgeGraph(
        graph_id=f"{graph.graph_id}:{scope}",
        documents=[
            document
            for document in graph.documents
            if f"evidence:{document.document_id}" in entity_ids
        ],
        entities=[entity for entity in graph.entities if entity.entity_id in entity_ids],
        relationships=[relationship for relationship in graph.relationships if relationship.relationship_id in relationship_ids],
        metadata={**graph.metadata, "retrieval_scope": scope, "retrieval_params": params},
    )
