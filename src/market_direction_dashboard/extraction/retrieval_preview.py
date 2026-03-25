from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from ..core.domain import Entity, KnowledgeGraph, Relationship


class InMemoryGraphRetriever:
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


def retrieval_preview(
    *,
    graph: KnowledgeGraph,
    target: str,
    prediction_date: str,
) -> dict[str, Any]:
    retriever = InMemoryGraphRetriever(graph)
    target_entity_id = f"instrument:{target}"
    neighborhood = retriever.neighborhood_lookup(entity_id=target_entity_id, hops=2)
    path = retriever.path_lookup(source_entity_id=f"run:{graph.metadata.get('run_id') or graph.graph_id.removeprefix('kg:')}", target_entity_id=target_entity_id, max_hops=2)
    time_slice = retriever.time_filtered_subgraph(start_at=f"{prediction_date}T00:00:00", end_at=f"{prediction_date}T23:59:59")
    ranked = retriever.rank_for_prompt(query=f"{target} market policy research sentiment", limit=5)
    return {
        "neighborhood_lookup": {
            "entity_id": target_entity_id,
            "entity_count": len(neighborhood.entities),
            "relationship_count": len(neighborhood.relationships),
        },
        "path_traversal": {
            "source_entity_id": f"run:{graph.metadata.get('run_id') or graph.graph_id.removeprefix('kg:')}",
            "target_entity_id": target_entity_id,
            "path_length": len(path),
        },
        "time_filtered_subgraph": {
            "entity_count": len(time_slice.entities),
            "relationship_count": len(time_slice.relationships),
        },
        "prompt_ranking": [entity.entity_id for entity in ranked],
    }


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
        documents=[document for document in graph.documents if f"evidence:{document.document_id}" in entity_ids],
        entities=[entity for entity in graph.entities if entity.entity_id in entity_ids],
        relationships=[relationship for relationship in graph.relationships if relationship.relationship_id in relationship_ids],
        metadata={**graph.metadata, "retrieval_scope": scope, "retrieval_params": params},
    )
