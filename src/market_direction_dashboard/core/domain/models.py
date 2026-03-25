from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Document:
    document_id: str
    source_type: str
    title: str
    source: str
    published_at: str = ""
    fetched_at: str = ""
    url: str = ""
    summary: str = ""
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Entity:
    entity_id: str
    entity_type: str
    name: str
    canonical_name: str = ""
    summary: str = ""
    properties: dict[str, Any] = field(default_factory=dict)
    source_document_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Relationship:
    relationship_id: str
    relationship_type: str
    source_entity_id: str
    target_entity_id: str
    weight: float = 1.0
    evidence_document_ids: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeGraph:
    graph_id: str
    documents: list[Document] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "documents": [document.to_dict() for document in self.documents],
            "entities": [entity.to_dict() for entity in self.entities],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "metadata": dict(self.metadata),
        }


@runtime_checkable
class GraphRetriever(Protocol):
    def neighborhood_lookup(self, *, entity_id: str, hops: int = 1) -> KnowledgeGraph: ...

    def path_lookup(self, *, source_entity_id: str, target_entity_id: str, max_hops: int = 3) -> list[Relationship]: ...

    def time_filtered_subgraph(self, *, start_at: str | None = None, end_at: str | None = None) -> KnowledgeGraph: ...

    def rank_for_prompt(self, *, query: str, limit: int = 10) -> list[Entity]: ...


@dataclass(frozen=True)
class Episode:
    episode_id: str
    agent_id: str
    occurred_at: str
    summary: str
    details: dict[str, Any] = field(default_factory=dict)
    graph_reference_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class AgentMemoryStore(Protocol):
    def get_recent(self, *, agent_id: str, limit: int = 20) -> list[Episode]: ...

    def append(self, *, agent_id: str, episode: Episode) -> None: ...


@runtime_checkable
class SharedMemoryStore(Protocol):
    def get_recent(self, *, community_id: str, limit: int = 20) -> list[Episode]: ...

    def append(self, *, community_id: str, episode: Episode) -> None: ...


@runtime_checkable
class EpisodeStore(Protocol):
    def list_episodes(self, *, agent_id: str | None = None, limit: int = 20) -> list[Episode]: ...

    def append_episode(self, episode: Episode) -> None: ...


@dataclass(frozen=True)
class SimulationState:
    state_id: str
    world_state: dict[str, Any]
    agent_state: dict[str, Any]
    memory_state: dict[str, Any]
    event_history: list[dict[str, Any]]
    graph: KnowledgeGraph
    graph_deltas: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "world_state": dict(self.world_state),
            "agent_state": dict(self.agent_state),
            "memory_state": dict(self.memory_state),
            "event_history": list(self.event_history),
            "graph": self.graph.to_dict(),
            "graph_deltas": list(self.graph_deltas),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ReportContext:
    context_id: str
    subject_type: str
    subject_id: str
    artifact: dict[str, Any]
    graph: KnowledgeGraph
    simulation_state: SimulationState | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "subject_type": self.subject_type,
            "subject_id": self.subject_id,
            "artifact": dict(self.artifact),
            "graph": self.graph.to_dict(),
            "simulation_state": self.simulation_state.to_dict() if self.simulation_state else None,
            "metadata": dict(self.metadata),
        }
