from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.domain import KnowledgeGraph
from ..extraction.pipeline import GraphIngestionAssembly, build_graph_first_knowledge_graph, enrich_graph_with_artifact
from ..extraction.retrieval_preview import InMemoryGraphRetriever


@dataclass(frozen=True)
class GraphFirstIngestionResult:
    graph: KnowledgeGraph
    stage_diagnostics: dict[str, Any]
    retrieval_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "stage_diagnostics": dict(self.stage_diagnostics),
            "retrieval_preview": dict(self.retrieval_preview),
        }


def build_graph_first_ingestion(
    *,
    items: list[Any],
    snapshot: dict[str, Any],
    target: str,
    prediction_date: str,
    run_id: str,
) -> GraphFirstIngestionResult:
    assembly: GraphIngestionAssembly = build_graph_first_knowledge_graph(
        items=items,
        snapshot=snapshot,
        target=target,
        prediction_date=prediction_date,
        run_id=run_id,
    )
    return GraphFirstIngestionResult(
        graph=assembly.graph,
        stage_diagnostics=assembly.stage_diagnostics,
        retrieval_preview=assembly.retrieval_preview,
    )


def enrich_knowledge_graph(graph: KnowledgeGraph, artifact: dict[str, Any]) -> KnowledgeGraph:
    return enrich_graph_with_artifact(graph, artifact)


__all__ = [
    "GraphFirstIngestionResult",
    "InMemoryGraphRetriever",
    "build_graph_first_ingestion",
    "enrich_knowledge_graph",
]
