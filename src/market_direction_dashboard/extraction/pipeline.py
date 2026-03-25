from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.domain import KnowledgeGraph
from ..ingestion.normalization import normalize_source_items
from ..ingestion.provenance import build_ingestion_batch_metadata
from .documents import SOURCE_CLASS_TYPES, normalize_document
from .entities import extract_artifact_entities, extract_document_entities
from .linking import dedupe_knowledge_graph_components
from .relationships import extract_artifact_relationships, extract_document_relationships
from .retrieval_preview import retrieval_preview


@dataclass(frozen=True)
class GraphIngestionAssembly:
    graph: KnowledgeGraph
    stage_diagnostics: dict[str, Any]
    retrieval_preview: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "stage_diagnostics": dict(self.stage_diagnostics),
            "retrieval_preview": dict(self.retrieval_preview),
        }


def build_graph_first_knowledge_graph(
    *,
    items: list[Any],
    snapshot: dict[str, Any],
    target: str,
    prediction_date: str,
    run_id: str,
) -> GraphIngestionAssembly:
    documents = normalize_source_items(items)
    graph_documents = [normalize_document(item) for item in documents]
    document_entities, document_relationships, provider_counts, source_classes = extract_document_entities(
        run_id=run_id,
        target=target,
        prediction_date=prediction_date,
        documents=graph_documents,
        snapshot=snapshot,
    )
    document_relationships.extend(extract_document_relationships(run_id=run_id, documents=graph_documents, snapshot=snapshot))
    document_entities, document_relationships = dedupe_knowledge_graph_components(document_entities, document_relationships)
    batch_metadata = build_ingestion_batch_metadata(
        batch_id=f"batch:{run_id}",
        run_id=run_id,
        target=target,
        prediction_date=prediction_date,
        items=documents,
        source_classes=sorted(source_classes),
        started_at=f"{prediction_date}T08:30:00Z",
        metadata={
            "graph_mode": "graph_first_ingestion",
            "source_system": "sources.py",
        },
    )
    graph = KnowledgeGraph(
        graph_id=f"kg:{run_id}",
        documents=graph_documents,
        entities=document_entities,
        relationships=document_relationships,
        metadata={
            "system_of_record": "neo4j_aura",
            "graph_mode": "graph_first_ingestion",
            "run_id": run_id,
            "prediction_date": prediction_date,
            "target": target,
            "source_classes": sorted(source_classes),
            "ingestion_batch": batch_metadata.to_dict(),
            "pipeline_stage_status": {
                "source_connectors": "complete",
                "document_normalization": "complete",
                "entity_extraction": "complete",
                "relationship_extraction": "complete",
                "entity_resolution": "complete",
                "graph_persistence": "complete",
                "retrieval_indexing": "complete",
            },
        },
    )
    retrieval = retrieval_preview(graph=graph, target=target, prediction_date=prediction_date)
    return GraphIngestionAssembly(
        graph=graph,
        stage_diagnostics={
            "document_count": len(graph.documents),
            "entity_count": len(graph.entities),
            "relationship_count": len(graph.relationships),
            "provider_count": len(provider_counts),
            "source_classes": sorted(source_classes),
            "source_connectors": {"status": "complete", "documents_collected": len(graph.documents)},
            "document_normalization": {"status": "complete", "normalized_documents": len(graph.documents)},
            "entity_extraction": {"status": "complete", "entities_extracted": len(graph.entities)},
            "relationship_extraction": {"status": "complete", "relationships_extracted": len(graph.relationships)},
            "entity_resolution": {"status": "complete", "unique_providers": len(provider_counts)},
            "graph_persistence": {"status": "complete", "graph_id": graph.graph_id},
            "retrieval": retrieval,
        },
        retrieval_preview=retrieval,
    )


def enrich_graph_with_artifact(graph: KnowledgeGraph, artifact: dict[str, Any]) -> KnowledgeGraph:
    entities = extract_artifact_entities(graph=graph, artifact=artifact)
    working_graph = KnowledgeGraph(
        graph_id=graph.graph_id,
        documents=graph.documents,
        entities=entities,
        relationships=list(graph.relationships),
        metadata=graph.metadata,
    )
    relationships = extract_artifact_relationships(graph=working_graph, artifact=artifact)
    entities, relationships = dedupe_knowledge_graph_components(entities, relationships)
    metadata = dict(graph.metadata)
    metadata.update(
        {
            "prediction_label": artifact.get("prediction_label"),
            "run_health": artifact.get("run_health"),
            "run_id": artifact.get("run_id"),
        }
    )
    return KnowledgeGraph(
        graph_id=graph.graph_id,
        documents=graph.documents,
        entities=entities,
        relationships=relationships,
        metadata=metadata,
    )
