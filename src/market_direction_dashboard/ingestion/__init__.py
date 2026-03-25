from __future__ import annotations

from ..retrieval import GraphRetrievalPreview, InMemoryGraphRetriever, build_graph_retrieval_preview
from .normalization import classify_source_item, normalize_source_item, normalize_source_items
from .provenance import IngestionBatchMetadata, build_ingestion_batch_metadata


def build_graph_first_ingestion(*args, **kwargs):
    from .graph_first import build_graph_first_ingestion as _build_graph_first_ingestion

    return _build_graph_first_ingestion(*args, **kwargs)


def enrich_knowledge_graph(*args, **kwargs):
    from .graph_first import enrich_knowledge_graph as _enrich_knowledge_graph

    return _enrich_knowledge_graph(*args, **kwargs)


def __getattr__(name: str):
    if name == "GraphFirstIngestionResult":
        from .graph_first import GraphFirstIngestionResult

        return GraphFirstIngestionResult
    raise AttributeError(name)

__all__ = [
    "GraphFirstIngestionResult",
    "GraphRetrievalPreview",
    "IngestionBatchMetadata",
    "InMemoryGraphRetriever",
    "build_graph_first_ingestion",
    "build_graph_retrieval_preview",
    "build_ingestion_batch_metadata",
    "classify_source_item",
    "enrich_knowledge_graph",
    "normalize_source_item",
    "normalize_source_items",
]
