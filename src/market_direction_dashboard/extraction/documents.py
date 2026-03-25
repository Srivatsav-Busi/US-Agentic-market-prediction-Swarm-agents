from __future__ import annotations

from typing import Any

from ..core.domain import Document
from ..domain import document_from_source_item
from ..ingestion.normalization import (
    SOURCE_CLASS_BREAKING_NEWS,
    SOURCE_CLASS_FINANCIAL_REPORT,
    SOURCE_CLASS_MARKET_SNAPSHOT,
    SOURCE_CLASS_POLICY_DRAFT,
    SOURCE_CLASS_RESEARCH_DOCUMENT,
    SOURCE_CLASS_SOCIAL_DISCUSSION,
    SOURCE_CLASS_TYPES,
    classify_source_item,
    normalize_source_item,
    normalize_source_items,
)


def normalize_document(item: Any) -> Document:
    return document_from_source_item(item.to_dict() if hasattr(item, "to_dict") else item)

__all__ = [
    "SOURCE_CLASS_BREAKING_NEWS",
    "SOURCE_CLASS_FINANCIAL_REPORT",
    "SOURCE_CLASS_MARKET_SNAPSHOT",
    "SOURCE_CLASS_POLICY_DRAFT",
    "SOURCE_CLASS_RESEARCH_DOCUMENT",
    "SOURCE_CLASS_SOCIAL_DISCUSSION",
    "SOURCE_CLASS_TYPES",
    "classify_source_item",
    "normalize_source_item",
    "normalize_source_items",
    "normalize_document",
]
