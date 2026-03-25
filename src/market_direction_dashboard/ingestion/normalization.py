from __future__ import annotations

from typing import Any

from ..models import SourceItem


SOURCE_CLASS_BREAKING_NEWS = "BreakingNews"
SOURCE_CLASS_FINANCIAL_REPORT = "FinancialReport"
SOURCE_CLASS_POLICY_DRAFT = "PolicyDraft"
SOURCE_CLASS_RESEARCH_DOCUMENT = "ResearchDocument"
SOURCE_CLASS_SOCIAL_DISCUSSION = "SocialDiscussion"
SOURCE_CLASS_MARKET_SNAPSHOT = "MarketSnapshot"

SOURCE_CLASS_TYPES = {
    SOURCE_CLASS_BREAKING_NEWS,
    SOURCE_CLASS_FINANCIAL_REPORT,
    SOURCE_CLASS_POLICY_DRAFT,
    SOURCE_CLASS_RESEARCH_DOCUMENT,
    SOURCE_CLASS_SOCIAL_DISCUSSION,
    SOURCE_CLASS_MARKET_SNAPSHOT,
}


def classify_source_item(item: dict[str, Any]) -> str:
    source = str(item.get("source") or "").lower()
    title = str(item.get("title") or "").lower()
    category = str(item.get("category") or "").lower()
    source_type = str(item.get("source_type") or "").lower()

    if source_type == "market" or category == "market":
        return SOURCE_CLASS_MARKET_SNAPSHOT if "latest move" in title or source.startswith("market-data/") else SOURCE_CLASS_FINANCIAL_REPORT
    if category == "political":
        return SOURCE_CLASS_POLICY_DRAFT
    if category == "social" or "sentiment" in title or "reddit" in source or "x.com" in source:
        return SOURCE_CLASS_SOCIAL_DISCUSSION
    if "research" in source or "research" in title or "survey" in title or "paper" in title or "fed" in source:
        return SOURCE_CLASS_RESEARCH_DOCUMENT
    if "earnings" in title or "report" in title or "filing" in title or "10-k" in title or "10-q" in title:
        return SOURCE_CLASS_FINANCIAL_REPORT
    return SOURCE_CLASS_BREAKING_NEWS


def normalize_source_item(item: Any) -> SourceItem:
    item_dict = item.to_dict() if hasattr(item, "to_dict") else dict(item)
    document_id = str(item_dict.get("id") or item_dict.get("document_id") or item_dict.get("title") or "document")
    source_class = classify_source_item(item_dict)
    return SourceItem(
        id=document_id,
        title=str(item_dict.get("title") or document_id),
        source=str(item_dict.get("source") or "unknown"),
        source_type=source_class,
        category=str(item_dict.get("category") or "news"),
        published_at=str(item_dict.get("published_at") or item_dict.get("fetched_at") or ""),
        url=str(item_dict.get("url") or ""),
        summary=str(item_dict.get("summary") or ""),
        raw_text=str(item_dict.get("raw_text") or ""),
        impact=str(item_dict.get("impact") or "neutral"),
        impact_score=float(item_dict.get("impact_score") or 0.0),
        instrument=str(item_dict.get("instrument") or ""),
        region=str(item_dict.get("region") or "US"),
        direction=str(item_dict.get("direction") or "neutral"),
        confidence_hint=float(item_dict.get("confidence_hint") or 0.5),
        freshness_score=float(item_dict.get("freshness_score") or 0.5),
        credibility_score=float(item_dict.get("credibility_score") or 0.5),
        proxy_used=bool(item_dict.get("proxy_used") or False),
        quality_score=float(item_dict.get("quality_score") or 0.5),
        base_quality_score=float(item_dict.get("base_quality_score") or item_dict.get("quality_score") or 0.5),
        evidence_kind=str(item_dict.get("evidence_kind") or "direct"),
        duplicate_cluster=str(item_dict.get("duplicate_cluster") or ""),
        graph_quality_adjustment=float(item_dict.get("graph_quality_adjustment") or 0.0),
        graph_quality_reasons=list(item_dict.get("graph_quality_reasons") or []),
        data_quality_flags=list(item_dict.get("data_quality_flags") or []),
    )


def normalize_source_items(items: list[Any]) -> list[SourceItem]:
    return [normalize_source_item(item) for item in items]
