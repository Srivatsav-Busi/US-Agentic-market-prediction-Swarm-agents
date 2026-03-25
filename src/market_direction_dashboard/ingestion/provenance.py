from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IngestionBatchMetadata:
    batch_id: str
    run_id: str
    target: str
    prediction_date: str
    source_classes: list[str]
    source_counts: dict[str, int]
    item_count: int
    started_at: str
    completed_at: str = ""
    batch_version: str = "graph_first_ingestion:v1"
    source_system: str = "sources.py"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_ingestion_batch_metadata(
    *,
    batch_id: str,
    run_id: str,
    target: str,
    prediction_date: str,
    items: list[Any],
    source_classes: list[str],
    started_at: str,
    completed_at: str = "",
    metadata: dict[str, Any] | None = None,
) -> IngestionBatchMetadata:
    source_counts: dict[str, int] = {}
    for item in items:
        source_type = str(getattr(item, "source_type", "") or "").strip() or "unknown"
        source_counts[source_type] = source_counts.get(source_type, 0) + 1
    return IngestionBatchMetadata(
        batch_id=batch_id,
        run_id=run_id,
        target=target,
        prediction_date=prediction_date,
        source_classes=sorted(set(source_classes)),
        source_counts=source_counts,
        item_count=len(items),
        started_at=started_at,
        completed_at=completed_at,
        metadata=dict(metadata or {}),
    )
