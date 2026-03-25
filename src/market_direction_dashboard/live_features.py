from __future__ import annotations

from collections import Counter

from .core.domain import KnowledgeGraph
from .models import SignalFeature, SourceItem
from .retrieval import build_retrieval_assisted_features


def extract_signal_features(
    items: list[SourceItem],
    snapshot: dict,
    graph: KnowledgeGraph | None = None,
) -> list[SignalFeature]:
    features: list[SignalFeature] = []
    series = snapshot.get("series", {})

    def add_feature(
        name: str,
        direction: str,
        strength: float,
        evidence_ids: list[str],
        category: str,
        summary: str,
        conflict_count: int = 0,
        time_decay_weight: float = 1.0,
        provenance: dict | None = None,
    ) -> None:
        if strength <= 0:
            return
        features.append(
            SignalFeature(
                name=name,
                direction=direction,
                strength=round(min(1.0, max(0.0, strength)), 3),
                supporting_evidence_ids=evidence_ids,
                conflict_count=conflict_count,
                time_decay_weight=round(min(1.0, max(0.0, time_decay_weight)), 3),
                category=category,
                summary=summary,
                provenance=dict(provenance or {"feature_pass": "pass_1", "source": "raw_items_snapshot"}),
            )
        )

    for label, name in (
        ("US 10 YR TREASURY", "yield_rising"),
        ("VIX", "vix_spike"),
        ("DXY", "dxy_strength"),
        ("WTI CRUDE OIL", "oil_inflationary_pressure"),
    ):
        values = series.get(label)
        if not values:
            continue
        pct_change = float(values.get("pct_change", 0.0))
        evidence_ids = [item.id for item in items if item.instrument == label]
        if label in {"US 10 YR TREASURY", "VIX", "DXY", "WTI CRUDE OIL"}:
            direction = "bearish" if pct_change > 0 else "bullish"
        else:
            direction = "bullish" if pct_change > 0 else "bearish"
        add_feature(
            name=name,
            direction=direction,
            strength=min(1.0, abs(pct_change) / 2.5),
            evidence_ids=evidence_ids,
            category="market" if label != "US 10 YR TREASURY" else "economic",
            summary=f"{label} moved {pct_change:+.2f}% in the latest market snapshot.",
        )

    risk_on_labels = ("S&P 500", "NASDAQ 100", "DOW JONES", "RUSSELL 2000")
    positive_indices = [label for label in risk_on_labels if float(series.get(label, {}).get("pct_change", 0.0)) > 0]
    negative_indices = [label for label in risk_on_labels if float(series.get(label, {}).get("pct_change", 0.0)) < 0]
    if positive_indices or negative_indices:
        direction = "bullish" if len(positive_indices) >= len(negative_indices) else "bearish"
        strength = abs(len(positive_indices) - len(negative_indices)) / max(1, len(risk_on_labels))
        add_feature(
            name="breadth_risk_on" if direction == "bullish" else "breadth_risk_off",
            direction=direction,
            strength=max(0.25, strength),
            evidence_ids=[item.id for item in items if item.instrument in risk_on_labels],
            category="market",
            summary=f"Index breadth favored {len(positive_indices)} advancing vs {len(negative_indices)} declining benchmark instruments.",
            conflict_count=min(len(positive_indices), len(negative_indices)),
        )

    if positive_indices and negative_indices:
        add_feature(
            name="cross_asset_conflict",
            direction="bearish",
            strength=min(1.0, min(len(positive_indices), len(negative_indices)) / len(risk_on_labels)),
            evidence_ids=[item.id for item in items if item.instrument in risk_on_labels],
            category="market",
            summary="Cross-asset benchmark indices disagreed on the direction of risk appetite.",
            conflict_count=min(len(positive_indices), len(negative_indices)),
        )

    item_text = " ".join(f"{item.title} {item.summary}".lower() for item in items)
    news_evidence = [item.id for item in items if item.source_type == "news"]
    if any(term in item_text for term in ("cooling inflation", "soft landing", "rate cut")):
        add_feature(
            name="growth_positive_macro_surprise",
            direction="bullish",
            strength=0.7,
            evidence_ids=news_evidence,
            category="economic",
            summary="News flow suggests softer inflation or easing-policy relief.",
        )
    if any(term in item_text for term in ("tariff", "sanction", "geopolitic", "war", "shutdown")):
        add_feature(
            name="policy_uncertainty_increase",
            direction="bearish",
            strength=0.75,
            evidence_ids=news_evidence,
            category="political",
            summary="Policy or geopolitical headlines increased uncertainty.",
        )
    if any(term in item_text for term in ("retail", "upbeat", "euphoric", "momentum")):
        add_feature(
            name="retail_euphoric_tone",
            direction="bullish",
            strength=0.45,
            evidence_ids=[item.id for item in items if item.category == "social"],
            category="social",
            summary="Retail or crowd-tone evidence skewed optimistic.",
        )
    if any(term in item_text for term in ("panic", "selloff", "slump", "concern")):
        add_feature(
            name="headline_risk_off_tone",
            direction="bearish",
            strength=0.45,
            evidence_ids=news_evidence,
            category="social",
            summary="Headlines carried a defensive or panic tone.",
        )

    directional_items = [item for item in items if item.direction in {"bullish", "bearish"}]
    bullish_count = sum(1 for item in directional_items if item.direction == "bullish")
    bearish_count = sum(1 for item in directional_items if item.direction == "bearish")
    if directional_items:
        alignment = abs(bullish_count - bearish_count) / len(directional_items)
        if alignment >= 0.35:
            add_feature(
                name="narrative_alignment",
                direction="bullish" if bullish_count > bearish_count else "bearish",
                strength=min(1.0, alignment),
                evidence_ids=[item.id for item in directional_items],
                category="market",
                summary="Market and news evidence showed a consistent directional narrative.",
            )
        else:
            add_feature(
                name="narrative_conflict",
                direction="bearish",
                strength=min(1.0, 1.0 - alignment),
                evidence_ids=[item.id for item in directional_items],
                category="market",
                summary="Evidence was directionally mixed and should weaken conviction.",
                conflict_count=min(bullish_count, bearish_count),
            )

    duplicate_clusters = [item.duplicate_cluster for item in items if item.duplicate_cluster]
    if duplicate_clusters:
        cluster_counts = Counter(duplicate_clusters)
        dominant_cluster_size = max(cluster_counts.values())
        if dominant_cluster_size >= 2:
            add_feature(
                name="narrative_crowding",
                direction="bearish",
                strength=min(1.0, dominant_cluster_size / max(len(items), 1)),
                evidence_ids=[item.id for item in items if item.duplicate_cluster in cluster_counts],
                category="social",
            summary="A single narrative cluster dominated the evidence set, increasing crowding risk.",
        )

    if graph is not None:
        features.extend(
            build_retrieval_assisted_features(
                graph=graph,
                items=items,
                snapshot=snapshot,
                base_features=list(features),
            )
        )

    return features
