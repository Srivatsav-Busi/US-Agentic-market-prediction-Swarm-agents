from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


GRAPH_DELTA_SCHEMA_VERSION = "graph_delta_summary:v1"
GRAPH_DELTA_COLUMNS = (
    "graph_delta__available",
    "graph_delta__evidence_node_change",
    "graph_delta__provider_concentration_change",
    "graph_delta__source_redundancy_change",
    "graph_delta__contradiction_density_change",
    "graph_delta__corroboration_change",
    "graph_delta__bullish_path_change",
    "graph_delta__bearish_path_change",
    "graph_delta__market_macro_connectivity_change",
    "graph_delta__freshness_weighted_evidence_mass_change",
    "graph_delta__market_intensity_change",
    "graph_delta__economic_intensity_change",
    "graph_delta__political_intensity_change",
    "graph_delta__social_intensity_change",
    "graph_delta__theme_acceleration",
    "graph_delta__delta_strength",
    "graph_delta__narrative_reversal_flag",
)


@dataclass(frozen=True)
class GraphDeltaSummary:
    prediction_date: str
    target: str
    schema_version: str
    previous_prediction_date: str | None
    features: dict[str, float]
    feature_groups: dict[str, str]
    theme_acceleration: float
    delta_strength: float
    narrative_reversal_flag: bool
    delta_available: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "previous_prediction_date": self.previous_prediction_date,
            "delta_available": self.delta_available,
            "theme_acceleration": self.theme_acceleration,
            "delta_strength": self.delta_strength,
            "narrative_reversal_flag": self.narrative_reversal_flag,
            "features": dict(self.features),
        }


def build_graph_delta_summary(
    *,
    prediction_date: str,
    target: str,
    current_features: dict[str, float],
    previous_summary: dict[str, Any] | None,
) -> GraphDeltaSummary:
    feature_groups = {name: "delta" for name in GRAPH_DELTA_COLUMNS}
    base_features = {name: 0.0 for name in GRAPH_DELTA_COLUMNS}
    if not previous_summary or not previous_summary.get("features"):
        return GraphDeltaSummary(
            prediction_date=prediction_date,
            target=target,
            schema_version=GRAPH_DELTA_SCHEMA_VERSION,
            previous_prediction_date=None,
            features=base_features,
            feature_groups=feature_groups,
            theme_acceleration=0.0,
            delta_strength=0.0,
            narrative_reversal_flag=False,
            delta_available=False,
        )

    previous_features = {str(key): float(value) for key, value in (previous_summary.get("features") or {}).items()}
    bullish_change = _delta(current_features, previous_features, "graph__bullish_path_strength")
    bearish_change = _delta(current_features, previous_features, "graph__bearish_path_strength")
    contradiction_change = _delta(current_features, previous_features, "graph__contradiction_density")
    corroboration_change = _delta(current_features, previous_features, "graph__disconnected_corroboration_score")
    market_intensity_change = _delta(current_features, previous_features, "graph__regime_cluster_intensity_market")
    economic_intensity_change = _delta(current_features, previous_features, "graph__regime_cluster_intensity_economic")
    political_intensity_change = _delta(current_features, previous_features, "graph__regime_cluster_intensity_political")
    social_intensity_change = _delta(current_features, previous_features, "graph__regime_cluster_intensity_social")
    evidence_node_change = _delta(current_features, previous_features, "graph__evidence_nodes_total")
    provider_change = _delta(current_features, previous_features, "graph__provider_concentration_index")
    redundancy_change = _delta(current_features, previous_features, "graph__source_redundancy_ratio")
    connectivity_change = _delta(current_features, previous_features, "graph__market_macro_connectivity")
    freshness_change = _delta(current_features, previous_features, "graph__freshness_weighted_evidence_mass")

    intensity_changes = (
        market_intensity_change,
        economic_intensity_change,
        political_intensity_change,
        social_intensity_change,
    )
    theme_acceleration = round(sum(abs(value) for value in intensity_changes) + abs(bullish_change - bearish_change) * 0.5, 6)
    delta_strength = round(
        sum(
            abs(value)
            for value in (
                evidence_node_change,
                provider_change,
                redundancy_change,
                contradiction_change,
                corroboration_change,
                bullish_change,
                bearish_change,
                connectivity_change,
                freshness_change,
                *intensity_changes,
            )
        )
        / 11.0,
        6,
    )
    prior_pressure = previous_features.get("graph__bullish_path_strength", 0.0) - previous_features.get("graph__bearish_path_strength", 0.0)
    current_pressure = current_features.get("graph__bullish_path_strength", 0.0) - current_features.get("graph__bearish_path_strength", 0.0)
    narrative_reversal_flag = bool(
        abs(current_pressure) > 0.05
        and abs(prior_pressure) > 0.05
        and ((prior_pressure > 0.0 > current_pressure) or (prior_pressure < 0.0 < current_pressure))
    )
    features = base_features | {
        "graph_delta__available": 1.0,
        "graph_delta__evidence_node_change": round(evidence_node_change, 6),
        "graph_delta__provider_concentration_change": round(provider_change, 6),
        "graph_delta__source_redundancy_change": round(redundancy_change, 6),
        "graph_delta__contradiction_density_change": round(contradiction_change, 6),
        "graph_delta__corroboration_change": round(corroboration_change, 6),
        "graph_delta__bullish_path_change": round(bullish_change, 6),
        "graph_delta__bearish_path_change": round(bearish_change, 6),
        "graph_delta__market_macro_connectivity_change": round(connectivity_change, 6),
        "graph_delta__freshness_weighted_evidence_mass_change": round(freshness_change, 6),
        "graph_delta__market_intensity_change": round(market_intensity_change, 6),
        "graph_delta__economic_intensity_change": round(economic_intensity_change, 6),
        "graph_delta__political_intensity_change": round(political_intensity_change, 6),
        "graph_delta__social_intensity_change": round(social_intensity_change, 6),
        "graph_delta__theme_acceleration": theme_acceleration,
        "graph_delta__delta_strength": delta_strength,
        "graph_delta__narrative_reversal_flag": 1.0 if narrative_reversal_flag else 0.0,
    }
    return GraphDeltaSummary(
        prediction_date=prediction_date,
        target=target,
        schema_version=GRAPH_DELTA_SCHEMA_VERSION,
        previous_prediction_date=str(previous_summary.get("prediction_date") or "") or None,
        features=features,
        feature_groups=feature_groups,
        theme_acceleration=theme_acceleration,
        delta_strength=delta_strength,
        narrative_reversal_flag=narrative_reversal_flag,
        delta_available=True,
    )


def _delta(current_features: dict[str, float], previous_features: dict[str, float], feature_name: str) -> float:
    return float(current_features.get(feature_name, 0.0)) - float(previous_features.get(feature_name, 0.0))
