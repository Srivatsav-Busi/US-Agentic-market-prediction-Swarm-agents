from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ReportAgentResult:
    outcomes: dict[str, Any]
    trends: dict[str, Any]
    insights: dict[str, Any]
    risks: dict[str, Any]
    follow_up_questions: list[str]
    simulation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_report_agent_result(
    *,
    artifact: dict[str, Any],
    outline: dict[str, Any],
    simulation_bundle: dict[str, Any] | None,
    graph_summary: dict[str, Any] | None,
) -> ReportAgentResult:
    top_drivers = artifact.get("top_drivers", [])[:5]
    signal_features = artifact.get("signal_features", [])[:6]
    confidence_notes = artifact.get("confidence_notes", [])[:4]
    warnings = artifact.get("warnings", [])[:4]
    analytics = dict((simulation_bundle or {}).get("analytics") or {})
    summary = dict(analytics.get("summary") or {})
    graph = dict(analytics.get("graph") or {})
    social = dict(analytics.get("social_dynamics") or {})
    memory = dict(analytics.get("memory") or {})

    risk_sections = []
    for section in outline.get("sections", []):
        title = str(section.get("title") or "").strip()
        if "risk" in title.lower() or "challenge" in title.lower():
            risk_sections.append(title)

    follow_ups = [
        f"What changed between rounds when the simulation moved toward {summary.get('dominant_stance', 'the final stance')}?",
        "Which graph deltas are material enough to project back into Aura?",
        "What additional evidence would most reduce the current confidence gap?",
    ]

    return ReportAgentResult(
        outcomes={
            "prediction_label": artifact.get("prediction_label"),
            "confidence": artifact.get("confidence"),
            "run_health": artifact.get("run_health"),
            "target": artifact.get("target"),
        },
        trends={
            "top_drivers": [
                {
                    "name": driver.get("name"),
                    "direction": driver.get("direction"),
                    "value": driver.get("value"),
                    "summary": driver.get("summary"),
                }
                for driver in top_drivers
            ],
            "signal_features": [
                {
                    "name": feature.get("name"),
                    "direction": feature.get("direction"),
                    "strength": feature.get("strength"),
                    "category": feature.get("category"),
                }
                for feature in signal_features
            ],
        },
        insights={
            "summary": outline.get("summary"),
            "section_titles": [str(section.get("title") or "").strip() for section in outline.get("sections", []) if section.get("title")],
            "graph_available": bool(graph_summary),
            "simulation_available": simulation_bundle is not None,
            "graph_delta_count": graph.get("count", 0),
            "decision_trace_count": dict(analytics.get("decision_traces") or {}).get("total_count", 0),
        },
        risks={
            "confidence_notes": list(confidence_notes),
            "warnings": list(warnings),
            "risk_sections": risk_sections,
            "top_conflicts": list(social.get("top_conflicts") or [])[:3],
        },
        follow_up_questions=follow_ups,
        simulation={
            "summary": summary,
            "graph": graph,
            "memory": memory,
            "social_dynamics": social,
        },
    )
