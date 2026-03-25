from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


STATE_SECTIONS = (
    "world_state",
    "agent_state",
    "memory_state",
    "event_history",
    "graph_deltas",
)


@dataclass(frozen=True)
class SimulationStateSummary:
    round_count: int
    event_count: int
    graph_delta_count: int
    decision_trace_count: int
    memory_episode_count: int
    dominant_stance: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_count": self.round_count,
            "event_count": self.event_count,
            "graph_delta_count": self.graph_delta_count,
            "decision_trace_count": self.decision_trace_count,
            "memory_episode_count": self.memory_episode_count,
            "dominant_stance": self.dominant_stance,
        }


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_state_section(simulation_state: dict[str, Any] | None, section: str) -> Any:
    state = simulation_state or {}
    if section in {"event_history", "graph_deltas"}:
        return list(state.get(section) or [])
    if section in {"world_state", "agent_state", "memory_state"}:
        return dict(state.get(section) or {})
    return state.get(section)


def normalize_graph_delta(delta: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(delta or {})
    created_action_nodes = list(payload.get("created_action_nodes") or [])
    if not created_action_nodes:
        node_count = int(payload.get("new_action_nodes", 0) or 0)
        round_index = int(payload.get("round_index", -1) or -1)
        created_action_nodes = [
            {
                "action_node_id": f"round-{round_index}-action-{index}",
                "round_index": round_index,
            }
            for index in range(node_count)
        ]

    referenced_features = sorted(
        {
            str(item).strip()
            for item in (payload.get("referenced_features") or payload.get("referenced_feature_names") or [])
            if str(item).strip()
        }
    )
    referenced_entities = sorted(
        {
            str(item).strip()
            for item in (payload.get("referenced_entities") or [])
            if str(item).strip()
        }
    )
    social_dynamics = dict(payload.get("social_dynamics") or {})
    consensus_shift = dict(payload.get("consensus_shift") or {})
    normalized = {
        **payload,
        "round_index": int(payload.get("round_index", 0) or 0),
        "timestamp": str(payload.get("timestamp") or iso_utc_now()),
        "created_action_nodes": created_action_nodes,
        "new_action_nodes": int(payload.get("new_action_nodes", len(created_action_nodes)) or len(created_action_nodes)),
        "referenced_entities": referenced_entities,
        "referenced_features": referenced_features,
        "referenced_feature_names": referenced_features,
        "social_dynamics": {
            "top_conflicts": list(social_dynamics.get("top_conflicts") or []),
            "top_influence_edges": list(social_dynamics.get("top_influence_edges") or []),
            "dominant_communities": list(social_dynamics.get("dominant_communities") or []),
        },
        "consensus_shift": {
            "consensus_score": float(consensus_shift.get("consensus_score", payload.get("consensus_score", 0.0)) or 0.0),
            "conflict_score": float(consensus_shift.get("conflict_score", payload.get("conflict_score", 0.0)) or 0.0),
            "consensus_delta": float(consensus_shift.get("consensus_delta", 0.0) or 0.0),
            "conflict_delta": float(consensus_shift.get("conflict_delta", 0.0) or 0.0),
        },
    }
    return normalized


def normalize_simulation_state(simulation_state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(simulation_state or {})
    memory_state = dict(state.get("memory_state") or state.get("memory") or {})
    agent_state = dict(state.get("agent_state") or {})
    decision_traces = {
        str(agent_id): list(traces or [])
        for agent_id, traces in dict(agent_state.get("decision_traces") or {}).items()
    }
    normalized = {
        **state,
        "world_state": dict(state.get("world_state") or {}),
        "agent_state": {
            **agent_state,
            "profiles": list(agent_state.get("profiles") or []),
            "active_agent_ids": list(agent_state.get("active_agent_ids") or []),
            "decision_traces": decision_traces,
            "social_dynamics": dict(agent_state.get("social_dynamics") or {}),
        },
        "memory_state": memory_state,
        "memory": memory_state,
        "event_history": list(state.get("event_history") or []),
        "graph_deltas": [normalize_graph_delta(item) for item in list(state.get("graph_deltas") or [])],
    }
    return normalized


def summarize_state_transitions(simulation_state: dict[str, Any] | None) -> SimulationStateSummary:
    state = normalize_simulation_state(simulation_state)
    decision_trace_count = sum(len(traces) for traces in ((state.get("agent_state") or {}).get("decision_traces") or {}).values())
    memory_episode_count = int((((state.get("memory_state") or {}).get("shared") or {}).get("episode_count", 0)) or 0)
    dominant_stance = str(
        ((state.get("world_state") or {}).get("dominant_stance"))
        or state.get("dominant_stance")
        or "mixed"
    )
    return SimulationStateSummary(
        round_count=int(state.get("round_count", 0) or 0),
        event_count=len(state.get("event_history") or []),
        graph_delta_count=len(state.get("graph_deltas") or []),
        decision_trace_count=decision_trace_count,
        memory_episode_count=memory_episode_count,
        dominant_stance=dominant_stance,
    )
