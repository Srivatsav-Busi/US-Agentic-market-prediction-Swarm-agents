from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.domain import AgentMemoryStore, Episode, EpisodeStore, SharedMemoryStore

DEFAULT_AGENT_ACTION_CAP = 3
DEFAULT_SHARED_ROUND_CAP = 4
DEFAULT_SHARED_REFERENCE_CAP = 24
DEFAULT_COMMUNITY_REFERENCE_CAP = 16
DEFAULT_SNAPSHOT_CAP = 10
DEFAULT_PROMPT_SHARED_ROUND_CAP = 3
DEFAULT_PROMPT_AGENT_ACTION_CAP = 2


def _episode_round(details: dict[str, Any]) -> int:
    value = details.get("round_index")
    if value is None:
        return -1
    return int(value)


def _score_episode(episode: Episode, latest_round: int | None = None) -> float:
    round_index = _episode_round(episode.details)
    if latest_round is None or round_index < 0:
        return 1.0
    age = max(0, latest_round - round_index)
    return round(1.0 / (1.0 + age), 4)


def _config_int(config: dict[str, Any] | None, key: str, default: int) -> int:
    if not config:
        return default
    return max(1, int(config.get(key, default) or default))


def _trim_unique_recent(values: list[str], cap: int) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in reversed(list(values or [])):
        value = str(raw).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
        if len(ordered) >= cap:
            break
    return list(reversed(ordered))


def _round_summary_payload(round_result: Any) -> dict[str, Any]:
    top_features = _trim_unique_recent(
        [
            feature_name
            for action in round_result.actions
            for feature_name in action.referenced_feature_names
        ],
        3,
    )
    dominant_stance = "mixed"
    bullish = int((round_result.stance_histogram or {}).get("bullish", 0) or 0)
    bearish = int((round_result.stance_histogram or {}).get("bearish", 0) or 0)
    if bullish > bearish:
        dominant_stance = "bullish"
    elif bearish > bullish:
        dominant_stance = "bearish"
    return {
        "round_index": int(round_result.round_index),
        "active_agent_count": len(round_result.active_agent_ids or []),
        "action_count": len(round_result.actions or []),
        "consensus_score": float(round_result.consensus_score),
        "conflict_score": float(round_result.conflict_score),
        "dominant_stance": dominant_stance,
        "top_features": top_features,
    }


def _action_memory_entry(action: Any, *, round_index: int) -> dict[str, Any]:
    feature_names = _trim_unique_recent(list(action.referenced_feature_names or []), 2)
    evidence_ids = _trim_unique_recent(list(action.referenced_evidence_ids or []), 2)
    summary_bits = [str(action.action_type), str(action.direction)]
    if feature_names:
        summary_bits.append(f"feature={','.join(feature_names)}")
    return {
        "round_index": int(round_index),
        "action_type": str(action.action_type),
        "direction": str(action.direction),
        "target_agent_id": action.target_agent_id,
        "feature_names": feature_names,
        "evidence_ids": evidence_ids,
        "summary": " ".join(summary_bits),
    }


def _build_shared_context(shared: dict[str, Any], *, round_cap: int) -> str:
    round_summaries = list(shared.get("round_summaries") or [])[-round_cap:]
    if not round_summaries:
        return ""
    parts = []
    for item in round_summaries:
        if not isinstance(item, dict):
            continue
        parts.append(
            "r{round_index}:{dominant_stance}:c{consensus_score:.2f}:x{conflict_score:.2f}:f={top_features}".format(
                round_index=int(item.get("round_index", 0) or 0),
                dominant_stance=str(item.get("dominant_stance") or "mixed"),
                consensus_score=float(item.get("consensus_score", 0.0) or 0.0),
                conflict_score=float(item.get("conflict_score", 0.0) or 0.0),
                top_features=",".join(item.get("top_features") or []) or "none",
            )
        )
    return " | ".join(parts)


def _build_agent_context(agent_entry: dict[str, Any], *, action_cap: int) -> str:
    recent_actions = list(agent_entry.get("recent_actions") or [])[-action_cap:]
    if not recent_actions:
        return ""
    parts = []
    for item in recent_actions:
        if not isinstance(item, dict):
            continue
        parts.append(
            "r{round_index}:{action_type}:{direction}:f={features}".format(
                round_index=int(item.get("round_index", 0) or 0),
                action_type=str(item.get("action_type") or "act"),
                direction=str(item.get("direction") or "neutral"),
                features=",".join(item.get("feature_names") or []) or "none",
            )
        )
    return " | ".join(parts)


@dataclass
class InMemoryEpisodeStore(EpisodeStore):
    episodes: list[Episode]

    def __init__(self, episodes: list[Episode] | None = None) -> None:
        self.episodes = list(episodes or [])

    def list_episodes(self, *, agent_id: str | None = None, limit: int = 20) -> list[Episode]:
        filtered = [episode for episode in self.episodes if agent_id is None or episode.agent_id == agent_id]
        return filtered[-limit:]

    def append_episode(self, episode: Episode) -> None:
        self.episodes.append(episode)


class InMemoryAgentMemoryStore(AgentMemoryStore):
    def __init__(self, episode_store: InMemoryEpisodeStore) -> None:
        self.episode_store = episode_store

    def get_recent(self, *, agent_id: str, limit: int = 20) -> list[Episode]:
        episodes = self.episode_store.list_episodes(agent_id=agent_id, limit=limit * 2)
        latest_round = max((_episode_round(item.details) for item in episodes), default=None)
        return sorted(episodes, key=lambda item: _score_episode(item, latest_round), reverse=True)[:limit]

    def append(self, *, agent_id: str, episode: Episode) -> None:
        if episode.agent_id != agent_id:
            episode = Episode(
                episode_id=episode.episode_id,
                agent_id=agent_id,
                occurred_at=episode.occurred_at,
                summary=episode.summary,
                details=dict(episode.details),
                graph_reference_ids=list(episode.graph_reference_ids),
            )
        self.episode_store.append_episode(episode)


class InMemorySharedMemoryStore(SharedMemoryStore):
    def __init__(self, episode_store: InMemoryEpisodeStore) -> None:
        self.episode_store = episode_store

    def get_recent(self, *, community_id: str, limit: int = 20) -> list[Episode]:
        filtered = [
            episode
            for episode in self.episode_store.list_episodes(limit=limit * 5)
            if str(episode.details.get("community_id") or "") == community_id
        ]
        latest_round = max((_episode_round(item.details) for item in filtered), default=None)
        return sorted(filtered, key=lambda item: _score_episode(item, latest_round), reverse=True)[:limit]

    def append(self, *, community_id: str, episode: Episode) -> None:
        details = dict(episode.details)
        details["community_id"] = community_id
        self.episode_store.append_episode(
            Episode(
                episode_id=episode.episode_id,
                agent_id=episode.agent_id,
                occurred_at=episode.occurred_at,
                summary=episode.summary,
                details=details,
                graph_reference_ids=list(episode.graph_reference_ids),
            )
        )


def bootstrap_memory_snapshot(*, profiles: list[Any], social_edges: list[Any], prior_memory: dict | None) -> dict[str, Any]:
    prior_memory = dict(prior_memory or {})
    prior_shared = dict(prior_memory.get("shared") or {})
    prior_individual = {entry.get("agent_id"): dict(entry) for entry in (prior_memory.get("individual") or []) if entry.get("agent_id")}
    communities = list(prior_memory.get("communities") or [])
    snapshots = list(prior_memory.get("snapshots") or [])
    config = dict(prior_memory.get("config") or {})
    agent_action_cap = _config_int(config, "swarm_memory_agent_action_cap", DEFAULT_AGENT_ACTION_CAP)
    shared_reference_cap = _config_int(config, "swarm_shared_reference_cap", DEFAULT_SHARED_REFERENCE_CAP)
    shared_round_cap = _config_int(config, "swarm_memory_shared_round_cap", DEFAULT_SHARED_ROUND_CAP)
    individual = []
    for profile in profiles:
        existing = prior_individual.get(profile.agent_id, {})
        individual.append(
            {
                "agent_id": profile.agent_id,
                "seed_evidence_ids": list(profile.seed_evidence_ids),
                "recent_actions": list(existing.get("recent_actions") or [])[-agent_action_cap:],
                "episode_count": int(existing.get("episode_count", 0) or 0),
                "graph_reference_ids": _trim_unique_recent(list(existing.get("graph_reference_ids") or []), shared_reference_cap),
                "recency_score": float(existing.get("recency_score", 0.0) or 0.0),
                "last_round_index": int(existing.get("last_round_index", -1) or -1),
                "last_action_type": str(existing.get("last_action_type") or ""),
                "last_direction": str(existing.get("last_direction") or "neutral"),
            }
        )
    return {
        "individual": individual,
        "shared": {
            "episode_count": int(prior_shared.get("episode_count", 0) or 0),
            "round_summaries": list(prior_shared.get("round_summaries") or [])[-shared_round_cap:],
            "shared_feature_names": _trim_unique_recent(list(prior_shared.get("shared_feature_names") or []), shared_reference_cap),
            "shared_evidence_ids": _trim_unique_recent(list(prior_shared.get("shared_evidence_ids") or []), shared_reference_cap),
            "snapshot_version": "phase4_memory:v1",
        },
        "communities": communities,
        "social_edges": [edge.to_dict() if hasattr(edge, "to_dict") else dict(edge) for edge in social_edges[:100]],
        "snapshots": snapshots[-DEFAULT_SNAPSHOT_CAP:],
        "config": {
            "swarm_memory_agent_action_cap": agent_action_cap,
            "swarm_memory_shared_round_cap": shared_round_cap,
            "swarm_shared_reference_cap": shared_reference_cap,
            "swarm_memory_community_reference_cap": _config_int(config, "swarm_memory_community_reference_cap", DEFAULT_COMMUNITY_REFERENCE_CAP),
            "swarm_prompt_shared_round_cap": _config_int(config, "swarm_prompt_shared_round_cap", DEFAULT_PROMPT_SHARED_ROUND_CAP),
            "swarm_prompt_agent_action_cap": _config_int(config, "swarm_prompt_agent_action_cap", DEFAULT_PROMPT_AGENT_ACTION_CAP),
        },
    }


def evolve_memory_snapshot(
    *,
    memory_state: dict[str, Any],
    round_result: Any,
    profiles: list[Any],
) -> dict[str, Any]:
    config = dict(memory_state.get("config") or {})
    agent_action_cap = _config_int(config, "swarm_memory_agent_action_cap", DEFAULT_AGENT_ACTION_CAP)
    shared_round_cap = _config_int(config, "swarm_memory_shared_round_cap", DEFAULT_SHARED_ROUND_CAP)
    shared_reference_cap = _config_int(config, "swarm_shared_reference_cap", DEFAULT_SHARED_REFERENCE_CAP)
    community_reference_cap = _config_int(config, "swarm_memory_community_reference_cap", DEFAULT_COMMUNITY_REFERENCE_CAP)
    shared_memory = dict(memory_state.get("shared") or {})
    shared_memory["episode_count"] = int(shared_memory.get("episode_count", 0) or 0) + 1
    shared_memory["round_summaries"] = (
        list(shared_memory.get("round_summaries", [])) + [_round_summary_payload(round_result)]
    )[-shared_round_cap:]
    shared_memory["shared_feature_names"] = _trim_unique_recent(
        list(shared_memory.get("shared_feature_names") or [])
        + [feature_name for action in round_result.actions for feature_name in action.referenced_feature_names],
        shared_reference_cap,
    )
    shared_memory["shared_evidence_ids"] = _trim_unique_recent(
        list(shared_memory.get("shared_evidence_ids") or [])
        + [evidence_id for action in round_result.actions for evidence_id in action.referenced_evidence_ids],
        shared_reference_cap,
    )
    indexed_individual = {
        entry.get("agent_id"): dict(entry)
        for entry in (memory_state.get("individual") or [])
        if entry.get("agent_id")
    }
    community_map: dict[str, dict[str, Any]] = {
        entry.get("community_id"): dict(entry)
        for entry in (memory_state.get("communities") or [])
        if entry.get("community_id")
    }
    profile_by_id = {profile.agent_id: profile for profile in profiles}

    for action in round_result.actions:
        entry = indexed_individual.setdefault(
            action.agent_id,
            {
                "agent_id": action.agent_id,
                "recent_actions": [],
                "seed_evidence_ids": [],
                "episode_count": 0,
                "graph_reference_ids": [],
                "recency_score": 0.0,
                "last_round_index": -1,
                "last_action_type": "",
                "last_direction": "neutral",
            },
        )
        entry["episode_count"] = int(entry.get("episode_count", 0) or 0) + 1
        entry["recent_actions"] = (
            list(entry.get("recent_actions", []))
            + [_action_memory_entry(action, round_index=round_result.round_index)]
        )[-agent_action_cap:]
        entry["graph_reference_ids"] = _trim_unique_recent(
            list(entry.get("graph_reference_ids") or [])
            + list(action.referenced_evidence_ids)
            + list(action.referenced_feature_names),
            shared_reference_cap,
        )
        entry["recency_score"] = round(1.0 / (1.0 + max(0, int(round_result.round_index))), 4)
        entry["last_round_index"] = int(round_result.round_index)
        entry["last_action_type"] = str(action.action_type)
        entry["last_direction"] = str(action.direction)
        profile = profile_by_id.get(action.agent_id)
        if profile is None:
            continue
        for category in profile.focus_categories:
            community = community_map.setdefault(
                category,
                {
                    "community_id": category,
                    "agent_ids": [],
                    "shared_feature_names": [],
                    "shared_evidence_ids": [],
                    "episode_count": 0,
                },
            )
            community["agent_ids"] = sorted({*community.get("agent_ids", []), action.agent_id})
            community["shared_feature_names"] = _trim_unique_recent(
                list(community.get("shared_feature_names") or []) + list(action.referenced_feature_names),
                community_reference_cap,
            )
            community["shared_evidence_ids"] = _trim_unique_recent(
                list(community.get("shared_evidence_ids") or []) + list(action.referenced_evidence_ids),
                community_reference_cap,
            )
            community["episode_count"] = int(community.get("episode_count", 0) or 0) + 1

    snapshots = list(memory_state.get("snapshots") or [])
    snapshots.append(
        {
            "round_index": round_result.round_index,
            "episode_count": shared_memory["episode_count"],
            "shared_feature_names": list(shared_memory["shared_feature_names"]),
            "shared_evidence_ids": list(shared_memory["shared_evidence_ids"]),
        }
    )

    return {
        **memory_state,
        "shared": shared_memory,
        "individual": sorted(indexed_individual.values(), key=lambda item: item["agent_id"]),
        "communities": sorted(community_map.values(), key=lambda item: item["community_id"]),
        "snapshots": snapshots[-DEFAULT_SNAPSHOT_CAP:],
        "config": config,
    }


def summarize_memory_diagnostics(*, environment: Any, rounds: list[Any]) -> dict[str, Any]:
    memory = environment.memory_snapshot
    shared = dict(memory.get("shared") or {})
    config = dict(memory.get("config") or {})
    agent_action_cap = _config_int(config, "swarm_memory_agent_action_cap", DEFAULT_AGENT_ACTION_CAP)
    shared_round_cap = _config_int(config, "swarm_memory_shared_round_cap", DEFAULT_SHARED_ROUND_CAP)
    shared_reference_cap = _config_int(config, "swarm_shared_reference_cap", DEFAULT_SHARED_REFERENCE_CAP)
    agent_state: dict[str, dict[str, Any]] = {}
    for profile in environment.profiles:
        agent_state[profile.agent_id] = {
            "agent_id": profile.agent_id,
            "name": profile.name,
            "stance_bias": profile.stance_bias,
            "seed_evidence_ids": list(profile.seed_evidence_ids),
            "recent_actions": [],
            "episode_count": 0,
            "graph_reference_ids": [],
            "recency_score": 0.0,
            "referenced_feature_names": set(),
            "referenced_evidence_ids": set(profile.seed_evidence_ids),
        }
    for round_result in rounds:
        recency_score = round(1.0 / (1.0 + max(0, int(round_result.round_index))), 4)
        for action in round_result.actions:
            entry = agent_state.setdefault(
                action.agent_id,
                {
                    "agent_id": action.agent_id,
                    "name": action.agent_id,
                    "stance_bias": "neutral",
                    "seed_evidence_ids": [],
                    "recent_actions": [],
                    "episode_count": 0,
                    "graph_reference_ids": [],
                    "recency_score": 0.0,
                    "referenced_feature_names": set(),
                    "referenced_evidence_ids": set(),
                },
            )
            entry["episode_count"] += 1
            entry["recent_actions"] = (
                list(entry["recent_actions"])
                + [_action_memory_entry(action, round_index=round_result.round_index)]
            )[-agent_action_cap:]
            entry["graph_reference_ids"] = _trim_unique_recent(
                list(entry["graph_reference_ids"])
                + list(action.referenced_feature_names)
                + list(action.referenced_evidence_ids),
                shared_reference_cap,
            )
            entry["recency_score"] = max(float(entry["recency_score"]), recency_score)
            entry["referenced_feature_names"].update(action.referenced_feature_names)
            entry["referenced_evidence_ids"].update(action.referenced_evidence_ids)
    individual = []
    for entry in agent_state.values():
        individual.append(
            {
                "agent_id": entry["agent_id"],
                "name": entry["name"],
                "stance_bias": entry["stance_bias"],
                "seed_evidence_ids": list(entry["seed_evidence_ids"]),
                "recent_actions": list(entry["recent_actions"]),
                "episode_count": int(entry["episode_count"]),
                "graph_reference_ids": list(entry["graph_reference_ids"]),
                "recency_score": float(entry["recency_score"]),
                "referenced_feature_names": sorted(entry["referenced_feature_names"]),
                "referenced_evidence_ids": sorted(entry["referenced_evidence_ids"]),
            }
        )
    individual.sort(key=lambda item: item["agent_id"])
    round_summaries = [round_result.summary for round_result in rounds]
    return {
        "individual_agent_count": len(individual),
        "individual": individual,
        "shared": {
            **shared,
            "episode_count": max(int(shared.get("episode_count", 0) or 0), len(rounds)),
            "round_summaries": (
                [_round_summary_payload(round_result) for round_result in rounds][-shared_round_cap:]
                if round_summaries
                else list(shared.get("round_summaries") or [])
            ),
            "total_action_count": sum(len(round_result.actions) for round_result in rounds),
        },
        "communities": list(memory.get("communities") or []),
        "episodic": [
            {
                "round_index": round_result.round_index,
                "summary": round_result.summary,
                "action_count": len(round_result.actions),
                "consensus_score": round_result.consensus_score,
                "conflict_score": round_result.conflict_score,
            }
            for round_result in rounds
        ],
        "time_awareness": {
            "round_hours": [environment.time_config.start_hour + (round_result.round_index * environment.time_config.minutes_per_round) // 60 for round_result in rounds],
            "minutes_per_round": environment.time_config.minutes_per_round,
            "start_hour": environment.time_config.start_hour,
            "decay_half_life_rounds": 3,
        },
    }


def render_memory_context(environment: Any | None, simulation_state: dict[str, object] | None, agent_id: str) -> str:
    if environment is None:
        return ""
    memory = ((simulation_state or {}).get("memory") or environment.memory_snapshot or {})
    config = dict(memory.get("config") or {})
    agent_entries = memory.get("individual") or []
    agent_entry = next((entry for entry in agent_entries if entry.get("agent_id") == agent_id), None)
    shared = memory.get("shared") or {}
    recent_actions = len((agent_entry or {}).get("recent_actions") or [])
    episode_count = int(shared.get("episode_count", 0) or 0)
    recency_score = float((agent_entry or {}).get("recency_score", 0.0) or 0.0)
    if recent_actions <= 0 and episode_count <= 0:
        return ""
    shared_context = _build_shared_context(
        shared,
        round_cap=_config_int(config, "swarm_prompt_shared_round_cap", DEFAULT_PROMPT_SHARED_ROUND_CAP),
    )
    agent_context = _build_agent_context(
        agent_entry or {},
        action_cap=_config_int(config, "swarm_prompt_agent_action_cap", DEFAULT_PROMPT_AGENT_ACTION_CAP),
    )
    parts = [f"shared_episodes={episode_count}", f"recency_score={recency_score:.2f}"]
    if shared_context:
        parts.append(f"shared_recent={shared_context}")
    if agent_context:
        parts.append(f"agent_recent={agent_context}")
    return "; ".join(parts)
