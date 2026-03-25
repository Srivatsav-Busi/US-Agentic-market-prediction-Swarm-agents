from __future__ import annotations

from collections import Counter
from typing import Any

from .simulation_state import normalize_simulation_state, summarize_state_transitions


def build_swarm_reporting_payload(
    *,
    profiles: list[dict[str, Any]] | None,
    rounds: list[dict[str, Any]] | None,
    setup: dict[str, Any] | None = None,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    profiles = list(profiles or [])
    rounds = list(rounds or [])
    setup = dict(setup or {})
    summary = dict(summary or {})
    profile_count = len(profiles)
    round_count = len(rounds) or int(summary.get("round_count", 0) or 0)
    active_counts = [len(round_payload.get("active_agent_ids") or []) for round_payload in rounds]
    total_actions = sum(len(round_payload.get("actions") or []) for round_payload in rounds)
    average_active = (sum(active_counts) / len(active_counts)) if active_counts else 0.0
    participation_rate = (average_active / profile_count) if profile_count else 0.0

    theme_stats = _build_theme_stats(rounds)
    cluster_composition = _build_cluster_composition(profiles)
    agent_activity = _build_agent_activity_summary(profiles, rounds)
    round_highlights = _build_round_highlights(rounds, profile_count, theme_stats)
    guardrails = _build_guardrail_snapshot(profiles, rounds)
    dominant_signals = _build_dominant_signals(summary, participation_rate, cluster_composition, theme_stats, rounds)

    return {
        "version": "swarm_reporting:v1",
        "scale_mode": "large" if guardrails["large_swarm"] else "standard",
        "totals": {
            "total_personas": profile_count,
            "round_count": round_count,
            "seed_post_count": len((setup.get("seed_posts") or [])),
            "total_actions": total_actions,
            "average_active_agents_per_round": round(average_active, 2),
            "min_active_agents_per_round": min(active_counts) if active_counts else 0,
            "max_active_agents_per_round": max(active_counts) if active_counts else 0,
            "participation_rate": round(participation_rate, 4),
        },
        "guardrails": guardrails,
        "dominant_signals": dominant_signals,
        "cluster_composition": cluster_composition,
        "top_debated_themes": _top_theme_items(theme_stats),
        "round_highlights": round_highlights,
        "agent_activity": agent_activity,
    }


def build_simulation_history_report(bundle: dict[str, Any]) -> dict[str, Any]:
    simulation_state = normalize_simulation_state(bundle.get("simulation_state") or {})
    queue_record = dict(bundle.get("queue_record") or {})
    memory_snapshot = dict(bundle.get("memory_snapshot") or {})
    actions = list(bundle.get("actions") or [])
    result = dict(bundle.get("result") or {})
    environment = dict(bundle.get("environment") or {})
    summary = summarize_state_transitions(simulation_state).to_dict()
    graph_summary = _summarize_graph_deltas(simulation_state.get("graph_deltas") or [])
    memory_summary = _summarize_memory_snapshot(memory_snapshot)
    decision_summary = _summarize_decision_traces((simulation_state.get("agent_state") or {}).get("decision_traces") or {})
    social_summary = _summarize_social_dynamics((simulation_state.get("agent_state") or {}).get("social_dynamics") or {})
    queue_summary = _summarize_queue_metadata(queue_record, summary=summary)
    event_summary = _summarize_event_history(simulation_state.get("event_history") or [])
    swarm_reporting = build_swarm_reporting_payload(
        profiles=environment.get("profiles") or result.get("swarm_agents") or [],
        rounds=result.get("swarm_rounds") or [],
        setup=result.get("swarm_setup") or {
            "seed_posts": environment.get("seed_posts") or [],
            "time_config": environment.get("time_config") or {},
        },
        summary=result.get("swarm_summary") or {},
    )
    return {
        "simulation_id": bundle.get("simulation_id"),
        "summary": summary,
        "queue": queue_summary,
        "events": event_summary,
        "graph": graph_summary,
        "memory": memory_summary,
        "decision_traces": decision_summary,
        "social_dynamics": social_summary,
        "latest": {
            "world_state": dict(simulation_state.get("world_state") or {}),
            "event": (simulation_state.get("event_history") or [None])[-1],
            "graph_delta": (simulation_state.get("graph_deltas") or [None])[-1],
        },
        "action_log": {
            "round_count": len(actions),
            "action_count": sum(len(item.get("actions") or []) for item in actions),
        },
        "swarm_reporting": swarm_reporting,
        "queue_status": queue_summary.get("status"),
        "retry_count": queue_summary.get("retry_count"),
        "event_count": summary["event_count"],
        "graph_delta_count": summary["graph_delta_count"],
        "decision_trace_agent_count": decision_summary["agent_count"],
        "shared_memory_episode_count": memory_summary["shared"]["episode_count"],
        "world_state": dict(simulation_state.get("world_state") or {}),
        "latest_event": (simulation_state.get("event_history") or [None])[-1],
        "latest_graph_delta": (simulation_state.get("graph_deltas") or [None])[-1],
    }


def _summarize_event_history(event_history: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(item.get("event") or "unknown") for item in event_history)
    return {
        "count": len(event_history),
        "by_type": dict(sorted(counts.items())),
        "latest": event_history[-1] if event_history else None,
    }


def _summarize_graph_deltas(graph_deltas: list[dict[str, Any]]) -> dict[str, Any]:
    created_action_nodes = sum(len(item.get("created_action_nodes") or []) for item in graph_deltas)
    referenced_features = sorted(
        {
            feature_name
            for item in graph_deltas
            for feature_name in (item.get("referenced_features") or item.get("referenced_feature_names") or [])
        }
    )
    referenced_entities = sorted(
        {
            entity_name
            for item in graph_deltas
            for entity_name in (item.get("referenced_entities") or [])
        }
    )
    return {
        "count": len(graph_deltas),
        "created_action_node_count": created_action_nodes,
        "feature_reference_count": len(referenced_features),
        "entity_reference_count": len(referenced_entities),
        "referenced_features": referenced_features,
        "referenced_entities": referenced_entities,
        "round_indexes": [int(item.get("round_index", 0) or 0) for item in graph_deltas],
        "latest": graph_deltas[-1] if graph_deltas else None,
    }


def _summarize_decision_traces(decision_traces: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    total_count = sum(len(items or []) for items in decision_traces.values())
    return {
        "agent_count": len(decision_traces),
        "total_count": total_count,
        "per_agent_counts": {
            str(agent_id): len(items or [])
            for agent_id, items in sorted(decision_traces.items())
        },
    }


def _summarize_memory_snapshot(memory_snapshot: dict[str, Any]) -> dict[str, Any]:
    individual = list(memory_snapshot.get("individual") or [])
    shared = dict(memory_snapshot.get("shared") or {})
    communities = list(memory_snapshot.get("communities") or [])
    return {
        "shared": {
            "episode_count": int(shared.get("episode_count", 0) or 0),
            "round_summary_count": len(shared.get("round_summaries") or []),
            "feature_reference_count": len(shared.get("shared_feature_names") or []),
            "evidence_reference_count": len(shared.get("shared_evidence_ids") or []),
        },
        "individual_agent_count": len(individual),
        "individual_recent_action_count": sum(len(item.get("recent_actions") or []) for item in individual),
        "community_count": len(communities),
        "dominant_communities": [
            {
                "community_id": community.get("community_id"),
                "agent_count": len(community.get("agent_ids") or []),
            }
            for community in communities[:5]
        ],
    }


def _summarize_social_dynamics(social_dynamics: dict[str, Any]) -> dict[str, Any]:
    top_conflicts = list(social_dynamics.get("top_conflicts") or [])
    top_influence_edges = list(social_dynamics.get("top_influence_edges") or [])
    dominant_communities = list(social_dynamics.get("dominant_communities") or [])
    return {
        "top_conflict_count": len(top_conflicts),
        "top_influence_edge_count": len(top_influence_edges),
        "community_count": len(dominant_communities),
        "top_conflicts": top_conflicts,
        "top_influence_edges": top_influence_edges,
        "dominant_communities": dominant_communities,
    }


def _summarize_queue_metadata(queue_record: dict[str, Any], *, summary: dict[str, Any]) -> dict[str, Any]:
    total_rounds = max(1, int(queue_record.get("total_rounds", summary.get("round_count", 0)) or 1))
    current_round = int(queue_record.get("current_round", summary.get("round_count", 0)) or 0)
    return {
        "status": queue_record.get("status"),
        "retry_count": int(queue_record.get("retry_count", 0) or 0),
        "max_retries": int(queue_record.get("max_retries", 0) or 0),
        "idempotency_key": queue_record.get("idempotency_key"),
        "queued_at": queue_record.get("queued_at"),
        "started_at": queue_record.get("started_at"),
        "completed_at": queue_record.get("completed_at"),
        "last_heartbeat_at": queue_record.get("last_heartbeat_at"),
        "worker_pool_size": int(queue_record.get("worker_pool_size", 0) or 0),
        "execution_options": dict(queue_record.get("execution_options") or {}),
        "progress": {
            "current_round": current_round,
            "total_rounds": total_rounds,
            "percent_complete": round(min(100.0, (current_round / total_rounds) * 100.0), 2),
        },
    }


def _build_guardrail_snapshot(profiles: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    active_counts = [len(round_payload.get("active_agent_ids") or []) for round_payload in rounds]
    profile_ids = [str(profile.get("agent_id") or "") for profile in profiles if profile.get("agent_id")]
    return {
        "exactly_50_personas": len(profiles) == 50,
        "at_least_10_rounds": len(rounds) >= 10,
        "at_least_10_active_agents_each_round": bool(active_counts) and min(active_counts) >= 10,
        "no_duplicate_ids": len(profile_ids) == len(set(profile_ids)),
        "large_swarm": len(profiles) >= 50 or len(rounds) >= 10 or (active_counts and min(active_counts) >= 10),
    }


def _build_cluster_composition(profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    sample_archetypes: dict[str, Counter[str]] = {}
    total = max(len(profiles), 1)
    for profile in profiles:
        focus_categories = list(profile.get("focus_categories") or [])
        cluster_key = str(
            profile.get("coverage_bucket")
            or (focus_categories[0] if focus_categories else "")
            or profile.get("entity_type")
            or "market"
        )
        counts[cluster_key] += 1
        sample_archetypes.setdefault(cluster_key, Counter())[str(profile.get("archetype") or "unknown")] += 1
    items = []
    for cluster_key, count in counts.most_common():
        items.append(
            {
                "cluster_id": cluster_key,
                "label": cluster_key.replace("_", " ").title(),
                "agent_count": count,
                "share": round(count / total, 4),
                "top_archetypes": [name for name, _ in sample_archetypes.get(cluster_key, Counter()).most_common(3)],
            }
        )
    return items


def _build_theme_stats(rounds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for round_payload in rounds:
        round_index = int(round_payload.get("round_index", 0) or 0)
        stance_histogram = dict(round_payload.get("stance_histogram") or {})
        round_conflict = float(round_payload.get("conflict_score", 0.0) or 0.0)
        for action in round_payload.get("actions") or []:
            theme_names = [
                str(name).strip()
                for name in (
                    action.get("referenced_feature_names")
                    or action.get("referenced_features")
                    or []
                )
                if str(name).strip()
            ]
            if not theme_names:
                action_type = str(action.get("action_type") or "discussion").strip()
                direction = str(action.get("direction") or "mixed").strip()
                theme_names = [f"{action_type}:{direction}"]
            for theme_name in theme_names:
                item = stats.setdefault(
                    theme_name,
                    {
                        "theme": theme_name,
                        "mentions": 0,
                        "agent_ids": set(),
                        "round_indexes": set(),
                        "conflict_weight": 0.0,
                        "bullish_mentions": 0,
                        "bearish_mentions": 0,
                    },
                )
                item["mentions"] += 1
                agent_id = str(action.get("agent_id") or "")
                if agent_id:
                    item["agent_ids"].add(agent_id)
                item["round_indexes"].add(round_index)
                item["conflict_weight"] += round_conflict
                item["bullish_mentions"] += int(stance_histogram.get("bullish", 0) or 0)
                item["bearish_mentions"] += int(stance_histogram.get("bearish", 0) or 0)
    return stats


def _top_theme_items(theme_stats: dict[str, dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    ranked = sorted(
        theme_stats.values(),
        key=lambda item: (
            -int(item["mentions"]),
            -len(item["agent_ids"]),
            -len(item["round_indexes"]),
            str(item["theme"]),
        ),
    )
    items = []
    for item in ranked[:limit]:
        items.append(
            {
                "theme": item["theme"],
                "mentions": item["mentions"],
                "distinct_agents": len(item["agent_ids"]),
                "round_span": len(item["round_indexes"]),
                "conflict_weight": round(float(item["conflict_weight"]), 4),
                "stance_balance": _theme_stance_balance(item),
            }
        )
    return items


def _theme_stance_balance(item: dict[str, Any]) -> str:
    bullish_mentions = int(item.get("bullish_mentions", 0) or 0)
    bearish_mentions = int(item.get("bearish_mentions", 0) or 0)
    if bullish_mentions > bearish_mentions:
        return "bullish"
    if bearish_mentions > bullish_mentions:
        return "bearish"
    return "mixed"


def _build_agent_activity_summary(profiles: list[dict[str, Any]], rounds: list[dict[str, Any]]) -> dict[str, Any]:
    profile_by_id = {
        str(profile.get("agent_id")): profile
        for profile in profiles
        if profile.get("agent_id")
    }
    agent_stats: dict[str, dict[str, Any]] = {}
    for round_payload in rounds:
        round_index = int(round_payload.get("round_index", 0) or 0)
        active_agent_ids = [str(agent_id) for agent_id in (round_payload.get("active_agent_ids") or []) if agent_id]
        for agent_id in active_agent_ids:
            payload = agent_stats.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "rounds_active": set(),
                    "action_count": 0,
                    "comment_count": 0,
                    "post_count": 0,
                },
            )
            payload["rounds_active"].add(round_index)
        for action in round_payload.get("actions") or []:
            agent_id = str(action.get("agent_id") or "")
            if not agent_id:
                continue
            payload = agent_stats.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "rounds_active": set(),
                    "action_count": 0,
                    "comment_count": 0,
                    "post_count": 0,
                },
            )
            payload["action_count"] += 1
            if str(action.get("action_type") or "") == "create_post":
                payload["post_count"] += 1
            else:
                payload["comment_count"] += 1
            payload["rounds_active"].add(round_index)

    items = []
    for agent_id, payload in agent_stats.items():
        profile = profile_by_id.get(agent_id, {})
        rounds_active = sorted(payload["rounds_active"])
        item = {
            "agent_id": agent_id,
            "name": profile.get("name") or agent_id,
            "stance_bias": profile.get("stance_bias") or "neutral",
            "cluster": (
                profile.get("coverage_bucket")
                or ((profile.get("focus_categories") or [None])[0])
                or profile.get("entity_type")
                or "market"
            ),
            "rounds_active": rounds_active,
            "action_count": payload["action_count"],
            "post_count": payload["post_count"],
            "comment_count": payload["comment_count"],
        }
        items.append(item)

    items.sort(key=lambda item: (-item["action_count"], -len(item["rounds_active"]), item["agent_id"]))
    highlighted = []
    low_priority = []
    for index, item in enumerate(items):
        enriched = {
            **item,
            "priority": "high" if index < 6 and item["action_count"] > 0 else "low",
        }
        if enriched["priority"] == "high":
            highlighted.append(enriched)
        else:
            low_priority.append(enriched)
    return {
        "highlighted_agents": highlighted,
        "low_priority_agents": low_priority[:12],
        "low_priority_count": len(low_priority),
    }


def _build_round_highlights(
    rounds: list[dict[str, Any]],
    profile_count: int,
    theme_stats: dict[str, dict[str, Any]],
    limit: int = 6,
) -> list[dict[str, Any]]:
    highlights = []
    for round_payload in rounds[:limit]:
        round_index = int(round_payload.get("round_index", 0) or 0)
        top_themes = []
        for item in _top_theme_items(
            {
                theme: stats
                for theme, stats in theme_stats.items()
                if round_index in stats["round_indexes"]
            },
            limit=3,
        ):
            top_themes.append(item["theme"])
        active_agents = len(round_payload.get("active_agent_ids") or [])
        highlights.append(
            {
                "round_index": round_index,
                "active_agents": active_agents,
                "participation_rate": round((active_agents / profile_count), 4) if profile_count else 0.0,
                "dominant_stance": _dominant_stance(round_payload.get("stance_histogram") or {}),
                "consensus_score": round(float(round_payload.get("consensus_score", 0.0) or 0.0), 4),
                "conflict_score": round(float(round_payload.get("conflict_score", 0.0) or 0.0), 4),
                "summary": str(round_payload.get("summary") or ""),
                "top_themes": top_themes,
            }
        )
    return highlights


def _dominant_stance(stance_histogram: dict[str, Any]) -> str:
    bullish = int(stance_histogram.get("bullish", 0) or 0)
    bearish = int(stance_histogram.get("bearish", 0) or 0)
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    return "mixed"


def _build_dominant_signals(
    summary: dict[str, Any],
    participation_rate: float,
    cluster_composition: list[dict[str, Any]],
    theme_stats: dict[str, dict[str, Any]],
    rounds: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signals = [
        {
            "label": "Dominant stance",
            "value": str(summary.get("dominant_stance") or "mixed"),
            "tone": str(summary.get("dominant_stance") or "mixed"),
            "summary": "Overall stance across swarm rounds.",
        },
        {
            "label": "Participation",
            "value": round(participation_rate, 4),
            "tone": "positive" if participation_rate >= 0.3 else "neutral",
            "summary": "Average active agents per round as a share of the total persona pool.",
        },
    ]
    if cluster_composition:
        lead_cluster = cluster_composition[0]
        signals.append(
            {
                "label": "Largest cluster",
                "value": lead_cluster["label"],
                "tone": "neutral",
                "summary": f"{lead_cluster['agent_count']} agents ({lead_cluster['share']:.0%}) concentrated in the leading focus cluster.",
            }
        )
    top_themes = _top_theme_items(theme_stats, limit=1)
    if top_themes:
        theme = top_themes[0]
        signals.append(
            {
                "label": "Most debated theme",
                "value": theme["theme"],
                "tone": theme["stance_balance"],
                "summary": f"Mentioned {theme['mentions']} times across {theme['round_span']} rounds by {theme['distinct_agents']} agents.",
            }
        )
    if rounds:
        max_conflict_round = max(rounds, key=lambda item: float(item.get("conflict_score", 0.0) or 0.0))
        signals.append(
            {
                "label": "Peak conflict",
                "value": f"Round {int(max_conflict_round.get('round_index', 0) or 0) + 1}",
                "tone": "warning",
                "summary": str(max_conflict_round.get("summary") or "Highest-conflict round."),
            }
        )
    return signals[:5]
