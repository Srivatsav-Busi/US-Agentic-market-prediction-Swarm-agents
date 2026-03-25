from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from random import Random
from time import perf_counter
from typing import Any
from uuid import uuid4

from .config import (
    SWARM_DEFAULT_AGENTS_PER_ROUND_MAX,
    SWARM_DEFAULT_AGENTS_PER_ROUND_MIN,
    SWARM_DEFAULT_PERSONA_COUNT,
    SWARM_DEFAULT_ROUNDS,
)
from .environment_builder import GraphDrivenEnvironmentBuilder
from .llm_clients import BaseLLMClient, clone_llm_client, with_max_completion_tokens
from .memory import bootstrap_memory_snapshot, evolve_memory_snapshot, render_memory_context, summarize_memory_diagnostics
from .models import SignalFeature, SourceItem
from .simulation_state import iso_utc_now, normalize_graph_delta, normalize_simulation_state

@dataclass
class AgentProfile:
    agent_id: str
    name: str
    username: str
    archetype: str
    entity_name: str
    entity_type: str
    stance_bias: str
    focus_categories: list[str]
    activity_level: float
    active_rounds: list[int]
    influence_weight: float
    system_prompt: str
    bio: str = ""
    persona: str = ""
    seed_evidence_ids: list[str] = field(default_factory=list)
    cluster: str = "market_structure"
    persona_layer: str = "stable"
    template_id: str = ""
    variant_key: str = ""
    category_focus_label: str = ""
    activity_bucket: str = "medium"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AgentBehaviorConfig:
    agent_id: str
    entity_name: str
    entity_type: str
    activity_level: float
    posts_per_hour: float
    comments_per_hour: float
    active_hours: list[int]
    response_delay_min: int
    response_delay_max: int
    sentiment_bias: float
    stance: str
    influence_weight: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TimeSimulationConfig:
    total_rounds: int
    start_hour: int
    minutes_per_round: int
    min_active_agents: int
    max_active_agents: int
    peak_hours: list[int]
    offpeak_hours: list[int]
    peak_activity_multiplier: float
    offpeak_activity_multiplier: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SwarmAction:
    round_index: int
    agent_id: str
    action_type: str
    target_agent_id: str | None
    content: str
    direction: str
    strength: float
    referenced_feature_names: list[str] = field(default_factory=list)
    referenced_evidence_ids: list[str] = field(default_factory=list)
    decision_trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SocialEdge:
    source_agent_id: str
    target_agent_id: str
    affinity_score: float
    trust_score: float
    influence_score: float
    conflict_score: float
    shared_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SwarmRoundResult:
    round_index: int
    active_agent_ids: list[str]
    actions: list[SwarmAction]
    summary: str
    stance_histogram: dict[str, int]
    consensus_score: float
    conflict_score: float
    selection_diagnostics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["actions"] = [action.to_dict() for action in self.actions]
        return payload


@dataclass
class SwarmSetupResult:
    profiles: list[AgentProfile]
    activity_configs: list[AgentBehaviorConfig]
    time_config: TimeSimulationConfig
    seed_posts: list[SwarmAction]
    social_edges: list[SocialEdge]
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "profiles": [profile.to_dict() for profile in self.profiles],
            "activity_configs": [config.to_dict() for config in self.activity_configs],
            "time_config": self.time_config.to_dict(),
            "seed_posts": [post.to_dict() for post in self.seed_posts],
            "social_edges": [edge.to_dict() for edge in self.social_edges],
            "diagnostics": self.diagnostics,
        }


@dataclass
class SimulationEnvironment:
    environment_id: str
    environment_version: str
    created_at: str
    mode: str
    base_run_stem: str
    prediction_date: str
    target: str
    input_feature_names: list[str]
    profiles: list[AgentProfile]
    activity_configs: list[AgentBehaviorConfig]
    time_config: TimeSimulationConfig
    seed_posts: list[SwarmAction]
    social_edges: list[SocialEdge]
    memory_snapshot: dict
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "created_at": self.created_at,
            "mode": self.mode,
            "base_run_stem": self.base_run_stem,
            "prediction_date": self.prediction_date,
            "target": self.target,
            "input_feature_names": list(self.input_feature_names),
            "profiles": [profile.to_dict() for profile in self.profiles],
            "activity_configs": [config.to_dict() for config in self.activity_configs],
            "time_config": self.time_config.to_dict(),
            "seed_posts": [post.to_dict() for post in self.seed_posts],
            "social_edges": [edge.to_dict() for edge in self.social_edges],
            "memory_snapshot": self.memory_snapshot,
            "diagnostics": self.diagnostics,
        }

    def setup_result(self) -> SwarmSetupResult:
        return SwarmSetupResult(
            profiles=self.profiles,
            activity_configs=self.activity_configs,
            time_config=self.time_config,
            seed_posts=self.seed_posts,
            social_edges=self.social_edges,
            diagnostics=self.diagnostics,
        )

    def summary(self) -> dict[str, str | int]:
        return {
            "environment_id": self.environment_id,
            "environment_version": self.environment_version,
            "mode": self.mode,
            "profile_count": len(self.profiles),
            "dynamic_profile_count": len([profile for profile in self.profiles if profile.archetype.startswith("dynamic_")]),
            "seed_post_count": len(self.seed_posts),
            "social_edge_count": len(self.social_edges),
            "round_count": self.time_config.total_rounds,
        }


@dataclass
class SwarmSimulationResult:
    environment: SimulationEnvironment
    setup: SwarmSetupResult
    profiles: list[AgentProfile]
    seed_posts: list[SwarmAction]
    rounds: list[SwarmRoundResult]
    derived_features: list[SignalFeature]
    summary_metrics: dict[str, float | int | str]
    diagnostics: dict
    priors: dict[str, float]
    simulation_state: dict[str, object]

    def to_dict(self) -> dict:
        return {
            "environment": self.environment.to_dict(),
            "setup": self.setup.to_dict(),
            "profiles": [profile.to_dict() for profile in self.profiles],
            "seed_posts": [post.to_dict() for post in self.seed_posts],
            "rounds": [round_result.to_dict() for round_result in self.rounds],
            "derived_features": [feature.to_dict() for feature in self.derived_features],
            "summary_metrics": self.summary_metrics,
            "diagnostics": self.diagnostics,
            "priors": self.priors,
            "simulation_state": self.simulation_state,
        }

def run_swarm_simulation(
    items: list[SourceItem],
    snapshot: dict,
    features: list[SignalFeature],
    llm_client: BaseLLMClient,
    config: dict,
    target: str,
) -> SwarmSimulationResult:
    environment = prepare_swarm_environment(
        items=items,
        snapshot=snapshot,
        features=features,
        llm_client=llm_client,
        config=config,
        target=target,
    )
    return run_swarm_from_environment(
        environment=environment,
        items=items,
        snapshot=snapshot,
        features=features,
        llm_client=llm_client,
        config=config,
        target=target,
    )


def prepare_swarm_environment(
    items: list[SourceItem],
    snapshot: dict,
    features: list[SignalFeature],
    llm_client: BaseLLMClient,
    config: dict,
    target: str,
    graph=None,
    memory_snapshot: dict | None = None,
    *,
    mode: str = "daily_run",
    base_run_stem: str = "",
    prediction_date: str = "",
) -> SimulationEnvironment:
    del snapshot, llm_client
    seed = int(config.get("swarm_random_seed", 42) or 42)
    time_config = _build_time_config(config)
    if not config.get("swarm_enabled", True):
        prior_memory = dict(memory_snapshot or {})
        prior_memory["config"] = _memory_config_from_config(config)
        return SimulationEnvironment(
            environment_id=f"env_{uuid4().hex[:12]}",
            environment_version="swarm_environment:v2",
            created_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
            mode=mode,
            base_run_stem=base_run_stem,
            prediction_date=prediction_date,
            target=target,
            input_feature_names=[feature.name for feature in features],
            profiles=[],
            activity_configs=[],
            time_config=time_config,
            seed_posts=[],
            social_edges=[],
            memory_snapshot=bootstrap_memory_snapshot(profiles=[], social_edges=[], prior_memory=prior_memory),
            diagnostics={"enabled": False, "seed": seed},
        )

    rng = Random(seed)
    builder = GraphDrivenEnvironmentBuilder(graph=graph, items=items, features=features, config=config, rng=rng)
    profiles = builder.build_profiles()
    for profile in profiles:
        profile.coverage_bucket = _resolve_coverage_bucket(
            archetype=profile.archetype,
            focus_categories=list(profile.focus_categories),
            entity_type=profile.entity_type,
        )
    activity_configs = _build_behavior_configs(profiles, config, rng)
    social_edges = builder.build_social_edges(profiles)
    prior_memory = dict(memory_snapshot or {})
    prior_memory["config"] = _memory_config_from_config(config)
    memory_snapshot_payload = bootstrap_memory_snapshot(profiles=profiles, social_edges=social_edges, prior_memory=prior_memory)
    seed_posts = _build_seed_posts(features, items, profiles, activity_configs, config)
    diagnostics = builder.build_diagnostics(
        profiles=profiles,
        activity_configs=activity_configs,
        seed_posts=seed_posts,
        social_edges=social_edges,
        memory_snapshot=memory_snapshot_payload,
    )
    diagnostics["enabled"] = True
    diagnostics["seed"] = seed
    return SimulationEnvironment(
        environment_id=f"env_{uuid4().hex[:12]}",
        environment_version="swarm_environment:v2",
        created_at=datetime.now().astimezone().replace(microsecond=0).isoformat(),
        mode=mode,
        base_run_stem=base_run_stem,
        prediction_date=prediction_date,
        target=target,
        input_feature_names=[feature.name for feature in features],
        profiles=profiles,
        activity_configs=activity_configs,
        time_config=time_config,
        seed_posts=seed_posts,
        social_edges=social_edges,
        memory_snapshot=memory_snapshot_payload,
        diagnostics=diagnostics,
    )


def run_swarm_from_environment(
    environment: SimulationEnvironment,
    items: list[SourceItem],
    snapshot: dict,
    features: list[SignalFeature],
    llm_client: BaseLLMClient,
    config: dict,
    target: str,
    *,
    round_callback=None,
) -> SwarmSimulationResult:
    del snapshot
    seed = int(config.get("swarm_random_seed", 42) or 42)
    rng = Random(seed)
    started_at = perf_counter()
    setup = environment.setup_result()
    profiles = list(environment.profiles)
    activity_configs = list(environment.activity_configs)
    time_config = environment.time_config
    seed_posts = list(environment.seed_posts)

    if not config.get("swarm_enabled", True):
        return SwarmSimulationResult(
            environment=environment,
            setup=setup,
            profiles=[],
            seed_posts=[],
            rounds=[],
            derived_features=[],
            summary_metrics={"enabled": 0},
            diagnostics={"enabled": False},
            priors={"swarm_up_bias": 0.0, "swarm_down_bias": 0.0, "swarm_confidence_modifier": 0.0},
            simulation_state={},
        )

    rounds: list[SwarmRoundResult] = []
    simulation_state = _make_initial_simulation_state(seed_posts=seed_posts, environment=environment, config=config)
    for round_index in range(time_config.total_rounds):
        selection_plan = _select_active_agents(
            profiles,
            activity_configs,
            time_config,
            round_index,
            rng,
            simulation_state,
        )
        active_profiles = [item["profile"] for item in selection_plan]
        actions, selection_diagnostics = _build_round_actions(
            round_index=round_index,
            selection_plan=selection_plan,
            seed_posts=seed_posts if round_index == 0 else [],
            features=features,
            items=items,
            llm_client=llm_client,
            target=target,
            rng=rng,
            config=config,
            simulation_state=simulation_state,
            environment=environment,
        )
        round_result = _summarize_round(round_index, active_profiles, actions, selection_diagnostics)
        rounds.append(round_result)
        simulation_state = _advance_simulation_state(simulation_state, round_result, config=config)
        if round_callback is not None:
            round_callback(round_result)

    derived_features = derive_swarm_features(rounds, profiles, items, features, config)
    priors = _build_swarm_priors(rounds, config)
    summary_metrics = _build_summary_metrics(profiles, seed_posts, rounds, derived_features)
    runtime_seconds = round(perf_counter() - started_at, 4)
    diagnostics = {
        "enabled": True,
        "seed": seed,
        "llm_provider": llm_client.diagnostics().get("provider", "unknown"),
        "runtime_seconds": runtime_seconds,
        "agent_count": len(profiles),
        "dynamic_agent_count": len([profile for profile in profiles if profile.archetype.startswith("dynamic_")]),
        "seed_post_count": len(seed_posts),
        "round_count": len(rounds),
        "derived_feature_count": len(derived_features),
        "fallback_mode": llm_client.diagnostics().get("backend_used") is False,
        "setup_agent_count": len(activity_configs),
        "setup_peak_hours": time_config.peak_hours,
        "setup_seeded_agents": sorted({post.agent_id for post in seed_posts}),
        "memory": summarize_memory_diagnostics(environment=environment, rounds=rounds),
        "stateful_runtime": _build_stateful_runtime_diagnostics(simulation_state),
    }
    return SwarmSimulationResult(
        environment=environment,
        setup=setup,
        profiles=profiles,
        seed_posts=seed_posts,
        rounds=rounds,
        derived_features=derived_features,
        summary_metrics=summary_metrics,
        diagnostics=diagnostics,
        priors=priors,
        simulation_state=simulation_state,
    )


def derive_swarm_features(
    rounds: list[SwarmRoundResult],
    profiles: list[AgentProfile],
    items: list[SourceItem],
    features: list[SignalFeature],
    config: dict,
) -> list[SignalFeature]:
    if not rounds:
        return []

    evidence_ids = [item.id for item in items][:8]
    feature_names = {feature.name for feature in features}
    profile_by_id = {profile.agent_id: profile for profile in profiles}
    consensus_values: list[float] = []
    conflict_values: list[float] = []
    bullish_share_values: list[float] = []
    bearish_share_values: list[float] = []
    momentum_values: list[float] = []
    market_focus_values: list[float] = []
    policy_focus_values: list[float] = []
    sector_focus_values: list[float] = []
    total_challenges = 0
    sufficient_participation_rounds = 0
    min_participation = float(config.get("swarm_derived_min_participation", 0.25) or 0.25)
    for round_result in rounds:
        stats = _compute_round_statistics(round_result)
        if stats["active_participation_rate"] >= min_participation:
            sufficient_participation_rounds += 1
        consensus_values.append(stats["consensus_score"])
        conflict_values.append(stats["conflict_score"])
        bullish_share_values.append(stats["bullish_share"])
        bearish_share_values.append(stats["bearish_share"])
        total_challenges += int(stats["challenge_count"])
        active_actions = max(1, len(round_result.actions))
        market_focus = 0
        policy_focus = 0
        momentum_focus = 0
        sector_focus = 0
        for action in round_result.actions:
            refs = set(action.referenced_feature_names)
            if refs & {"yield_rising", "vix_spike", "dxy_strength"}:
                market_focus += 1
            if "policy_uncertainty_increase" in refs:
                policy_focus += 1
            if refs & {"retail_euphoric_tone", "breadth_risk_on", "narrative_alignment"}:
                momentum_focus += 1
            profile = profile_by_id.get(action.agent_id)
            if profile is not None and "sector" in profile.archetype:
                sector_focus += 1
        market_focus_values.append(market_focus / active_actions)
        policy_focus_values.append(policy_focus / active_actions)
        momentum_values.append(momentum_focus / active_actions)
        sector_focus_values.append(sector_focus / active_actions)

    max_influence = float(config.get("swarm_max_influence", 0.12) or 0.12)
    feature_cap = max(0.3, min(1.0, max_influence / 0.12))
    consensus_threshold = float(config.get("swarm_derived_consensus_threshold", 0.18) or 0.18)
    conflict_threshold = float(config.get("swarm_derived_conflict_threshold", 0.22) or 0.22)
    derived: list[SignalFeature] = []

    def add(name: str, direction: str, raw_strength: float, category: str, summary: str, conflict_count: int = 0) -> None:
        strength = round(min(1.0, max(0.0, raw_strength * feature_cap)), 3)
        if strength <= 0:
            return
        derived.append(
            SignalFeature(
                name=name,
                direction=direction,
                strength=strength,
                supporting_evidence_ids=evidence_ids,
                conflict_count=conflict_count,
                time_decay_weight=0.95,
                category=category,
                summary=summary,
            )
        )

    average_consensus = sum(consensus_values) / max(len(consensus_values), 1)
    average_conflict = sum(conflict_values) / max(len(conflict_values), 1)
    average_bullish_share = sum(bullish_share_values) / max(len(bullish_share_values), 1)
    average_bearish_share = sum(bearish_share_values) / max(len(bearish_share_values), 1)
    if sufficient_participation_rounds > 0:
        if average_bullish_share > average_bearish_share and average_consensus >= consensus_threshold:
            add("swarm_bullish_consensus", "bullish", average_consensus, "market", "Swarm discussion skewed bullish across active participants.")
        elif average_bearish_share > average_bullish_share and average_consensus >= consensus_threshold:
            add("swarm_bearish_consensus", "bearish", average_consensus, "market", "Swarm discussion skewed bearish across active participants.")
    if average_conflict >= conflict_threshold:
        add("swarm_conflict_spike", "bearish", average_conflict, "market", "The swarm displayed elevated disagreement normalized by active participation.", conflict_count=total_challenges)
    average_market_focus = sum(market_focus_values) / max(len(market_focus_values), 1)
    average_policy_focus = sum(policy_focus_values) / max(len(policy_focus_values), 1)
    average_momentum = sum(momentum_values) / max(len(momentum_values), 1)
    average_sector_focus = sum(sector_focus_values) / max(len(sector_focus_values), 1)
    if average_market_focus >= min_participation * 0.6:
        add("swarm_macro_risk_focus", "bearish" if "yield_rising" in feature_names or "vix_spike" in feature_names else "bullish", average_market_focus, "economic", "Swarm attention concentrated on macro and volatility risk signals.")
    if average_policy_focus >= min_participation * 0.5:
        add("swarm_policy_fear", "bearish", average_policy_focus, "political", "Swarm discussion repeatedly centered on policy and geopolitical uncertainty.")
    if average_momentum >= min_participation * 0.5:
        direction = "bullish" if average_bullish_share >= average_bearish_share else "bearish"
        add("swarm_momentum_chase", direction, average_momentum, "social", "Swarm behavior amplified momentum and crowd-tone narratives.")
    if average_sector_focus >= min_participation * 0.4:
        direction = "bullish" if average_bullish_share >= average_bearish_share else "bearish"
        add("swarm_sector_rotation", direction, average_sector_focus, "market", "Swarm discussion emphasized sector leadership and rotation themes.")
    return derived

def _build_behavior_configs(
    profiles: list[AgentProfile],
    config: dict,
    rng: Random,
) -> list[AgentBehaviorConfig]:
    seed = int(config.get("swarm_random_seed", 42) or 42)
    builder = GraphDrivenEnvironmentBuilder(graph=None, items=[], features=[], config=config, rng=Random(seed))
    activity_configs: list[AgentBehaviorConfig] = []
    for profile in profiles:
        overrides = builder.build_behavior_overrides(profile)
        sentiment_bias = 0.15 if profile.stance_bias == "bullish" else -0.15 if profile.stance_bias == "bearish" else 0.0
        activity_level = round(min(1.0, max(0.35, profile.activity_level + rng.uniform(-0.04, 0.04))), 3)
        activity_configs.append(
            AgentBehaviorConfig(
                agent_id=profile.agent_id,
                entity_name=profile.entity_name,
                entity_type=profile.entity_type,
                activity_level=activity_level,
                posts_per_hour=round(float(overrides["posts_per_hour"]) + rng.uniform(-0.06, 0.06), 3),
                comments_per_hour=round(float(overrides["comments_per_hour"]) + rng.uniform(-0.1, 0.1), 3),
                active_hours=list(overrides["active_hours"]),
                response_delay_min=int(overrides["response_delay_min"]),
                response_delay_max=int(overrides["response_delay_max"]),
                sentiment_bias=round(sentiment_bias, 3),
                stance=profile.stance_bias,
                influence_weight=profile.influence_weight,
            )
        )
    return activity_configs


def _build_time_config(config: dict) -> TimeSimulationConfig:
    min_count = max(1, int(config.get("swarm_agents_per_round_min", SWARM_DEFAULT_AGENTS_PER_ROUND_MIN) or SWARM_DEFAULT_AGENTS_PER_ROUND_MIN))
    max_count = max(min_count, int(config.get("swarm_agents_per_round_max", SWARM_DEFAULT_AGENTS_PER_ROUND_MAX) or SWARM_DEFAULT_AGENTS_PER_ROUND_MAX))
    return TimeSimulationConfig(
        total_rounds=max(1, int(config.get("swarm_rounds", SWARM_DEFAULT_ROUNDS) or SWARM_DEFAULT_ROUNDS)),
        start_hour=int(config.get("swarm_start_hour", 8) or 8),
        minutes_per_round=int(config.get("swarm_minutes_per_round", 120) or 120),
        min_active_agents=min_count,
        max_active_agents=max_count,
        peak_hours=list(config.get("swarm_peak_hours", [9, 10, 11, 14, 15])),
        offpeak_hours=list(config.get("swarm_offpeak_hours", [6, 7, 12, 13, 16, 17, 18])),
        peak_activity_multiplier=float(config.get("swarm_peak_activity_multiplier", 1.2) or 1.2),
        offpeak_activity_multiplier=float(config.get("swarm_offpeak_activity_multiplier", 0.78) or 0.78),
    )


def _build_seed_posts(
    features: list[SignalFeature],
    items: list[SourceItem],
    profiles: list[AgentProfile],
    activity_configs: list[AgentBehaviorConfig],
    config: dict,
) -> list[SwarmAction]:
    if not profiles:
        return []
    config_by_id = {item.agent_id: item for item in activity_configs}
    seed_posts = max(1, int(config.get("swarm_seed_posts", 3) or 3))
    ranked_features = sorted(features, key=lambda feature: feature.strength, reverse=True)
    actions: list[SwarmAction] = []
    for feature in ranked_features[:seed_posts]:
        profile = _match_seed_profile(feature.category, profiles, config_by_id)
        actions.append(
            SwarmAction(
                round_index=-1,
                agent_id=profile.agent_id,
                action_type="create_post",
                target_agent_id=None,
                content=f"{profile.name}: opening view on {feature.name.replace('_', ' ')}.",
                direction=feature.direction,
                strength=round(feature.strength, 3),
                referenced_feature_names=[feature.name],
                referenced_evidence_ids=list(feature.supporting_evidence_ids[:4]),
            )
        )
    if not actions and items:
        item = items[0]
        profile = profiles[0]
        actions.append(
            SwarmAction(
                round_index=-1,
                agent_id=profile.agent_id,
                action_type="create_post",
                target_agent_id=None,
                content=f"{profile.name}: opening read on {item.title}.",
                direction=item.direction if item.direction in {"bullish", "bearish"} else "neutral",
                strength=round(abs(item.impact_score), 3),
                referenced_feature_names=[],
                referenced_evidence_ids=[item.id],
            )
        )
    return actions

def _select_active_agents(
    profiles: list[AgentProfile],
    activity_configs: list[AgentBehaviorConfig],
    time_config: TimeSimulationConfig,
    round_index: int,
    rng: Random,
    simulation_state: dict[str, object] | None = None,
) -> list[dict]:
    del rng
    current_hour = _simulated_hour_for_round(time_config, round_index)
    multiplier = 1.0
    if current_hour in time_config.peak_hours:
        multiplier = time_config.peak_activity_multiplier
    elif current_hour in time_config.offpeak_hours:
        multiplier = time_config.offpeak_activity_multiplier
    if simulation_state:
        previous_conflict = float(simulation_state.get("latest_conflict_score", 0.0) or 0.0)
        previous_consensus = float(simulation_state.get("latest_consensus_score", 0.0) or 0.0)
        multiplier *= 1.0 + min(0.18, previous_conflict * 0.15)
        if previous_consensus < 0.4:
            multiplier *= 1.05

    seed = int((simulation_state or {}).get("selection_seed", 42) or 42)
    required_buckets = _required_coverage_buckets(simulation_state)
    diversity_min_buckets = int((simulation_state or {}).get("selection_diversity_min_buckets", len(required_buckets)) or len(required_buckets))
    config_by_id = {item.agent_id: item for item in activity_configs}
    recent_history = list((simulation_state or {}).get("active_agent_history") or [])
    recent_agent_counts = Counter(agent_id for entry in recent_history[-3:] for agent_id in (entry.get("active_agent_ids") or []))
    recent_bucket_counts = Counter(bucket for entry in recent_history[-3:] for bucket in (entry.get("coverage_buckets") or []))
    eligible: list[dict] = []
    for profile in profiles:
        if round_index not in profile.active_rounds:
            continue
        behavior = config_by_id.get(profile.agent_id)
        if behavior is None or current_hour not in behavior.active_hours:
            continue
        bucket = getattr(profile, "coverage_bucket", "market") or "market"
        base_score = behavior.activity_level * behavior.influence_weight * multiplier
        recency_penalty = min(0.18, recent_agent_counts.get(profile.agent_id, 0) * 0.06)
        bucket_penalty = min(0.12, recent_bucket_counts.get(bucket, 0) * 0.025)
        jitter = _stable_fraction(seed, round_index, profile.agent_id, "eligibility")
        eligible.append(
            {
                "profile": profile,
                "coverage_bucket": bucket,
                "selection_score": round(base_score - recency_penalty - bucket_penalty + jitter * 0.03, 6),
                "selection_reason_codes": ["base_score", f"hour_{current_hour}"],
                "required_bucket": bucket in required_buckets,
            }
        )

    if len(eligible) < time_config.min_active_agents:
        eligible_ids = {item["profile"].agent_id for item in eligible}
        for profile in profiles:
            if len(eligible) >= time_config.min_active_agents:
                break
            if profile.agent_id in eligible_ids or round_index not in profile.active_rounds:
                continue
            behavior = config_by_id.get(profile.agent_id)
            if behavior is None:
                continue
            bucket = getattr(profile, "coverage_bucket", "market") or "market"
            jitter = _stable_fraction(seed, round_index, profile.agent_id, "guardrail_fill")
            eligible.append(
                {
                    "profile": profile,
                    "coverage_bucket": bucket,
                    "selection_score": round((behavior.activity_level * behavior.influence_weight * 0.92) + jitter * 0.02, 6),
                    "selection_reason_codes": ["guardrail_fill", f"hour_{current_hour}"],
                    "required_bucket": bucket in required_buckets,
                }
            )
            eligible_ids.add(profile.agent_id)

    eligible.sort(
        key=lambda item: (
            1 if item["required_bucket"] else 0,
            item["selection_score"],
            item["profile"].influence_weight,
            item["profile"].agent_id,
        ),
        reverse=True,
    )
    if len(eligible) < max(1, time_config.min_active_agents):
        for profile in profiles:
            if profile.agent_id in {item["profile"].agent_id for item in eligible}:
                continue
            behavior = config_by_id.get(profile.agent_id)
            if behavior is None or round_index not in profile.active_rounds:
                continue
            bucket = getattr(profile, "coverage_bucket", "market") or "market"
            eligible.append(
                {
                    "profile": profile,
                    "coverage_bucket": bucket,
                    "selection_score": round(behavior.activity_level * behavior.influence_weight * max(multiplier, 0.75), 6),
                    "selection_reason_codes": ["fallback_capacity", f"hour_{current_hour}"],
                    "required_bucket": bucket in required_buckets,
                }
            )

    if not eligible:
        return []

    min_count = min(len(eligible), max(1, time_config.min_active_agents))
    max_count = max(min_count, min(len(eligible), time_config.max_active_agents))
    count = _stable_int(seed, round_index, "round_size", min_count, max_count)
    selected: list[dict] = []
    selected_ids: set[str] = set()
    bucket_counts: Counter[str] = Counter()
    archetype_counts: Counter[str] = Counter()
    stance_counts: Counter[str] = Counter()

    for bucket in required_buckets:
        candidate = next((item for item in eligible if item["coverage_bucket"] == bucket and item["profile"].agent_id not in selected_ids), None)
        if candidate is None:
            continue
        candidate["selection_reason_codes"] = list(candidate["selection_reason_codes"]) + ["required_coverage"]
        selected.append(candidate)
        selected_ids.add(candidate["profile"].agent_id)
        bucket_counts[bucket] += 1
        archetype_counts[candidate["profile"].archetype] += 1
        stance_counts[candidate["profile"].stance_bias] += 1

    remaining = [item for item in eligible if item["profile"].agent_id not in selected_ids]
    while len(selected) < count and remaining:
        best = max(
            remaining,
            key=lambda item: _selection_priority(
                item=item,
                bucket_counts=bucket_counts,
                archetype_counts=archetype_counts,
                stance_counts=stance_counts,
                diversity_min_buckets=diversity_min_buckets,
            ),
        )
        reasons = list(best["selection_reason_codes"])
        if bucket_counts.get(best["coverage_bucket"], 0) == 0:
            reasons.append("bucket_diversity")
        elif archetype_counts.get(best["profile"].archetype, 0) == 0:
            reasons.append("archetype_diversity")
        else:
            reasons.append("best_available")
        best["selection_reason_codes"] = reasons
        selected.append(best)
        selected_ids.add(best["profile"].agent_id)
        bucket_counts[best["coverage_bucket"]] += 1
        archetype_counts[best["profile"].archetype] += 1
        stance_counts[best["profile"].stance_bias] += 1
        remaining = [item for item in remaining if item["profile"].agent_id not in selected_ids]

    _assign_execution_tiers(selected, required_buckets=required_buckets, simulation_state=simulation_state)
    return selected


def _build_round_actions(
    round_index: int,
    selection_plan: list[dict],
    seed_posts: list[SwarmAction],
    features: list[SignalFeature],
    items: list[SourceItem],
    llm_client: BaseLLMClient,
    target: str,
    rng: Random,
    config: dict,
    simulation_state: dict[str, object] | None = None,
    environment: SimulationEnvironment | None = None,
) -> tuple[list[SwarmAction], dict]:
    actions = list(seed_posts)
    ranked_features = sorted(features, key=lambda feature: feature.strength, reverse=True)
    evidence_ids = [item.id for item in items[:5]]
    prior_conflict = float((simulation_state or {}).get("latest_conflict_score", 0.0) or 0.0)
    prior_consensus = float((simulation_state or {}).get("latest_consensus_score", 0.0) or 0.0)
    prior_dominant_stance = str((simulation_state or {}).get("dominant_stance", "mixed") or "mixed")
    profiles = [item["profile"] for item in selection_plan]
    plans: list[dict] = []
    for index, selected in enumerate(selection_plan):
        profile = selected["profile"]
        top_feature = ranked_features[index % len(ranked_features)] if ranked_features else None
        if top_feature and profile.stance_bias == top_feature.direction:
            endorse_bias = 0.35 + min(0.2, prior_consensus * 0.1)
            if prior_dominant_stance == profile.stance_bias:
                endorse_bias += 0.1
            action_type = "endorse" if rng.random() < min(0.85, endorse_bias) else "create_post"
        elif top_feature and profile.stance_bias in {"bullish", "bearish"} and top_feature.direction in {"bullish", "bearish"}:
            challenge_bias = 0.65 + min(0.2, prior_conflict * 0.25)
            if prior_dominant_stance not in {"mixed", profile.stance_bias}:
                challenge_bias = min(0.9, challenge_bias + 0.08)
            action_type = "challenge" if rng.random() < challenge_bias else "comment"
        else:
            observe_bias = 0.5 - min(0.2, prior_conflict * 0.1)
            if prior_consensus < 0.45:
                observe_bias -= 0.1
            action_type = "observe" if rng.random() < max(0.2, observe_bias) else "comment"
        direction = top_feature.direction if top_feature else profile.stance_bias
        plans.append(
            {
                "index": index,
                "profile": profile,
                "top_feature": top_feature,
                "action_type": action_type,
                "direction": direction if direction in {"bullish", "bearish"} else "neutral",
                "target_agent_id": profiles[(index - 1) % len(profiles)].agent_id if len(profiles) > 1 and action_type in {"comment", "challenge", "endorse"} else None,
                "strength": round(min(1.0, 0.35 + profile.activity_level * 0.3 + (top_feature.strength * 0.2 if top_feature else 0.0)), 3),
                "referenced_feature_names": [top_feature.name] if top_feature else [],
                "referenced_evidence_ids": list((top_feature.supporting_evidence_ids if top_feature else evidence_ids)[:4]),
                "coverage_bucket": selected["coverage_bucket"],
                "selection_score": selected["selection_score"],
                "selection_reason_codes": list(selected.get("selection_reason_codes") or []),
                "execution_tier": selected["execution_tier"],
            }
        )

    def build_action(plan: dict) -> SwarmAction:
        profile = plan["profile"]
        top_feature = plan["top_feature"]
        content = _render_action_content(
            profile,
            plan["action_type"],
            top_feature,
            clone_llm_client(llm_client),
            target,
            config=config,
            execution_tier=plan["execution_tier"],
            simulation_state=simulation_state,
            graph_context=_format_graph_context(environment, profile, compressed=plan["execution_tier"] != "primary"),
            memory_context=render_memory_context(environment, simulation_state, profile.agent_id) if plan["execution_tier"] == "primary" else _format_memory_context(environment, simulation_state, profile.agent_id),
        )
        return SwarmAction(
            round_index=round_index,
            agent_id=profile.agent_id,
            action_type=plan["action_type"],
            target_agent_id=plan["target_agent_id"],
            content=content,
            direction=plan["direction"],
            strength=plan["strength"],
            referenced_feature_names=plan["referenced_feature_names"],
            referenced_evidence_ids=plan["referenced_evidence_ids"],
            decision_trace={
                "round_index": round_index,
                "agent_id": profile.agent_id,
                "action_type": plan["action_type"],
                "target_agent_id": plan["target_agent_id"],
                "top_feature": top_feature.name if top_feature else None,
                "feature_direction": top_feature.direction if top_feature else None,
                "feature_strength": top_feature.strength if top_feature else None,
                "prior_conflict": prior_conflict,
                "prior_consensus": prior_consensus,
                "prior_dominant_stance": prior_dominant_stance,
                "coverage_bucket": plan["coverage_bucket"],
                "selection_score": plan["selection_score"],
                "execution_tier": plan["execution_tier"],
                "selection_reason_codes": plan["selection_reason_codes"],
            },
        )

    llm_backed_plans = [plan for plan in plans if plan["execution_tier"] in {"primary", "secondary"}]
    templated_plans = [plan for plan in plans if plan["execution_tier"] == "background"]
    if _should_parallelize_round_actions(llm_backed_plans, llm_client, config):
        worker_count = _resolve_parallel_worker_count(config, len(llm_backed_plans))
        with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="swarm-round") as executor:
            round_actions = list(executor.map(build_action, llm_backed_plans))
    else:
        round_actions = [build_action(item) for item in llm_backed_plans]
    round_actions.extend(build_action(item) for item in templated_plans)
    ordered_ids = [plan["profile"].agent_id for plan in plans]
    round_actions.sort(key=lambda action: ordered_ids.index(action.agent_id))

    actions.extend(round_actions)
    execution_diagnostics = {
        "selected_count": len(selection_plan),
        "coverage_buckets": sorted({item["coverage_bucket"] for item in selection_plan}),
        "missing_coverage_buckets": sorted(set(_required_coverage_buckets(simulation_state)) - {item["coverage_bucket"] for item in selection_plan}),
        "tier_counts": dict(Counter(plan["execution_tier"] for plan in plans)),
        "llm_action_count": len(llm_backed_plans),
        "templated_action_count": len(templated_plans),
        "parallel_worker_cap": _resolve_parallel_worker_count(config, len(llm_backed_plans)) if llm_backed_plans else 0,
        "estimated_token_budget": _estimate_round_token_budget(config, plans),
        "batching_enabled": bool(config.get("swarm_enable_batching", False)),
        "plan": [
            {
                "agent_id": plan["profile"].agent_id,
                "coverage_bucket": plan["coverage_bucket"],
                "selection_score": plan["selection_score"],
                "execution_tier": plan["execution_tier"],
                "selection_reason_codes": plan["selection_reason_codes"],
            }
            for plan in plans
        ],
    }
    return actions, execution_diagnostics


def _should_parallelize_round_actions(
    plans: list[dict],
    llm_client: BaseLLMClient,
    config: dict,
) -> bool:
    if not config.get("swarm_parallel_enabled", True):
        return False
    if len(plans) <= 1:
        return False
    if llm_client.diagnostics().get("provider") == "rule_based":
        return False
    return _resolve_parallel_worker_count(config, len(plans)) > 1


def _resolve_parallel_worker_count(config: dict, active_profile_count: int) -> int:
    cap = int(config.get("swarm_parallel_worker_cap", 0) or 0)
    configured = int(config.get("swarm_parallel_workers", 0) or 0)
    if configured > 0:
        resolved = max(1, min(configured, active_profile_count))
        return min(resolved, cap) if cap > 0 else resolved
    default_cap = max(1, int(config.get("swarm_agents_per_round_max", 5) or 5))
    resolved = max(1, min(default_cap, active_profile_count))
    return min(resolved, cap) if cap > 0 else resolved


def _render_action_content(
    profile: AgentProfile,
    action_type: str,
    feature: SignalFeature | None,
    llm_client: BaseLLMClient,
    target: str,
    *,
    config: dict,
    execution_tier: str,
    simulation_state: dict[str, object] | None = None,
    graph_context: str = "",
    memory_context: str = "",
) -> str:
    fallback = f"{profile.name} {action_type.replace('_', ' ')} on {target}"
    context_hint = _format_simulation_state_hint(simulation_state)
    if execution_tier == "background":
        return _render_background_action(profile, action_type, feature, target, context_hint)
    if action_type in {"endorse", "challenge", "observe"}:
        if feature:
            if context_hint:
                return f"{profile.name} {action_type}s the signal {feature.name.replace('_', ' ')} after {context_hint}."
            return f"{profile.name} {action_type}s the signal {feature.name.replace('_', ' ')}."
        return fallback
    prompt = _build_action_prompt(
        profile=profile,
        target=target,
        action_type=action_type,
        feature=feature,
        prompt_style="full" if execution_tier == "primary" else str(config.get("swarm_secondary_prompt_style", "compressed") or "compressed"),
    )
    context_parts = [feature.summary if feature else profile.system_prompt]
    if context_hint:
        context_parts.append(f"Current simulation context: {context_hint}")
    if graph_context:
        context_parts.append(f"Graph context: {graph_context}")
    if memory_context:
        context_parts.append(f"Memory context: {memory_context}")
    if execution_tier == "secondary":
        context_parts = context_parts[:2]
    max_tokens = int(config.get("swarm_primary_max_completion_tokens", 96) or 96) if execution_tier == "primary" else int(config.get("swarm_secondary_max_completion_tokens", 48) or 48)
    llm_client = with_max_completion_tokens(llm_client, max_tokens)
    return llm_client.summarize(prompt, "\n".join(context_parts), fallback)


def _compute_round_statistics(round_result: SwarmRoundResult) -> dict[str, float | int]:
    bullish = int((round_result.stance_histogram or {}).get("bullish", 0) or 0)
    bearish = int((round_result.stance_histogram or {}).get("bearish", 0) or 0)
    neutral = int((round_result.stance_histogram or {}).get("neutral", 0) or 0)
    active_agent_count = max(1, len(round_result.active_agent_ids or []))
    action_count = len(round_result.actions or [])
    directional_total = bullish + bearish
    participation_base = max(active_agent_count, action_count, 1)
    active_participation_rate = min(1.0, action_count / participation_base)
    consensus_score = abs(bullish - bearish) / participation_base
    challenge_count = sum(1 for action in round_result.actions if action.action_type == "challenge")
    challenge_rate = challenge_count / participation_base
    split_balance = (2.0 * min(bullish, bearish) / directional_total) if directional_total else 0.0
    split_rate = split_balance * (directional_total / participation_base if participation_base else 0.0)
    conflict_score = min(1.0, challenge_rate * 0.5 + split_rate * 0.5)
    return {
        "bullish": bullish,
        "bearish": bearish,
        "neutral": neutral,
        "active_agent_count": active_agent_count,
        "action_count": action_count,
        "directional_total": directional_total,
        "challenge_count": challenge_count,
        "challenge_rate": challenge_rate,
        "active_participation_rate": active_participation_rate,
        "consensus_score": consensus_score,
        "conflict_score": conflict_score,
        "bullish_share": bullish / participation_base,
        "bearish_share": bearish / participation_base,
        "neutral_share": neutral / participation_base,
        "dominant_share": max(bullish, bearish, neutral) / participation_base,
    }


def _summarize_round(
    round_index: int,
    profiles: list[AgentProfile],
    actions: list[SwarmAction],
    selection_diagnostics: dict,
) -> SwarmRoundResult:
    stance_histogram = {"bullish": 0, "bearish": 0, "neutral": 0}
    for action in actions:
        stance_histogram[action.direction if action.direction in stance_histogram else "neutral"] += 1
    stats = _compute_round_statistics(
        SwarmRoundResult(
            round_index=round_index,
            active_agent_ids=[profile.agent_id for profile in profiles],
            actions=actions,
            summary="",
            stance_histogram=stance_histogram,
            consensus_score=0.0,
            conflict_score=0.0,
            selection_diagnostics=selection_diagnostics,
        )
    )
    active_names = ", ".join(profile.name for profile in profiles[:4]) or "none"
    summary = (
        f"Round {round_index + 1} activated {len(profiles)} agents ({active_names}) "
        f"with normalized consensus {float(stats['consensus_score']):.2f} and conflict {float(stats['conflict_score']):.2f}."
    )
    return SwarmRoundResult(
        round_index=round_index,
        active_agent_ids=[profile.agent_id for profile in profiles],
        actions=actions,
        summary=summary,
        stance_histogram=stance_histogram,
        consensus_score=round(float(stats["consensus_score"]), 3),
        conflict_score=round(float(stats["conflict_score"]), 3),
        selection_diagnostics=selection_diagnostics,
    )


def _required_coverage_buckets(simulation_state: dict[str, object] | None) -> list[str]:
    configured = list((simulation_state or {}).get("required_coverage_buckets") or ["macro", "market", "sentiment", "policy"])
    return [str(bucket) for bucket in configured]


def _selection_priority(
    *,
    item: dict,
    bucket_counts: Counter[str],
    archetype_counts: Counter[str],
    stance_counts: Counter[str],
    diversity_min_buckets: int,
) -> tuple[float, float]:
    score = float(item["selection_score"])
    bucket = str(item["coverage_bucket"])
    profile = item["profile"]
    bucket_bonus = 0.2 if len(bucket_counts) < diversity_min_buckets and bucket_counts.get(bucket, 0) == 0 else 0.0
    bucket_penalty = bucket_counts.get(bucket, 0) * 0.08
    archetype_penalty = archetype_counts.get(profile.archetype, 0) * 0.05
    stance_penalty = stance_counts.get(profile.stance_bias, 0) * 0.02
    return (score + bucket_bonus - bucket_penalty - archetype_penalty - stance_penalty, profile.influence_weight)


def _assign_execution_tiers(selected: list[dict], *, required_buckets: list[str], simulation_state: dict[str, object] | None) -> None:
    primary_count = int((simulation_state or {}).get("primary_agent_count", max(len(required_buckets), 4)) or max(len(required_buckets), 4))
    secondary_count = int((simulation_state or {}).get("secondary_agent_count", 4) or 4)
    ranked = sorted(
        selected,
        key=lambda item: (
            1 if item["coverage_bucket"] in required_buckets else 0,
            item["selection_score"],
            item["profile"].influence_weight,
        ),
        reverse=True,
    )
    primary_ids = {item["profile"].agent_id for item in ranked[: min(primary_count, len(ranked))]}
    secondary_ids = {
        item["profile"].agent_id
        for item in ranked[min(primary_count, len(ranked)): min(primary_count + secondary_count, len(ranked))]
    }
    for item in selected:
        agent_id = item["profile"].agent_id
        if agent_id in primary_ids:
            item["execution_tier"] = "primary"
        elif agent_id in secondary_ids:
            item["execution_tier"] = "secondary"
        else:
            item["execution_tier"] = "background"


def _stable_fraction(seed: int, round_index: int, agent_id: str, salt: str) -> float:
    digest = hashlib.sha256(f"{seed}:{round_index}:{agent_id}:{salt}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _stable_int(seed: int, round_index: int, salt: str, low: int, high: int) -> int:
    if high <= low:
        return low
    digest = hashlib.sha256(f"{seed}:{round_index}:{salt}".encode("utf-8")).hexdigest()
    return low + (int(digest[:8], 16) % (high - low + 1))


def _build_action_prompt(*, profile: AgentProfile, target: str, action_type: str, feature: SignalFeature | None, prompt_style: str) -> str:
    if prompt_style == "compressed":
        return f"{profile.name}: {action_type} for {target}. Signal={feature.name if feature else 'none'}. Keep it to one short evidence-led sentence."
    return f"You are {profile.name}. Produce one short market discussion action for {target}. Action={action_type}. Signal={feature.name if feature else 'none'}."


def _render_background_action(profile: AgentProfile, action_type: str, feature: SignalFeature | None, target: str, context_hint: str) -> str:
    signal = feature.name.replace("_", " ") if feature else target
    if context_hint:
        return f"{profile.name} {action_type.replace('_', ' ')} on {signal} after {context_hint}."
    return f"{profile.name} {action_type.replace('_', ' ')} on {signal}."


def _estimate_round_token_budget(config: dict, plans: list[dict]) -> int:
    primary_budget = int(config.get("swarm_primary_max_completion_tokens", 96) or 96)
    secondary_budget = int(config.get("swarm_secondary_max_completion_tokens", 48) or 48)
    return sum(primary_budget if plan["execution_tier"] == "primary" else secondary_budget if plan["execution_tier"] == "secondary" else 0 for plan in plans)


def _memory_config_from_config(config: dict[str, Any]) -> dict[str, int]:
    return {
        "swarm_memory_agent_action_cap": int(config.get("swarm_memory_agent_action_cap", 3) or 3),
        "swarm_memory_shared_round_cap": int(config.get("swarm_memory_shared_round_cap", 4) or 4),
        "swarm_shared_reference_cap": int(config.get("swarm_shared_reference_cap", 24) or 24),
        "swarm_memory_community_reference_cap": int(config.get("swarm_memory_community_reference_cap", 16) or 16),
        "swarm_prompt_shared_round_cap": int(config.get("swarm_prompt_shared_round_cap", 3) or 3),
        "swarm_prompt_agent_action_cap": int(config.get("swarm_prompt_agent_action_cap", 2) or 2),
    }


def _make_initial_simulation_state(*, seed_posts: list[SwarmAction], environment: SimulationEnvironment, config: dict) -> dict[str, object]:
    memory_state = dict(environment.memory_snapshot or {})
    memory_state["config"] = _memory_config_from_config(config)
    return normalize_simulation_state(
        {
        "round_count": 0,
        "latest_round_index": -1,
        "latest_summary": "",
        "latest_consensus_score": 0.0,
        "latest_conflict_score": 0.0,
        "dominant_stance": "mixed",
        "active_agent_history": [],
        "selection_seed": int(config.get("swarm_random_seed", 42) or 42),
        "required_coverage_buckets": list(config.get("swarm_required_coverage_buckets", ["macro", "market", "sentiment", "policy"])),
        "selection_diversity_min_buckets": int(config.get("swarm_diversity_min_buckets_per_round", 4) or 4),
        "primary_agent_count": int(config.get("swarm_primary_agent_count", 4) or 4),
        "secondary_agent_count": int(config.get("swarm_secondary_agent_count", 4) or 4),
        "cumulative_action_counts": {"bullish": 0, "bearish": 0, "neutral": 0},
        "seed_post_count": len(seed_posts),
        "memory": memory_state,
        "social_edges": [edge.to_dict() for edge in environment.social_edges[:50]],
        "world_state": {
            "target": environment.target,
            "prediction_date": environment.prediction_date,
            "current_round": 0,
            "total_rounds": environment.time_config.total_rounds,
            "dominant_stance": "mixed",
            "latest_summary": "",
        },
        "agent_state": {
            "profiles": [profile.to_dict() for profile in environment.profiles],
            "active_agent_ids": [],
            "decision_traces": {},
            "social_dynamics": {"top_conflicts": [], "top_influence_edges": []},
        },
        "memory_state": memory_state,
        "event_history": [
            {
                "event": "simulation_initialized",
                "timestamp": iso_utc_now(),
                "seed_post_count": len(seed_posts),
                "profile_count": len(environment.profiles),
            }
        ],
        "graph_deltas": [],
        }
    )


def _advance_simulation_state(
    simulation_state: dict[str, object],
    round_result: SwarmRoundResult,
    *,
    config: dict[str, Any],
) -> dict[str, object]:
    cumulative_action_counts = {
        "bullish": int(((simulation_state.get("cumulative_action_counts") or {}).get("bullish", 0))),
        "bearish": int(((simulation_state.get("cumulative_action_counts") or {}).get("bearish", 0))),
        "neutral": int(((simulation_state.get("cumulative_action_counts") or {}).get("neutral", 0))),
    }
    for action in round_result.actions:
        direction = action.direction if action.direction in cumulative_action_counts else "neutral"
        cumulative_action_counts[direction] += 1

    prior_consensus = float(simulation_state.get("latest_consensus_score", 0.0) or 0.0)
    prior_conflict = float(simulation_state.get("latest_conflict_score", 0.0) or 0.0)
    decision_trace_cap = int(config.get("swarm_decision_trace_cap", 4) or 4)
    event_history_cap = int(config.get("swarm_event_history_cap", 12) or 12)
    graph_delta_cap = int(config.get("swarm_graph_delta_cap", 12) or 12)
    active_history_cap = int(config.get("swarm_active_history_cap", 10) or 10)
    bullish = cumulative_action_counts["bullish"]
    bearish = cumulative_action_counts["bearish"]
    if bullish > bearish:
        dominant_stance = "bullish"
    elif bearish > bullish:
        dominant_stance = "bearish"
    else:
        dominant_stance = "mixed"

    active_history = list(simulation_state.get("active_agent_history") or [])
    active_history.append(
        {
            "round_index": round_result.round_index,
            "active_agent_ids": list(round_result.active_agent_ids),
            "coverage_buckets": list((round_result.selection_diagnostics or {}).get("coverage_buckets", [])),
            "summary": round_result.summary,
            "consensus_score": round_result.consensus_score,
            "conflict_score": round_result.conflict_score,
            "action_count": len(round_result.actions),
        }
    )
    active_history = active_history[-active_history_cap:]
    memory_state = evolve_memory_snapshot(
        memory_state=dict(simulation_state.get("memory_state") or simulation_state.get("memory") or {}),
        round_result=round_result,
        profiles=[
            AgentProfile(**profile)
            for profile in ((simulation_state.get("agent_state") or {}).get("profiles") or [])
            if isinstance(profile, dict)
        ],
    )

    decision_traces = dict((simulation_state.get("agent_state") or {}).get("decision_traces") or {})
    for action in round_result.actions:
        traces = list(decision_traces.get(action.agent_id) or [])
        traces.append(action.decision_trace)
        decision_traces[action.agent_id] = traces[-decision_trace_cap:]

    social_edges = list(simulation_state.get("social_edges") or [])
    sorted_conflicts = sorted(social_edges, key=lambda item: float(item.get("conflict_score", 0.0) or 0.0), reverse=True)[:5]
    sorted_influence = sorted(social_edges, key=lambda item: float(item.get("influence_score", 0.0) or 0.0), reverse=True)[:5]
    dominant_communities = [
        {
            "community_id": community.get("community_id"),
            "agent_count": len(community.get("agent_ids") or []),
        }
        for community in list(memory_state.get("communities") or [])[:5]
    ]

    graph_deltas = list(simulation_state.get("graph_deltas") or [])
    graph_deltas.append(
        normalize_graph_delta(
            {
                "round_index": round_result.round_index,
                "timestamp": iso_utc_now(),
                "new_action_nodes": len(round_result.actions),
                "created_action_nodes": [
                    {
                        "action_node_id": f"sim-round-{round_result.round_index}-action-{index}-{action.agent_id}",
                        "agent_id": action.agent_id,
                        "action_type": action.action_type,
                        "direction": action.direction,
                    }
                    for index, action in enumerate(round_result.actions)
                ],
                "referenced_features": sorted(
                    {
                        feature_name
                        for action in round_result.actions
                        for feature_name in action.referenced_feature_names
                    }
                ),
                "referenced_entities": sorted(
                    {
                        evidence_id
                        for action in round_result.actions
                        for evidence_id in action.referenced_evidence_ids
                    }
                ),
                "social_dynamics": {
                    "top_conflicts": sorted_conflicts,
                    "top_influence_edges": sorted_influence,
                    "dominant_communities": dominant_communities,
                },
                "consensus_shift": {
                    "consensus_score": round_result.consensus_score,
                    "conflict_score": round_result.conflict_score,
                    "consensus_delta": round_result.consensus_score - prior_consensus,
                    "conflict_delta": round_result.conflict_score - prior_conflict,
                },
            }
        )
    )
    graph_deltas = graph_deltas[-graph_delta_cap:]

    event_history = list(simulation_state.get("event_history") or [])
    event_history.append(
        {
            "event": "round_completed",
            "timestamp": iso_utc_now(),
            "round_index": round_result.round_index,
            "summary": round_result.summary,
            "consensus_score": round_result.consensus_score,
            "conflict_score": round_result.conflict_score,
            "active_agent_ids": list(round_result.active_agent_ids),
        }
    )
    event_history = event_history[-event_history_cap:]

    return normalize_simulation_state(
        {
            "round_count": int(simulation_state.get("round_count", 0)) + 1,
            "latest_round_index": round_result.round_index,
            "latest_summary": round_result.summary,
            "latest_consensus_score": round_result.consensus_score,
            "latest_conflict_score": round_result.conflict_score,
            "dominant_stance": dominant_stance,
            "active_agent_history": active_history,
            "selection_seed": simulation_state.get("selection_seed", 42),
            "required_coverage_buckets": list(simulation_state.get("required_coverage_buckets") or ["macro", "market", "sentiment", "policy"]),
            "selection_diversity_min_buckets": int(simulation_state.get("selection_diversity_min_buckets", 4) or 4),
            "primary_agent_count": int(simulation_state.get("primary_agent_count", 4) or 4),
            "secondary_agent_count": int(simulation_state.get("secondary_agent_count", 4) or 4),
            "cumulative_action_counts": cumulative_action_counts,
            "seed_post_count": simulation_state.get("seed_post_count", 0),
            "memory": memory_state,
            "social_edges": social_edges,
            "world_state": {
                **dict(simulation_state.get("world_state") or {}),
                "current_round": round_result.round_index + 1,
                "dominant_stance": dominant_stance,
                "latest_summary": round_result.summary,
            },
            "agent_state": {
                **dict(simulation_state.get("agent_state") or {}),
                "active_agent_ids": list(round_result.active_agent_ids),
                "decision_traces": decision_traces,
                "social_dynamics": {
                    "top_conflicts": sorted_conflicts,
                    "top_influence_edges": sorted_influence,
                    "dominant_communities": dominant_communities,
                },
            },
            "memory_state": memory_state,
            "event_history": event_history,
            "graph_deltas": graph_deltas,
        }
    )


def _format_simulation_state_hint(simulation_state: dict[str, object] | None) -> str:
    if not simulation_state:
        return ""
    latest_round_index = int(simulation_state.get("latest_round_index", -1) or -1)
    if latest_round_index < 0:
        return ""
    dominant_stance = str(simulation_state.get("dominant_stance", "mixed") or "mixed")
    consensus_score = float(simulation_state.get("latest_consensus_score", 0.0) or 0.0)
    conflict_score = float(simulation_state.get("latest_conflict_score", 0.0) or 0.0)
    return (
        f"round={latest_round_index + 1}, stance={dominant_stance}, "
        f"consensus={consensus_score:.2f}, conflict={conflict_score:.2f}"
    )


def _build_swarm_priors(rounds: list[SwarmRoundResult], config: dict) -> dict[str, float]:
    if not rounds:
        return {"swarm_up_bias": 0.0, "swarm_down_bias": 0.0, "swarm_confidence_modifier": 0.0}
    max_influence = float(config.get("swarm_max_influence", 0.12) or 0.12)
    round_stats = [_compute_round_statistics(round_result) for round_result in rounds]
    average_bullish_share = sum(float(item["bullish_share"]) for item in round_stats) / len(round_stats)
    average_bearish_share = sum(float(item["bearish_share"]) for item in round_stats) / len(round_stats)
    raw_bias = average_bullish_share - average_bearish_share
    up_bias = max(0.0, min(max_influence, raw_bias * max_influence))
    down_bias = max(0.0, min(max_influence, -raw_bias * max_influence))
    average_consensus = sum(float(item["consensus_score"]) for item in round_stats) / len(round_stats)
    average_conflict = sum(float(item["conflict_score"]) for item in round_stats) / len(round_stats)
    confidence_modifier = max(-max_influence, min(max_influence, (average_consensus - average_conflict) * max_influence))
    return {
        "swarm_up_bias": round(up_bias, 4),
        "swarm_down_bias": round(down_bias, 4),
        "swarm_confidence_modifier": round(confidence_modifier, 4),
    }


def _build_summary_metrics(
    profiles: list[AgentProfile],
    seed_posts: list[SwarmAction],
    rounds: list[SwarmRoundResult],
    derived_features: list[SignalFeature],
) -> dict[str, float | int | str]:
    round_stats = [_compute_round_statistics(round_result) for round_result in rounds]
    average_bullish_share = sum(float(item["bullish_share"]) for item in round_stats) / max(len(round_stats), 1)
    average_bearish_share = sum(float(item["bearish_share"]) for item in round_stats) / max(len(round_stats), 1)
    consensus = sum(float(item["consensus_score"]) for item in round_stats) / max(len(round_stats), 1)
    conflict = sum(float(item["conflict_score"]) for item in round_stats) / max(len(round_stats), 1)
    active_agent_mean = sum(int(item["active_agent_count"]) for item in round_stats) / max(len(round_stats), 1) if round_stats else 0.0
    if average_bullish_share > average_bearish_share:
        dominant = "bullish"
    elif average_bearish_share > average_bullish_share:
        dominant = "bearish"
    else:
        dominant = "mixed"
    return {
        "agent_count": len(profiles),
        "seed_post_count": len(seed_posts),
        "round_count": len(rounds),
        "dominant_stance": dominant,
        "average_active_agents": round(active_agent_mean, 2),
        "average_consensus": round(consensus, 4),
        "average_conflict": round(conflict, 4),
        "derived_feature_count": len(derived_features),
    }

def _build_memory_diagnostics(
    *,
    environment: SimulationEnvironment,
    rounds: list[SwarmRoundResult],
) -> dict:
    agent_state: dict[str, dict] = {}
    for profile in environment.profiles:
        agent_state[profile.agent_id] = {
            "agent_id": profile.agent_id,
            "name": profile.name,
            "stance_bias": profile.stance_bias,
            "seed_evidence_ids": list(profile.seed_evidence_ids),
            "active_rounds": [],
            "recent_actions": [],
            "referenced_evidence_ids": set(profile.seed_evidence_ids),
            "referenced_feature_names": set(),
            "last_simulated_hour": None,
        }

    episodes: list[dict] = []
    total_actions = 0
    for round_result in rounds:
        simulated_hour = _simulated_hour_for_round(environment.time_config, round_result.round_index)
        episodes.append(
            {
                "round_index": round_result.round_index,
                "simulated_hour": simulated_hour,
                "summary": round_result.summary,
                "active_agent_ids": list(round_result.active_agent_ids),
                "action_count": len(round_result.actions),
                "consensus_score": round_result.consensus_score,
                "conflict_score": round_result.conflict_score,
            }
        )
        for action in round_result.actions:
            total_actions += 1
            state = agent_state.setdefault(
                action.agent_id,
                {
                    "agent_id": action.agent_id,
                    "name": action.agent_id,
                    "stance_bias": "neutral",
                    "seed_evidence_ids": [],
                    "active_rounds": [],
                    "recent_actions": [],
                    "referenced_evidence_ids": set(),
                    "referenced_feature_names": set(),
                    "last_simulated_hour": None,
                },
            )
            if round_result.round_index not in state["active_rounds"]:
                state["active_rounds"].append(round_result.round_index)
            state["recent_actions"].append(
                {
                    "round_index": round_result.round_index,
                    "simulated_hour": simulated_hour,
                    "action_type": action.action_type,
                    "direction": action.direction,
                    "content": action.content,
                }
            )
            state["recent_actions"] = state["recent_actions"][-5:]
            state["referenced_evidence_ids"].update(action.referenced_evidence_ids)
            state["referenced_feature_names"].update(action.referenced_feature_names)
            state["last_simulated_hour"] = simulated_hour

    individual = []
    for state in agent_state.values():
        individual.append(
            {
                "agent_id": state["agent_id"],
                "name": state["name"],
                "stance_bias": state["stance_bias"],
                "seed_evidence_ids": list(state["seed_evidence_ids"]),
                "active_rounds": list(state["active_rounds"]),
                "last_simulated_hour": state["last_simulated_hour"],
                "recent_actions": list(state["recent_actions"]),
                "referenced_evidence_ids": sorted(state["referenced_evidence_ids"]),
                "referenced_feature_names": sorted(state["referenced_feature_names"]),
            }
        )
    individual.sort(key=lambda item: item["agent_id"])

    shared_feature_names = sorted(
        {
            feature_name
            for state in individual
            for feature_name in state["referenced_feature_names"]
        }
    )
    shared_evidence_ids = sorted(
        {
            evidence_id
            for state in individual
            for evidence_id in state["referenced_evidence_ids"]
        }
    )
    return {
        "individual_agent_count": len(individual),
        "individual": individual,
        "shared": {
            "episode_count": len(episodes),
            "total_action_count": total_actions,
            "round_summaries": [episode["summary"] for episode in episodes],
            "shared_feature_names": shared_feature_names,
            "shared_evidence_ids": shared_evidence_ids,
        },
        "communities": _build_community_memory(environment, individual),
        "episodic": episodes,
        "time_awareness": {
            "round_hours": [episode["simulated_hour"] for episode in episodes],
            "minutes_per_round": environment.time_config.minutes_per_round,
            "start_hour": environment.time_config.start_hour,
            "decay_half_life_rounds": 3,
        },
    }


def _build_stateful_runtime_diagnostics(simulation_state: dict[str, object]) -> dict[str, object]:
    return {
        "round_count": int(simulation_state.get("round_count", 0) or 0),
        "event_count": len(simulation_state.get("event_history") or []),
        "graph_delta_count": len(simulation_state.get("graph_deltas") or []),
        "decision_trace_agent_count": len(((simulation_state.get("agent_state") or {}).get("decision_traces") or {})),
        "world_state": dict(simulation_state.get("world_state") or {}),
    }


def _build_community_memory(environment: SimulationEnvironment, individual: list[dict]) -> list[dict]:
    community_map: dict[str, dict] = {}
    for profile in environment.profiles:
        for category in profile.focus_categories:
            state = community_map.setdefault(
                category,
                {
                    "community_id": category,
                    "agent_ids": [],
                    "shared_feature_names": set(),
                    "shared_evidence_ids": set(),
                },
            )
            state["agent_ids"].append(profile.agent_id)
    for agent_memory in individual:
        profile = next((item for item in environment.profiles if item.agent_id == agent_memory["agent_id"]), None)
        if profile is None:
            continue
        for category in profile.focus_categories:
            community_map[category]["shared_feature_names"].update(agent_memory["referenced_feature_names"])
            community_map[category]["shared_evidence_ids"].update(agent_memory["referenced_evidence_ids"])
    communities = []
    for state in sorted(community_map.values(), key=lambda item: item["community_id"]):
        communities.append(
            {
                "community_id": state["community_id"],
                "agent_ids": sorted(state["agent_ids"]),
                "shared_feature_names": sorted(state["shared_feature_names"]),
                "shared_evidence_ids": sorted(state["shared_evidence_ids"]),
            }
        )
    return communities


def _make_memory_snapshot(*, profiles: list[AgentProfile], social_edges: list[SocialEdge], prior_memory: dict | None) -> dict:
    prior_memory = prior_memory or {}
    return {
        "individual": [
            {
                "agent_id": profile.agent_id,
                "seed_evidence_ids": list(profile.seed_evidence_ids),
                "recent_actions": [],
            }
            for profile in profiles
        ],
        "shared": {
            "episode_count": int((prior_memory.get("shared") or {}).get("episode_count", 0) or 0),
            "round_summaries": list((prior_memory.get("shared") or {}).get("round_summaries", []))[-10:],
        },
        "communities": list((prior_memory.get("communities") or [])),
        "social_edges": [edge.to_dict() for edge in social_edges[:100]],
    }


def _format_graph_context(environment: SimulationEnvironment | None, profile: AgentProfile, *, compressed: bool = False) -> str:
    if environment is None:
        return ""
    graph_entity_count = int(environment.diagnostics.get("graph_entity_count", 0) or 0)
    if graph_entity_count <= 0:
        return ""
    if compressed:
        return f"graph entities={graph_entity_count}, bucket={profile.coverage_bucket}"
    return (
        f"graph entities={graph_entity_count}, profile_focus={', '.join(profile.focus_categories) or 'none'}, "
        f"social_edges={len(environment.social_edges)}"
    )


def _format_memory_context(environment: SimulationEnvironment | None, simulation_state: dict[str, object] | None, agent_id: str) -> str:
    if environment is None:
        return ""
    memory = ((simulation_state or {}).get("memory") or environment.memory_snapshot or {})
    agent_entries = memory.get("individual") or []
    agent_entry = next((entry for entry in agent_entries if entry.get("agent_id") == agent_id), None)
    shared = memory.get("shared") or {}
    recent_actions = len((agent_entry or {}).get("recent_actions") or [])
    episode_count = int(shared.get("episode_count", 0) or 0)
    if recent_actions <= 0 and episode_count <= 0:
        return ""
    return (
        f"recent_actions={recent_actions}, shared_episodes={episode_count}, "
        f"last_action={str((agent_entry or {}).get('last_action_type') or 'none')}, "
        f"last_direction={str((agent_entry or {}).get('last_direction') or 'neutral')}"
    )


def _resolve_coverage_bucket(*, archetype: str, focus_categories: list[str], entity_type: str) -> str:
    lowered_archetype = archetype.lower()
    normalized = {category.strip().lower() for category in focus_categories if category}
    if "economic" in normalized or "macro" in lowered_archetype:
        return "macro"
    if "political" in normalized or "policy" in lowered_archetype or "geopolitical" in lowered_archetype:
        return "policy"
    if "social" in normalized or "sentiment" in lowered_archetype or "retail" in lowered_archetype:
        return "sentiment"
    if "market" in normalized or "sector" in lowered_archetype or entity_type == "community":
        return "market"
    return "market"


def _build_username(name: str, index: int) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")
    slug = "_".join(filter(None, slug.split("_")))
    return f"{slug or 'agent'}_{index + 1}"


def _match_seed_profile(
    category: str,
    profiles: list[AgentProfile],
    activity_configs: dict[str, AgentBehaviorConfig],
) -> AgentProfile:
    for profile in profiles:
        if category in profile.focus_categories:
            return profile
    return max(
        profiles,
        key=lambda profile: (
            activity_configs.get(profile.agent_id).influence_weight if profile.agent_id in activity_configs else 0.0,
            profile.activity_level,
        ),
    )


def _simulated_hour_for_round(time_config: TimeSimulationConfig, round_index: int) -> int:
    total_minutes = time_config.start_hour * 60 + round_index * time_config.minutes_per_round
    return (total_minutes // 60) % 24
