import pytest

from market_direction_dashboard.config import load_config
from market_direction_dashboard.llm_clients import RuleBasedLLMClient
from market_direction_dashboard.models import SignalFeature, SourceItem
from market_direction_dashboard.swarm_simulation import (
    SwarmAction,
    _advance_simulation_state,
    _make_initial_simulation_state,
    _resolve_parallel_worker_count,
    _summarize_round,
    derive_swarm_features,
    prepare_swarm_environment,
    run_swarm_from_environment,
)


def _sample_item() -> SourceItem:
    return SourceItem(
        id="doc-1",
        title="Sample market note",
        source="Reuters",
        category="market",
        published_at="2026-03-18",
        url="https://example.com/doc-1",
        summary="Breadth improved across major indices.",
        impact="medium",
        impact_score=0.5,
        direction="bullish",
        freshness_score=0.9,
        credibility_score=0.9,
        quality_score=0.9,
    )


def _sample_feature() -> SignalFeature:
    return SignalFeature(
        name="breadth_risk_on",
        direction="bullish",
        strength=0.7,
        category="market",
        summary="Breadth improved across cyclicals.",
        supporting_evidence_ids=["doc-1"],
    )


def test_load_config_applies_new_swarm_defaults() -> None:
    config = load_config()

    assert config["swarm_persona_count"] == 50
    assert config["swarm_dynamic_agent_count"] == 12
    assert config["swarm_rounds"] == 10
    assert config["swarm_agents_per_round_min"] == 10
    assert config["swarm_agents_per_round_max"] == 16


def test_load_config_validates_swarm_bounds() -> None:
    with pytest.raises(ValueError, match="swarm_agents_per_round_min"):
        load_config({"swarm_agents_per_round_min": 6, "swarm_agents_per_round_max": 5})

    with pytest.raises(ValueError, match="swarm_persona_count"):
        load_config({"swarm_persona_count": 4, "swarm_agents_per_round_min": 3, "swarm_agents_per_round_max": 5})

    with pytest.raises(ValueError, match="swarm_rounds"):
        load_config({"swarm_rounds": 0})


def test_prepare_swarm_environment_uses_configured_counts_and_rounds() -> None:
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_persona_count": 12,
            "swarm_dynamic_agent_count": 0,
            "swarm_rounds": 4,
            "swarm_agents_per_round_min": 3,
            "swarm_agents_per_round_max": 5,
        }
    )

    environment = prepare_swarm_environment(
        items=[_sample_item()],
        snapshot={},
        features=[_sample_feature()],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    assert environment.time_config.total_rounds == 4
    assert environment.time_config.min_active_agents == 3
    assert environment.time_config.max_active_agents == 5
    assert len(environment.profiles) == 12
    assert all(profile.active_rounds == [0, 1, 2, 3] for profile in environment.profiles)
    assert len({profile.name for profile in environment.profiles}) == 12


def test_swarm_simulation_runs_with_legacy_sized_config() -> None:
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_persona_count": 8,
            "swarm_dynamic_agent_count": 3,
            "swarm_rounds": 3,
            "swarm_agents_per_round_min": 3,
            "swarm_agents_per_round_max": 5,
        }
    )
    item = _sample_item()
    feature = _sample_feature()
    environment = prepare_swarm_environment(
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    result = run_swarm_from_environment(
        environment=environment,
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    assert len(result.rounds) == 3
    assert all(3 <= len(round_result.active_agent_ids) <= 5 for round_result in result.rounds)


def test_swarm_simulation_runs_with_new_default_config() -> None:
    config = load_config({"llm_provider": "rule_based"})
    item = _sample_item()
    feature = _sample_feature()
    environment = prepare_swarm_environment(
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    result = run_swarm_from_environment(
        environment=environment,
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    assert environment.time_config.total_rounds == 10
    assert len(environment.profiles) == 50
    assert len({profile.agent_id for profile in environment.profiles}) == len(environment.profiles)
    assert len(result.rounds) == 10
    assert all(10 <= len(round_result.active_agent_ids) <= 16 for round_result in result.rounds)
    assert result.diagnostics["runtime_seconds"] > 0.0


def test_parallel_worker_count_defaults_to_configured_max_active_agents() -> None:
    config = load_config({"swarm_agents_per_round_max": 16, "swarm_parallel_worker_cap": 16})

    assert _resolve_parallel_worker_count(config, 30) == 16
    assert _resolve_parallel_worker_count(config, 6) == 6


def test_simulation_state_and_memory_are_bounded_across_many_rounds() -> None:
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_rounds": 10,
            "swarm_persona_count": 12,
            "swarm_dynamic_agent_count": 0,
            "swarm_agents_per_round_min": 4,
            "swarm_agents_per_round_max": 4,
            "swarm_memory_agent_action_cap": 2,
            "swarm_memory_shared_round_cap": 3,
            "swarm_shared_reference_cap": 5,
            "swarm_decision_trace_cap": 2,
            "swarm_event_history_cap": 4,
            "swarm_graph_delta_cap": 4,
            "swarm_active_history_cap": 4,
        }
    )
    item = _sample_item()
    feature = _sample_feature()
    environment = prepare_swarm_environment(
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    simulation_state = _make_initial_simulation_state(seed_posts=[], environment=environment, config=config)
    profiles = environment.profiles[:4]
    for round_index in range(8):
        actions = [
            SwarmAction(
                round_index=round_index,
                agent_id=profile.agent_id,
                action_type="comment" if index % 2 == 0 else "challenge",
                target_agent_id=None,
                content=f"raw-content-{round_index}-{index}",
                direction="bullish" if index % 2 == 0 else "bearish",
                strength=0.5,
                referenced_feature_names=[f"feature_{round_index}_{index}"],
                referenced_evidence_ids=[f"evidence_{round_index}_{index}"],
                decision_trace={"round_index": round_index, "agent_id": profile.agent_id},
            )
            for index, profile in enumerate(profiles)
        ]
        round_result = _summarize_round(round_index, profiles, actions, {"coverage_buckets": ["market"]})
        simulation_state = _advance_simulation_state(simulation_state, round_result, config=config)

    memory_state = simulation_state["memory_state"]
    assert len(simulation_state["event_history"]) == 4
    assert len(simulation_state["graph_deltas"]) == 4
    assert len(simulation_state["active_agent_history"]) == 4
    for traces in (simulation_state.get("agent_state") or {}).get("decision_traces", {}).values():
        assert len(traces) <= 2
    assert len((memory_state.get("shared") or {}).get("round_summaries", [])) == 3
    assert len((memory_state.get("shared") or {}).get("shared_feature_names", [])) <= 5
    assert len((memory_state.get("shared") or {}).get("shared_evidence_ids", [])) <= 5
    for entry in memory_state.get("individual", []):
        assert len(entry.get("recent_actions", [])) <= 2
        if entry.get("recent_actions"):
            assert "content" not in entry["recent_actions"][-1]


def test_normalized_derived_features_are_consistent_across_swarm_sizes() -> None:
    config = load_config({"llm_provider": "rule_based"})
    items = [_sample_item()]
    features = [_sample_feature()]
    profiles_small = _make_profiles(10)
    profiles_large = _make_profiles(50)
    rounds_small = _make_consistent_rounds(agent_count=10)
    rounds_large = _make_consistent_rounds(agent_count=50)

    derived_small = {feature.name: feature for feature in derive_swarm_features(rounds_small, profiles_small, items, features, config)}
    derived_large = {feature.name: feature for feature in derive_swarm_features(rounds_large, profiles_large, items, features, config)}

    assert derived_small.keys() == derived_large.keys()
    for name in derived_small:
        assert abs(derived_small[name].strength - derived_large[name].strength) < 0.02


def _make_profiles(count: int):
    return [
        type(
            "Profile",
            (),
            {"agent_id": f"agent-{index}", "archetype": "stable_market", "name": f"Agent {index}"},
        )()
        for index in range(count)
    ]


def _make_consistent_rounds(*, agent_count: int):
    rounds = []
    profiles = _make_profiles(agent_count)
    for round_index in range(3):
        actions = []
        bullish_cutoff = int(agent_count * 0.6)
        for index, profile in enumerate(profiles):
            direction = "bullish" if index < bullish_cutoff else "bearish"
            action_type = "challenge" if index % 5 == 0 else "comment"
            feature_names = ["breadth_risk_on"]
            if index % 4 == 0:
                feature_names.append("narrative_alignment")
            actions.append(
                SwarmAction(
                    round_index=round_index,
                    agent_id=profile.agent_id,
                    action_type=action_type,
                    target_agent_id=None,
                    content=f"action-{round_index}-{index}",
                    direction=direction,
                    strength=0.5,
                    referenced_feature_names=feature_names,
                    referenced_evidence_ids=[f"doc-{index % 3}"],
                    decision_trace={},
                )
            )
        rounds.append(_summarize_round(round_index, profiles, actions, {"coverage_buckets": ["market", "sentiment"]}))
    return rounds
