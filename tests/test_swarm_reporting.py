from __future__ import annotations

from market_direction_dashboard.config import load_config
from market_direction_dashboard.llm_clients import RuleBasedLLMClient
from market_direction_dashboard.models import SignalFeature, SourceItem
from market_direction_dashboard.simulation_reporting import build_swarm_reporting_payload
from market_direction_dashboard.swarm_simulation import prepare_swarm_environment, run_swarm_from_environment


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


def test_build_swarm_reporting_payload_summarizes_large_swarm_readably() -> None:
    profiles = [
        {
            "agent_id": "a1",
            "name": "Macro Hawk",
            "coverage_bucket": "macro",
            "focus_categories": ["economic"],
            "entity_type": "persona",
            "archetype": "macro_hawk",
            "stance_bias": "bearish",
        },
        {
            "agent_id": "a2",
            "name": "Tape Watch",
            "coverage_bucket": "market",
            "focus_categories": ["market"],
            "entity_type": "persona",
            "archetype": "market_technician",
            "stance_bias": "bullish",
        },
        {
            "agent_id": "a3",
            "name": "Retail Pulse",
            "coverage_bucket": "market",
            "focus_categories": ["social"],
            "entity_type": "persona",
            "archetype": "sentiment_tracker",
            "stance_bias": "bullish",
        },
        {
            "agent_id": "a4",
            "name": "Policy Wire",
            "coverage_bucket": "policy",
            "focus_categories": ["political"],
            "entity_type": "persona",
            "archetype": "policy_watch",
            "stance_bias": "neutral",
        },
    ]
    rounds = [
        {
            "round_index": 0,
            "active_agent_ids": ["a1", "a2"],
            "summary": "Rates and breadth dominated the discussion.",
            "stance_histogram": {"bullish": 1, "bearish": 1},
            "consensus_score": 0.35,
            "conflict_score": 0.55,
            "actions": [
                {
                    "agent_id": "a1",
                    "action_type": "create_post",
                    "direction": "bearish",
                    "referenced_feature_names": ["rates_repricing"],
                },
                {
                    "agent_id": "a2",
                    "action_type": "comment",
                    "direction": "bullish",
                    "referenced_feature_names": ["breadth_risk_on"],
                },
            ],
        },
        {
            "round_index": 1,
            "active_agent_ids": ["a1", "a2", "a3", "a4"],
            "summary": "Breadth stayed strong while rates remained contested.",
            "stance_histogram": {"bullish": 2, "bearish": 1},
            "consensus_score": 0.58,
            "conflict_score": 0.42,
            "actions": [
                {
                    "agent_id": "a2",
                    "action_type": "create_post",
                    "direction": "bullish",
                    "referenced_feature_names": ["breadth_risk_on"],
                },
                {
                    "agent_id": "a3",
                    "action_type": "comment",
                    "direction": "bullish",
                    "referenced_feature_names": ["breadth_risk_on"],
                },
            ],
        },
    ]

    payload = build_swarm_reporting_payload(
        profiles=profiles,
        rounds=rounds,
        setup={"seed_posts": [{"agent_id": "a1"}]},
        summary={"dominant_stance": "bullish", "average_consensus": 0.465, "average_conflict": 0.485},
    )

    assert payload["totals"]["total_personas"] == 4
    assert payload["totals"]["round_count"] == 2
    assert payload["totals"]["participation_rate"] == 0.75
    assert payload["cluster_composition"][0]["cluster_id"] == "market"
    assert payload["top_debated_themes"][0]["theme"] == "breadth_risk_on"
    assert payload["agent_activity"]["highlighted_agents"][0]["agent_id"] == "a2"
    assert payload["agent_activity"]["low_priority_count"] == 1
    assert payload["round_highlights"][0]["top_themes"]
    assert payload["guardrails"]["no_duplicate_ids"] is True


def test_swarm_reporting_is_deterministic_for_fixed_seed_in_rule_based_mode() -> None:
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_persona_count": 12,
            "swarm_dynamic_agent_count": 0,
            "swarm_rounds": 4,
            "swarm_agents_per_round_min": 3,
            "swarm_agents_per_round_max": 5,
            "swarm_parallel_enabled": False,
            "swarm_random_seed": 17,
        }
    )
    item = _sample_item()
    feature = _sample_feature()

    environment_one = prepare_swarm_environment(
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )
    environment_two = prepare_swarm_environment(
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    assert [profile.agent_id for profile in environment_one.profiles] == [profile.agent_id for profile in environment_two.profiles]
    assert [profile.name for profile in environment_one.profiles] == [profile.name for profile in environment_two.profiles]
    assert environment_one.time_config.to_dict() == environment_two.time_config.to_dict()

    result_one = run_swarm_from_environment(
        environment=environment_one,
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )
    result_two = run_swarm_from_environment(
        environment=environment_two,
        items=[item],
        snapshot={},
        features=[feature],
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
    )

    assert [round_result.to_dict() for round_result in result_one.rounds] == [round_result.to_dict() for round_result in result_two.rounds]
    assert [feature_item.to_dict() for feature_item in result_one.derived_features] == [feature_item.to_dict() for feature_item in result_two.derived_features]
    assert result_one.summary_metrics == result_two.summary_metrics
    assert result_one.priors == result_two.priors


def test_large_swarm_rule_based_smoke_validation_records_runtime_and_quality_signals() -> None:
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_parallel_enabled": False,
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
    reporting = build_swarm_reporting_payload(
        profiles=[profile.to_dict() for profile in result.profiles],
        rounds=[round_result.to_dict() for round_result in result.rounds],
        setup=result.setup.to_dict(),
        summary=result.summary_metrics,
    )

    assert reporting["scale_mode"] == "large"
    assert reporting["guardrails"]["exactly_50_personas"] is True
    assert reporting["guardrails"]["at_least_10_rounds"] is True
    assert reporting["guardrails"]["at_least_10_active_agents_each_round"] is True
    assert reporting["dominant_signals"]
    assert reporting["top_debated_themes"]
    assert reporting["round_highlights"]
    assert result.diagnostics["fallback_mode"] is True
    assert result.diagnostics["runtime_seconds"] > 0.0
