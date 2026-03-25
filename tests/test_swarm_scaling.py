from market_direction_dashboard.config import load_config
from market_direction_dashboard.llm_clients import RuleBasedLLMClient
from market_direction_dashboard.models import SignalFeature, SourceItem
from market_direction_dashboard.swarm_simulation import prepare_swarm_environment, run_swarm_from_environment


def _source_item(item_id: str, category: str, direction: str) -> SourceItem:
    return SourceItem(
        id=item_id,
        title=f"{category} item",
        source=f"{category}-source",
        category=category,
        published_at="2026-03-18",
        url="",
        summary=f"{category} summary",
        impact="medium",
        impact_score=0.6,
        direction=direction,
        quality_score=0.8,
        freshness_score=0.8,
        credibility_score=0.8,
    )


def _feature(name: str, category: str, direction: str, evidence_id: str, strength: float) -> SignalFeature:
    return SignalFeature(
        name=name,
        direction=direction,
        strength=strength,
        supporting_evidence_ids=[evidence_id],
        category=category,
        summary=f"{name} summary",
    )


def _build_inputs() -> tuple[list[SourceItem], list[SignalFeature], dict]:
    items = [
        _source_item("macro-doc", "economic", "bearish"),
        _source_item("market-doc", "market", "bullish"),
        _source_item("sentiment-doc", "social", "bullish"),
        _source_item("policy-doc", "political", "bearish"),
    ]
    features = [
        _feature("macro_pressure", "economic", "bearish", "macro-doc", 0.9),
        _feature("market_breadth", "market", "bullish", "market-doc", 0.85),
        _feature("retail_sentiment", "social", "bullish", "sentiment-doc", 0.8),
        _feature("policy_uncertainty", "political", "bearish", "policy-doc", 0.82),
    ]
    return items, features, {"series": {}}


def _run_simulation(seed: int = 42):
    items, features, snapshot = _build_inputs()
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_persona_count": 20,
            "swarm_dynamic_agent_count": 8,
            "swarm_rounds": 4,
            "swarm_agents_per_round_min": 10,
            "swarm_agents_per_round_max": 12,
            "swarm_primary_agent_count": 4,
            "swarm_secondary_agent_count": 3,
            "swarm_parallel_worker_cap": 2,
            "swarm_random_seed": seed,
        }
    )
    llm_client = RuleBasedLLMClient()
    environment = prepare_swarm_environment(
        items=items,
        snapshot=snapshot,
        features=features,
        llm_client=llm_client,
        config=config,
        target="S&P 500",
    )
    return run_swarm_from_environment(
        environment=environment,
        items=items,
        snapshot=snapshot,
        features=features,
        llm_client=llm_client,
        config=config,
        target="S&P 500",
    )


def test_swarm_rounds_are_large_diverse_and_tiered() -> None:
    result = _run_simulation()

    round_sizes = [len(round_result.active_agent_ids) for round_result in result.rounds]
    assert min(round_sizes) >= 10
    assert len(set(round_sizes)) > 1

    for round_result in result.rounds:
        diagnostics = round_result.selection_diagnostics
        assert {"macro", "market", "sentiment", "policy"} <= set(diagnostics["coverage_buckets"])
        assert diagnostics["tier_counts"]["primary"] == 4
        assert diagnostics["tier_counts"]["secondary"] == 3
        assert diagnostics["tier_counts"]["background"] >= 1
        assert diagnostics["llm_action_count"] == 7
        assert diagnostics["templated_action_count"] >= 1
        assert diagnostics["parallel_worker_cap"] <= 2
        assert len({entry["coverage_bucket"] for entry in diagnostics["plan"]}) >= 4


def test_swarm_selection_is_deterministic_for_same_seed() -> None:
    left = _run_simulation(seed=77)
    right = _run_simulation(seed=77)

    assert [
        (
            round_result.active_agent_ids,
            [(entry["agent_id"], entry["execution_tier"]) for entry in round_result.selection_diagnostics["plan"]],
        )
        for round_result in left.rounds
    ] == [
        (
            round_result.active_agent_ids,
            [(entry["agent_id"], entry["execution_tier"]) for entry in round_result.selection_diagnostics["plan"]],
        )
        for round_result in right.rounds
    ]
