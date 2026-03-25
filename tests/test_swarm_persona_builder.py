from market_direction_dashboard.config import load_config
from market_direction_dashboard.core.domain import Entity, KnowledgeGraph
from market_direction_dashboard.llm_clients import RuleBasedLLMClient
from market_direction_dashboard.models import SignalFeature, SourceItem
from market_direction_dashboard.swarm_simulation import prepare_swarm_environment


def _sample_items() -> list[SourceItem]:
    return [
        SourceItem(
            id="doc-market-1",
            title="Breadth improves across major indices",
            source="Reuters",
            category="market",
            published_at="2026-03-18",
            url="https://example.com/market-1",
            summary="Breadth and participation improved.",
            impact="medium",
            impact_score=0.6,
            direction="bullish",
        ),
        SourceItem(
            id="doc-economic-1",
            title="Inflation data cools again",
            source="Bloomberg",
            category="economic",
            published_at="2026-03-18",
            url="https://example.com/economic-1",
            summary="Inflation cooled more than expected.",
            impact="high",
            impact_score=0.7,
            direction="bullish",
        ),
        SourceItem(
            id="doc-social-1",
            title="Retail dip buyers return",
            source="SocialWire",
            category="social",
            published_at="2026-03-18",
            url="https://example.com/social-1",
            summary="Retail chatter turned aggressively risk-on.",
            impact="medium",
            impact_score=0.45,
            direction="bullish",
        ),
        SourceItem(
            id="doc-political-1",
            title="Trade tensions rise",
            source="AP",
            category="political",
            published_at="2026-03-18",
            url="https://example.com/political-1",
            summary="Trade rhetoric introduced fresh uncertainty.",
            impact="high",
            impact_score=0.65,
            direction="bearish",
        ),
        SourceItem(
            id="doc-sector-1",
            title="Credit spreads widen in cyclicals",
            source="FT",
            category="sector",
            published_at="2026-03-18",
            url="https://example.com/sector-1",
            summary="Cyclical sectors and credit spreads weakened together.",
            impact="high",
            impact_score=0.72,
            direction="bearish",
        ),
    ]


def _sample_features() -> list[SignalFeature]:
    return [
        SignalFeature(name="breadth_risk_on", direction="bullish", strength=0.74, category="market", summary="Broad participation improved.", supporting_evidence_ids=["doc-market-1"]),
        SignalFeature(name="inflation_cooling", direction="bullish", strength=0.68, category="economic", summary="Inflation eased.", supporting_evidence_ids=["doc-economic-1"]),
        SignalFeature(name="retail_euphoria", direction="bullish", strength=0.57, category="social", summary="Retail enthusiasm increased.", supporting_evidence_ids=["doc-social-1"]),
        SignalFeature(name="policy_uncertainty", direction="bearish", strength=0.63, category="political", summary="Policy risk intensified.", supporting_evidence_ids=["doc-political-1"]),
        SignalFeature(name="credit_spread_stress", direction="bearish", strength=0.71, category="credit", summary="Credit spreads widened.", supporting_evidence_ids=["doc-sector-1"]),
    ]


def _sample_graph() -> KnowledgeGraph:
    entities = [
        Entity(entity_id="evidence:doc-market-1", entity_type="Evidence", name="Breadth improves", properties={"document_id": "doc-market-1", "category": "market", "direction": "bullish", "source": "Reuters"}),
        Entity(entity_id="evidence:doc-economic-1", entity_type="Evidence", name="Inflation data cools", properties={"document_id": "doc-economic-1", "category": "economic", "direction": "bullish", "source": "Bloomberg"}),
        Entity(entity_id="evidence:doc-social-1", entity_type="Evidence", name="Retail dip buyers return", properties={"document_id": "doc-social-1", "category": "social", "direction": "bullish", "source": "SocialWire"}),
        Entity(entity_id="evidence:doc-political-1", entity_type="Evidence", name="Trade tensions rise", properties={"document_id": "doc-political-1", "category": "political", "direction": "bearish", "source": "AP"}),
        Entity(entity_id="evidence:doc-sector-1", entity_type="Evidence", name="Credit spreads widen", properties={"document_id": "doc-sector-1", "category": "credit", "direction": "bearish", "source": "FT"}),
        Entity(entity_id="evidence:doc-vol-1", entity_type="Evidence", name="Volatility demand climbs", properties={"document_id": "doc-vol-1", "category": "volatility", "direction": "bearish", "source": "CBOE"}),
    ]
    return KnowledgeGraph(graph_id="swarm-builder-test", entities=entities, relationships=[])


def _build_environment(seed: int = 42):
    config = load_config(
        {
            "llm_provider": "rule_based",
            "swarm_random_seed": seed,
            "swarm_persona_count": 50,
            "swarm_rounds": 4,
            "swarm_agents_per_round_min": 6,
            "swarm_agents_per_round_max": 9,
        }
    )
    return prepare_swarm_environment(
        items=_sample_items(),
        snapshot={},
        features=_sample_features(),
        llm_client=RuleBasedLLMClient(),
        config=config,
        target="S&P 500",
        graph=_sample_graph(),
    )


def test_prepare_swarm_environment_builds_balanced_deterministic_population() -> None:
    environment = _build_environment()

    assert len(environment.profiles) == 50
    assert environment.diagnostics["profile_count"] == 50
    assert environment.diagnostics["cluster_counts"] == {
        "macro": 14,
        "market_structure": 12,
        "sentiment_retail": 8,
        "policy_geopolitical": 7,
        "sector_credit_volatility": 9,
    }
    assert environment.diagnostics["cluster_targets"] == environment.diagnostics["cluster_counts"]
    assert environment.diagnostics["cluster_balanced"] is True
    assert environment.diagnostics["duplicate_agent_ids"] is False
    assert len({profile.agent_id for profile in environment.profiles}) == 50
    assert any(profile.persona_layer == "dynamic" for profile in environment.profiles)
    assert any(profile.persona_layer == "community" for profile in environment.profiles)
    assert any(profile.persona_layer == "meta" for profile in environment.profiles)
    assert any(profile.persona_layer == "variant" for profile in environment.profiles)


def test_prepare_swarm_environment_is_stable_for_same_seed() -> None:
    first = _build_environment(seed=99)
    second = _build_environment(seed=99)

    assert [
        (
            profile.agent_id,
            profile.username,
            profile.bio,
            profile.cluster,
            profile.persona_layer,
        )
        for profile in first.profiles
    ] == [
        (
            profile.agent_id,
            profile.username,
            profile.bio,
            profile.cluster,
            profile.persona_layer,
        )
        for profile in second.profiles
    ]


def test_prepare_swarm_environment_changes_variant_identity_when_seed_changes() -> None:
    first = _build_environment(seed=41)
    second = _build_environment(seed=43)

    assert [profile.agent_id for profile in first.profiles] != [profile.agent_id for profile in second.profiles]
    assert first.diagnostics["cluster_counts"] == second.diagnostics["cluster_counts"]
