from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha1
from random import Random
from typing import Any

from .config import (
    SWARM_DEFAULT_PERSONA_COUNT,
    SWARM_DEFAULT_ROUNDS,
    _default_swarm_dynamic_agent_count,
)


SWARM_CLUSTER_ORDER = [
    "macro",
    "market_structure",
    "sentiment_retail",
    "policy_geopolitical",
    "sector_credit_volatility",
]
SWARM_CLUSTER_BASE_QUOTAS = {
    "macro": 14,
    "market_structure": 12,
    "sentiment_retail": 8,
    "policy_geopolitical": 7,
    "sector_credit_volatility": 9,
}
_STANCES = ("bullish", "bearish", "neutral")
_ACTIVITY_LEVELS = ("low", "medium", "high")
_CATEGORY_LABELS = {
    "economic": "macro",
    "market": "market_structure",
    "social": "sentiment_retail",
    "political": "policy_geopolitical",
    "sector": "sector_credit_volatility",
    "credit": "sector_credit_volatility",
    "volatility": "sector_credit_volatility",
    "community": "market_structure",
}
_CLUSTER_CATEGORY_POOLS = {
    "macro": ["economic", "market", "credit"],
    "market_structure": ["market", "economic", "volatility"],
    "sentiment_retail": ["social", "market"],
    "policy_geopolitical": ["political", "economic", "market"],
    "sector_credit_volatility": ["sector", "credit", "volatility", "market"],
}
_CLUSTER_ACTIVITY_HOURS = {
    "macro": [7, 8, 9, 10, 11, 12, 13, 14, 15],
    "market_structure": [8, 9, 10, 11, 12, 13, 14, 15, 16],
    "sentiment_retail": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "policy_geopolitical": [7, 8, 9, 10, 11, 12, 13, 14, 15],
    "sector_credit_volatility": [8, 9, 10, 11, 12, 13, 14, 15, 16],
}


@dataclass(frozen=True)
class PersonaTemplate:
    template_id: str
    cluster: str
    archetype: str
    name: str
    base_stance: str
    focus_categories: tuple[str, ...]
    activity_band: tuple[float, float]
    influence_band: tuple[float, float]
    descriptor: str
    voice: str
    title_pool: tuple[str, ...]


@dataclass(frozen=True)
class CandidatePersona:
    layer: str
    cluster: str
    template_id: str
    stable_key: str
    name: str
    archetype: str
    entity_name: str
    entity_type: str
    stance_bias: str
    focus_categories: list[str]
    activity_level: float
    influence_weight: float
    bio: str
    persona: str
    system_prompt: str
    seed_evidence_ids: list[str]
    category_focus_label: str
    variant_key: str


TEMPLATE_CATALOG: tuple[PersonaTemplate, ...] = (
    PersonaTemplate("macro_hawk", "macro", "macro_hawk", "Macro Hawk", "bearish", ("economic", "market"), (0.58, 0.86), (0.82, 1.12), "rate pressure and slowing liquidity", "crisp macro desk notes", ("Rate Desk", "Inflation Watch", "Liquidity Monitor")),
    PersonaTemplate("macro_dove", "macro", "macro_dove", "Macro Dove", "bullish", ("economic", "market"), (0.56, 0.84), (0.8, 1.08), "disinflation and easing conditions", "calm soft-landing framing", ("Soft Landing", "Disinflation Desk", "Growth Balance")),
    PersonaTemplate("yield_curve_strategist", "macro", "yield_curve_strategist", "Yield Curve Strategist", "bearish", ("economic", "credit"), (0.54, 0.82), (0.78, 1.02), "curve stress and funding fragility", "cross-asset macro framing", ("Curve Desk", "Funding Tape", "Rates Crosscheck")),
    PersonaTemplate("labor_cycle_analyst", "macro", "labor_cycle_analyst", "Labor Cycle Analyst", "neutral", ("economic",), (0.48, 0.76), (0.72, 0.96), "jobs, wages, and demand durability", "slow-moving data synthesis", ("Payroll Lens", "Demand Tracker", "Cycle Notes")),
    PersonaTemplate("market_microstructure_analyst", "market_structure", "market_microstructure_analyst", "Market Microstructure Analyst", "neutral", ("market", "volatility"), (0.58, 0.88), (0.82, 1.08), "flows, breadth, and tape quality", "order-flow language", ("Tape Monitor", "Flow Book", "Breadth Check")),
    PersonaTemplate("momentum_trader", "market_structure", "momentum_trader", "Momentum Trader", "bullish", ("market", "social"), (0.62, 0.92), (0.78, 1.02), "trend persistence and breakout follow-through", "fast tape commentary", ("Breakout Desk", "Momentum Board", "Trend Scout")),
    PersonaTemplate("mean_reversion_trader", "market_structure", "mean_reversion_trader", "Mean Reversion Trader", "bearish", ("market",), (0.56, 0.82), (0.76, 0.98), "crowded positioning and snapback risk", "skeptical tactical notes", ("Reversion Watch", "Stretch Check", "Fade Book")),
    PersonaTemplate("index_flow_observer", "market_structure", "index_flow_observer", "Index Flow Observer", "neutral", ("market", "economic"), (0.52, 0.8), (0.74, 0.98), "index leadership and passive flow concentration", "measured market plumbing commentary", ("Index Breadth", "Leadership Map", "Flow Check")),
    PersonaTemplate("retail_sentiment_watcher", "sentiment_retail", "retail_sentiment_watcher", "Retail Sentiment Watcher", "bullish", ("social", "market"), (0.62, 0.9), (0.68, 0.92), "retail enthusiasm and crowd chasing", "short social-tone bursts", ("Crowd Pulse", "Retail Tape", "Momentum Feed")),
    PersonaTemplate("options_chatter_analyst", "sentiment_retail", "options_chatter_analyst", "Options Chatter Analyst", "neutral", ("social", "volatility"), (0.58, 0.88), (0.72, 0.94), "speculative options flow and gamma chatter", "derivatives-fluent crowd commentary", ("Gamma Watch", "Flow Scanner", "Spec Desk")),
    PersonaTemplate("geopolitical_watcher", "policy_geopolitical", "geopolitical_watcher", "Geopolitical Watcher", "bearish", ("political", "market"), (0.5, 0.78), (0.76, 1.0), "headline risk and regional escalation", "event-driven geopolitical framing", ("Risk Radar", "Conflict Brief", "Headline Desk")),
    PersonaTemplate("policy_whip_counter", "policy_geopolitical", "policy_whip_counter", "Policy Whip Counter", "neutral", ("political", "economic"), (0.46, 0.74), (0.74, 0.98), "policy sequencing and legislative friction", "Washington process notes", ("Policy Sheet", "Capitol Scan", "Regulation Watch")),
    PersonaTemplate("sector_rotation_watcher", "sector_credit_volatility", "sector_rotation_watcher", "Sector Rotation Watcher", "bullish", ("sector", "market"), (0.56, 0.84), (0.76, 1.0), "leadership changes across sectors", "portfolio rotation notes", ("Leadership Grid", "Rotation Map", "Sector Breadth")),
    PersonaTemplate("credit_risk_watcher", "sector_credit_volatility", "credit_risk_watcher", "Credit Risk Watcher", "bearish", ("credit", "economic"), (0.54, 0.82), (0.84, 1.12), "credit spread stress and financing conditions", "balance-sheet-first commentary", ("Spread Watch", "Funding Stress", "Credit Pulse")),
    PersonaTemplate("volatility_watcher", "sector_credit_volatility", "volatility_watcher", "Volatility Watcher", "bearish", ("volatility", "market"), (0.58, 0.86), (0.8, 1.06), "volatility regime shifts and hedging demand", "defensive risk framing", ("Vol Monitor", "Hedge Tape", "Risk Regime")),
)


@dataclass(frozen=True)
class GraphDrivenEnvironmentBuilder:
    graph: Any
    items: list[Any]
    features: list[Any]
    config: dict[str, Any]
    rng: Random

    def build_profiles(self) -> list[Any]:
        from .swarm_simulation import AgentProfile

        seed = int(self.config.get("swarm_random_seed", 42) or 42)
        total_rounds = max(1, int(self.config.get("swarm_rounds", SWARM_DEFAULT_ROUNDS) or SWARM_DEFAULT_ROUNDS))
        target_count = max(1, int(self.config.get("swarm_persona_count", SWARM_DEFAULT_PERSONA_COUNT) or SWARM_DEFAULT_PERSONA_COUNT))
        dynamic_target = max(0, min(target_count, int(self.config.get("swarm_dynamic_agent_count", _default_swarm_dynamic_agent_count(target_count)) or 0)))
        quotas = _resolve_cluster_quotas(target_count)
        profiles: list[AgentProfile] = []
        counts: Counter[str] = Counter()
        dynamic_counts: Counter[str] = Counter()
        community_counts: Counter[str] = Counter()
        variant_counts: Counter[str] = Counter()
        seen_ids: set[str] = set()
        templates_by_cluster = _templates_by_cluster()

        for template in TEMPLATE_CATALOG:
            if counts[template.cluster] >= quotas[template.cluster]:
                continue
            profile = self._make_stable_profile(template, seed=seed, total_rounds=total_rounds)
            profiles.append(_claim_profile(profile, seen_ids))
            counts[template.cluster] += 1

        dynamic_cap = {cluster: max(1, quota // 4) for cluster, quota in quotas.items()}
        for candidate in self._dynamic_candidates(seed=seed):
            if sum(dynamic_counts.values()) >= dynamic_target:
                break
            if counts[candidate.cluster] >= quotas[candidate.cluster]:
                continue
            if dynamic_counts[candidate.cluster] >= dynamic_cap[candidate.cluster]:
                continue
            profile = self._candidate_to_profile(candidate, total_rounds=total_rounds)
            profiles.append(_claim_profile(profile, seen_ids))
            counts[candidate.cluster] += 1
            dynamic_counts[candidate.cluster] += 1

        community_cap = {cluster: 1 for cluster in quotas}
        meta_cap = {cluster: 1 for cluster in quotas}
        for candidate in self._community_candidates(seed=seed):
            if counts[candidate.cluster] >= quotas[candidate.cluster]:
                continue
            if candidate.layer == "meta" and community_counts[(candidate.cluster, "meta")] >= meta_cap[candidate.cluster]:
                continue
            if candidate.layer == "community" and community_counts[(candidate.cluster, "community")] >= community_cap[candidate.cluster]:
                continue
            profile = self._candidate_to_profile(candidate, total_rounds=total_rounds)
            profiles.append(_claim_profile(profile, seen_ids))
            counts[candidate.cluster] += 1
            community_counts[(candidate.cluster, candidate.layer)] += 1

        for cluster in SWARM_CLUSTER_ORDER:
            cluster_templates = templates_by_cluster[cluster]
            while counts[cluster] < quotas[cluster]:
                ordinal = variant_counts[cluster] + 1
                template = cluster_templates[(ordinal - 1) % len(cluster_templates)]
                candidate = self._make_variant_candidate(template, ordinal=ordinal, seed=seed)
                profile = self._candidate_to_profile(candidate, total_rounds=total_rounds)
                profiles.append(_claim_profile(profile, seen_ids))
                counts[cluster] += 1
                variant_counts[cluster] += 1

        profiles.sort(key=lambda profile: profile.agent_id)
        if len(profiles) != target_count:
            raise ValueError(f"expected {target_count} personas, built {len(profiles)}")
        if len({profile.agent_id for profile in profiles}) != len(profiles):
            raise ValueError("duplicate persona IDs generated")
        return profiles

    def build_social_edges(self, profiles: list[Any]) -> list[Any]:
        from .swarm_simulation import SocialEdge

        edges: list[Any] = []
        for source in profiles:
            for target in profiles:
                if source.agent_id == target.agent_id:
                    continue
                shared_categories = sorted(set(source.focus_categories) & set(target.focus_categories))
                same_cluster = getattr(source, "cluster", "") == getattr(target, "cluster", "")
                same_layer = getattr(source, "persona_layer", "") == getattr(target, "persona_layer", "")
                same_stance = source.stance_bias == target.stance_bias
                shared_count = len(shared_categories)
                affinity = 0.18 + shared_count * 0.16 + (0.12 if same_cluster else 0.0) + (0.08 if same_layer else 0.0)
                trust = 0.16 + shared_count * 0.09 + (0.18 if same_stance else 0.0) + (0.08 if same_cluster else 0.0)
                conflict = 0.08 + (0.28 if {source.stance_bias, target.stance_bias} == {"bullish", "bearish"} else 0.0) - shared_count * 0.04
                influence = (source.influence_weight / max(target.influence_weight, 0.1)) * (0.38 if same_cluster else 0.32)
                edges.append(
                    SocialEdge(
                        source_agent_id=source.agent_id,
                        target_agent_id=target.agent_id,
                        affinity_score=round(min(1.0, max(0.0, affinity)), 3),
                        trust_score=round(min(1.0, max(0.0, trust)), 3),
                        influence_score=round(min(1.0, max(0.0, influence)), 3),
                        conflict_score=round(min(1.0, max(0.0, conflict)), 3),
                        shared_categories=shared_categories,
                    )
                )
        return edges

    def build_diagnostics(
        self,
        *,
        profiles: list[Any],
        activity_configs: list[Any],
        seed_posts: list[Any],
        social_edges: list[Any],
        memory_snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        target_count = max(1, int(self.config.get("swarm_persona_count", SWARM_DEFAULT_PERSONA_COUNT) or SWARM_DEFAULT_PERSONA_COUNT))
        cluster_targets = _resolve_cluster_quotas(target_count)
        cluster_counts = Counter(getattr(profile, "cluster", "unknown") or "unknown" for profile in profiles)
        layer_counts = Counter(getattr(profile, "persona_layer", "unknown") or "unknown" for profile in profiles)
        dynamic_profiles = [profile for profile in profiles if str(profile.archetype).startswith("dynamic_")]
        community_profiles = [
            profile
            for profile in profiles
            if getattr(profile, "persona_layer", "") in {"community", "meta"}
        ]
        return {
            "profile_count": len(profiles),
            "dynamic_profile_count": len(dynamic_profiles),
            "community_profile_count": len(community_profiles),
            "behavior_config_count": len(activity_configs),
            "seed_post_count": len(seed_posts),
            "social_edge_count": len(social_edges),
            "seeded_agent_ids": sorted({post.agent_id for post in seed_posts}),
            "entity_types": sorted({profile.entity_type for profile in profiles}),
            "graph_driven": self.graph is not None,
            "graph_entity_count": len(getattr(self.graph, "entities", [])) if self.graph is not None else 0,
            "memory_loaded": bool((memory_snapshot.get("shared") or {}).get("episode_count")),
            "communities": [profile.agent_id for profile in community_profiles],
            "cluster_targets": dict(cluster_targets),
            "cluster_counts": {cluster: cluster_counts.get(cluster, 0) for cluster in SWARM_CLUSTER_ORDER},
            "cluster_balanced": all(cluster_counts.get(cluster, 0) == target for cluster, target in cluster_targets.items()),
            "layer_counts": dict(layer_counts),
            "duplicate_agent_ids": len({profile.agent_id for profile in profiles}) != len(profiles),
        }

    def build_behavior_overrides(self, profile: Any) -> dict[str, Any]:
        cluster = getattr(profile, "cluster", "market_structure") or "market_structure"
        activity_label = getattr(profile, "activity_bucket", _bucket_activity(profile.activity_level))
        cluster_hours = list(_CLUSTER_ACTIVITY_HOURS.get(cluster, _CLUSTER_ACTIVITY_HOURS["market_structure"]))
        if activity_label == "high":
            posts_per_hour = 0.82
            comments_per_hour = 1.28
            response_delay = (4, 18)
        elif activity_label == "low":
            posts_per_hour = 0.42
            comments_per_hour = 0.72
            response_delay = (18, 52)
        else:
            posts_per_hour = 0.6
            comments_per_hour = 0.96
            response_delay = (10, 34)
        if cluster == "policy_geopolitical":
            response_delay = (max(6, response_delay[0]), max(24, response_delay[1]))
        return {
            "active_hours": cluster_hours,
            "posts_per_hour": posts_per_hour,
            "comments_per_hour": comments_per_hour,
            "response_delay_min": response_delay[0],
            "response_delay_max": response_delay[1],
        }

    def _make_stable_profile(self, template: PersonaTemplate, *, seed: int, total_rounds: int) -> Any:
        from .swarm_simulation import AgentProfile

        token = _stable_token(seed, template.cluster, "stable", template.template_id)
        activity_level = _value_in_range(template.activity_band, seed, template.template_id, "stable", "activity")
        influence_weight = _value_in_range(template.influence_band, seed, template.template_id, "stable", "influence")
        prompt_focus = ", ".join(template.focus_categories)
        return AgentProfile(
            agent_id=f"stable-{template.cluster}-{template.template_id}",
            name=template.name,
            username=_build_username(template.name, token),
            archetype=template.archetype,
            entity_name=template.name,
            entity_type="persona",
            stance_bias=template.base_stance,
            focus_categories=list(template.focus_categories),
            activity_level=activity_level,
            active_rounds=list(range(total_rounds)),
            influence_weight=influence_weight,
            system_prompt=f"You are {template.name}. Focus on {prompt_focus} while speaking in {template.voice}.",
            bio=f"{template.name} tracks {template.descriptor} across {prompt_focus}.",
            persona=f"{template.name} publishes concise market reactions with a {template.base_stance} lean.",
            seed_evidence_ids=[],
            cluster=template.cluster,
            persona_layer="stable",
            template_id=template.template_id,
            variant_key="stable",
            category_focus_label=prompt_focus,
            activity_bucket=_bucket_activity(activity_level),
        )

    def _make_variant_candidate(self, template: PersonaTemplate, *, ordinal: int, seed: int) -> CandidatePersona:
        title = template.title_pool[(ordinal - 1) % len(template.title_pool)]
        variant_key = f"variant-{ordinal:02d}"
        name = f"{template.name} {title}"
        stance_bias = _variant_stance(template.base_stance, seed, template.template_id, ordinal)
        focus_categories = _variant_focus_categories(template, seed=seed, ordinal=ordinal)
        activity_level = _value_in_range(template.activity_band, seed, template.template_id, ordinal, "activity")
        influence_weight = _value_in_range(template.influence_band, seed, template.template_id, ordinal, "influence")
        focus_label = ", ".join(focus_categories)
        return CandidatePersona(
            layer="variant",
            cluster=template.cluster,
            template_id=template.template_id,
            stable_key=f"{template.cluster}:{template.template_id}:{variant_key}",
            name=name,
            archetype=f"{template.archetype}_variant",
            entity_name=name,
            entity_type="persona",
            stance_bias=stance_bias,
            focus_categories=focus_categories,
            activity_level=activity_level,
            influence_weight=influence_weight,
            bio=f"{name} extends {template.name.lower()} coverage into {focus_label} with attention to {template.descriptor}.",
            persona=f"{name} writes in {template.voice} with a {stance_bias} stance and a {focus_label} focus.",
            system_prompt=f"You are {name}. Cover {focus_label} with {template.voice} and respect your {stance_bias} bias.",
            seed_evidence_ids=[],
            category_focus_label=focus_label,
            variant_key=variant_key,
        )

    def _dynamic_candidates(self, *, seed: int) -> list[CandidatePersona]:
        candidates: list[CandidatePersona] = []
        for feature in sorted(self.features, key=lambda item: (-float(item.strength), item.name)):
            cluster = _cluster_for_category(getattr(feature, "category", "market"), feature_name=getattr(feature, "name", ""))
            direction = feature.direction if feature.direction in _STANCES else "neutral"
            focus_categories = _coerce_focus_categories([getattr(feature, "category", "market")], cluster)
            name = f"{_display_name(feature.name)} Signal Desk"
            activity_level = round(min(0.97, 0.66 + min(0.2, float(feature.strength) * 0.24)), 3)
            influence_weight = round(min(1.18, 0.84 + min(0.26, float(feature.strength) * 0.28)), 3)
            candidates.append(
                CandidatePersona(
                    layer="dynamic",
                    cluster=cluster,
                    template_id=f"feature_{feature.category}",
                    stable_key=f"feature:{feature.name}",
                    name=name,
                    archetype=f"dynamic_{feature.category}",
                    entity_name=feature.name,
                    entity_type=feature.category,
                    stance_bias=direction,
                    focus_categories=focus_categories,
                    activity_level=activity_level,
                    influence_weight=influence_weight,
                    bio=f"{name} is anchored to the live signal {feature.name} and reacts to {getattr(feature, 'summary', '') or feature.category} evidence.",
                    persona=f"Feature-driven agent with a {direction} lean based on {feature.name}.",
                    system_prompt=f"You are a dynamic analyst anchored to the signal {feature.name}. Argue from concrete evidence only.",
                    seed_evidence_ids=list(getattr(feature, "supporting_evidence_ids", [])[:4]),
                    category_focus_label=", ".join(focus_categories),
                    variant_key=_stable_token(seed, "feature", feature.name, length=8),
                )
            )
        if candidates:
            return candidates

        for item in sorted(self.items, key=lambda current: (-abs(float(getattr(current, "impact_score", 0.0))), getattr(current, "id", ""), getattr(current, "title", ""))):
            cluster = _cluster_for_category(getattr(item, "category", "market"), feature_name=getattr(item, "title", ""))
            direction = item.direction if item.direction in _STANCES else "neutral"
            focus_categories = _coerce_focus_categories([getattr(item, "category", "market")], cluster)
            seed_key = getattr(item, "id", "") or getattr(item, "source", "") or getattr(item, "title", "")
            name = f"{_display_name(getattr(item, 'source', '') or getattr(item, 'category', 'market'))} Flow Desk"
            candidates.append(
                CandidatePersona(
                    layer="dynamic",
                    cluster=cluster,
                    template_id=f"item_{item.category}",
                    stable_key=f"item:{seed_key}",
                    name=name,
                    archetype=f"dynamic_{item.category}",
                    entity_name=getattr(item, "title", "") or getattr(item, "source", ""),
                    entity_type=getattr(item, "category", "market"),
                    stance_bias=direction,
                    focus_categories=focus_categories,
                    activity_level=0.66,
                    influence_weight=0.86,
                    bio=f"{name} is derived from {getattr(item, 'source', 'source')} and tracks {', '.join(focus_categories)} developments.",
                    persona=f"Source-grounded market commentator with a {direction} stance when evidence supports it.",
                    system_prompt=f"You are an analyst derived from {getattr(item, 'source', 'a source')}. Focus on {', '.join(focus_categories)} evidence.",
                    seed_evidence_ids=[getattr(item, "id", "")] if getattr(item, "id", "") else [],
                    category_focus_label=", ".join(focus_categories),
                    variant_key=_stable_token(seed, "item", seed_key, length=8),
                )
            )
        return candidates

    def _community_candidates(self, *, seed: int) -> list[CandidatePersona]:
        evidence_entities = [
            entity
            for entity in getattr(self.graph, "entities", [])
            if getattr(entity, "entity_type", "") == "Evidence"
        ] if self.graph is not None else []
        by_cluster: dict[str, list[Any]] = defaultdict(list)
        for entity in evidence_entities:
            properties = getattr(entity, "properties", {}) or {}
            category = str(properties.get("category") or "").strip().lower()
            cluster = _cluster_for_category(category, feature_name=str(getattr(entity, "name", "")))
            by_cluster[cluster].append(entity)

        candidates: list[CandidatePersona] = []
        for cluster in SWARM_CLUSTER_ORDER:
            entities = sorted(by_cluster.get(cluster, []), key=lambda entity: getattr(entity, "entity_id", ""))
            if not entities:
                continue
            source_names = sorted({
                str((getattr(entity, "properties", {}) or {}).get("source") or "unknown")
                for entity in entities
            })
            seed_evidence_ids = _seed_evidence_ids(entities)
            category_focuses = sorted({
                str((getattr(entity, "properties", {}) or {}).get("category") or "").strip().lower()
                for entity in entities
                if str((getattr(entity, "properties", {}) or {}).get("category") or "").strip()
            }) or list(_CLUSTER_CATEGORY_POOLS.get(cluster, ["market"]))[:1]
            focus_label = ", ".join(category_focuses[:3])
            community_name = f"{_display_name(cluster)} Community"
            candidates.append(
                CandidatePersona(
                    layer="community",
                    cluster=cluster,
                    template_id=f"community_{cluster}",
                    stable_key=f"community:{cluster}",
                    name=community_name,
                    archetype=f"community_{cluster}",
                    entity_name=community_name,
                    entity_type="community",
                    stance_bias=_community_stance(cluster, entities),
                    focus_categories=category_focuses[:3],
                    activity_level=round(min(0.88, 0.54 + len(entities) * 0.018), 3),
                    influence_weight=round(min(1.08, 0.78 + len(source_names) * 0.04), 3),
                    bio=f"{community_name} aggregates {len(entities)} evidence nodes from {', '.join(source_names[:3])} across {focus_label}.",
                    persona=f"Community-level market commentator for {cluster.replace('_', ' ')} narratives.",
                    system_prompt=f"You represent the {cluster.replace('_', ' ')} community emerging from the graph evidence.",
                    seed_evidence_ids=seed_evidence_ids,
                    category_focus_label=focus_label,
                    variant_key=_stable_token(seed, "community", cluster, length=8),
                )
            )
            candidates.append(
                CandidatePersona(
                    layer="meta",
                    cluster=cluster,
                    template_id=f"meta_{cluster}",
                    stable_key=f"meta:{cluster}",
                    name=f"{_display_name(cluster)} Meta Desk",
                    archetype=f"meta_{cluster}",
                    entity_name=f"{cluster}_meta",
                    entity_type="community",
                    stance_bias=_community_stance(cluster, entities),
                    focus_categories=_coerce_focus_categories(category_focuses[:2], cluster),
                    activity_level=round(min(0.84, 0.48 + len(category_focuses) * 0.08), 3),
                    influence_weight=round(min(1.02, 0.74 + len(seed_evidence_ids) * 0.05), 3),
                    bio=f"{_display_name(cluster)} Meta Desk summarizes the strongest themes, contradictions, and concentration risks in {focus_label}.",
                    persona=f"Meta agent that synthesizes {cluster.replace('_', ' ')} narratives into broad market framing.",
                    system_prompt=f"You are a meta analyst summarizing the strongest {cluster.replace('_', ' ')} narratives.",
                    seed_evidence_ids=seed_evidence_ids,
                    category_focus_label=focus_label,
                    variant_key=_stable_token(seed, "meta", cluster, length=8),
                )
            )
        return candidates

    def _candidate_to_profile(self, candidate: CandidatePersona, *, total_rounds: int) -> Any:
        from .swarm_simulation import AgentProfile

        return AgentProfile(
            agent_id=_candidate_agent_id(candidate),
            name=candidate.name,
            username=_build_username(candidate.name, candidate.variant_key),
            archetype=candidate.archetype,
            entity_name=candidate.entity_name,
            entity_type=candidate.entity_type,
            stance_bias=candidate.stance_bias,
            focus_categories=list(candidate.focus_categories),
            activity_level=candidate.activity_level,
            active_rounds=list(range(total_rounds)),
            influence_weight=candidate.influence_weight,
            system_prompt=candidate.system_prompt,
            bio=candidate.bio,
            persona=candidate.persona,
            seed_evidence_ids=list(candidate.seed_evidence_ids),
            cluster=candidate.cluster,
            persona_layer=candidate.layer,
            template_id=candidate.template_id,
            variant_key=candidate.variant_key,
            category_focus_label=candidate.category_focus_label,
            activity_bucket=_bucket_activity(candidate.activity_level),
        )


def _templates_by_cluster() -> dict[str, list[PersonaTemplate]]:
    grouped: dict[str, list[PersonaTemplate]] = {cluster: [] for cluster in SWARM_CLUSTER_ORDER}
    for template in TEMPLATE_CATALOG:
        grouped[template.cluster].append(template)
    return grouped


def _resolve_cluster_quotas(target_count: int) -> dict[str, int]:
    base_total = sum(SWARM_CLUSTER_BASE_QUOTAS.values())
    quotas = {cluster: int(target_count * SWARM_CLUSTER_BASE_QUOTAS[cluster] / base_total) for cluster in SWARM_CLUSTER_ORDER}
    remainder = target_count - sum(quotas.values())
    if remainder:
        ranked = sorted(
            SWARM_CLUSTER_ORDER,
            key=lambda cluster: (
                target_count * SWARM_CLUSTER_BASE_QUOTAS[cluster] / base_total - quotas[cluster],
                SWARM_CLUSTER_BASE_QUOTAS[cluster],
            ),
            reverse=True,
        )
        for cluster in ranked[:remainder]:
            quotas[cluster] += 1
    return quotas


def _stable_token(seed: int, *parts: object, length: int = 10) -> str:
    raw = "::".join(str(part) for part in (seed,) + parts)
    return sha1(raw.encode("utf-8")).hexdigest()[:length]


def _display_name(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value).replace("-", "_").split("_") if part)


def _build_username(name: str, suffix: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in name).strip("_")
    slug = "_".join(filter(None, slug.split("_")))
    return f"{slug or 'agent'}_{str(suffix).lower()[:8]}"


def _value_in_range(bounds: tuple[float, float], seed: int, *parts: object) -> float:
    low, high = bounds
    if high <= low:
        return round(low, 3)
    token = int(_stable_token(seed, *parts, length=12), 16)
    ratio = token / float(16**12 - 1)
    return round(low + (high - low) * ratio, 3)


def _variant_stance(base_stance: str, seed: int, template_id: str, ordinal: int) -> str:
    options_by_base = {
        "bullish": ("bullish", "bullish", "neutral"),
        "bearish": ("bearish", "bearish", "neutral"),
        "neutral": ("neutral", "bullish", "bearish"),
    }
    options = options_by_base.get(base_stance, _STANCES)
    token = int(_stable_token(seed, template_id, ordinal, "stance", length=4), 16)
    return options[token % len(options)]


def _variant_focus_categories(template: PersonaTemplate, *, seed: int, ordinal: int) -> list[str]:
    categories = list(template.focus_categories)
    pool = [item for item in _CLUSTER_CATEGORY_POOLS.get(template.cluster, []) if item not in categories]
    if pool:
        token = int(_stable_token(seed, template.template_id, ordinal, "focus", length=4), 16)
        categories.append(pool[token % len(pool)])
    return categories[:3]


def _cluster_for_category(category: str, *, feature_name: str = "") -> str:
    normalized = str(category or "").strip().lower()
    feature_hint = str(feature_name or "").lower()
    if "sector" in normalized or "sector" in feature_hint:
        return "sector_credit_volatility"
    if "credit" in normalized or "credit" in feature_hint:
        return "sector_credit_volatility"
    if "vol" in normalized or "vix" in feature_hint or "gamma" in feature_hint:
        return "sector_credit_volatility"
    return _CATEGORY_LABELS.get(normalized, "market_structure")


def _coerce_focus_categories(categories: list[str], cluster: str) -> list[str]:
    merged = [str(category).strip().lower() for category in categories if str(category).strip()]
    for fallback in _CLUSTER_CATEGORY_POOLS.get(cluster, []):
        if fallback not in merged:
            merged.append(fallback)
        if len(merged) >= 3:
            break
    return merged[:3] or list(_CLUSTER_CATEGORY_POOLS.get(cluster, ["market"]))[:1]


def _seed_evidence_ids(entities: list[Any]) -> list[str]:
    ids: list[str] = []
    for entity in entities:
        source_document_ids = list(getattr(entity, "source_document_ids", []) or [])
        if source_document_ids:
            ids.extend(str(item) for item in source_document_ids if str(item))
            continue
        properties = getattr(entity, "properties", {}) or {}
        document_id = str(properties.get("document_id") or "").strip()
        if document_id:
            ids.append(document_id)
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped[:4]


def _community_stance(cluster: str, entities: list[Any]) -> str:
    directions = Counter(
        str((getattr(entity, "properties", {}) or {}).get("direction") or "neutral").strip().lower()
        for entity in entities
    )
    bullish = directions.get("bullish", 0)
    bearish = directions.get("bearish", 0)
    if bullish > bearish:
        return "bullish"
    if bearish > bullish:
        return "bearish"
    if cluster in {"macro", "policy_geopolitical", "sector_credit_volatility"}:
        return "bearish"
    if cluster in {"market_structure", "sentiment_retail"}:
        return "bullish"
    return "neutral"


def _candidate_agent_id(candidate: CandidatePersona) -> str:
    return f"{candidate.layer}-{candidate.cluster}-{candidate.template_id}-{candidate.variant_key}".replace("_", "-")


def _bucket_activity(activity_level: float) -> str:
    if activity_level >= 0.78:
        return "high"
    if activity_level <= 0.57:
        return "low"
    return "medium"


def _claim_profile(profile: Any, seen_ids: set[str]) -> Any:
    if profile.agent_id in seen_ids:
        raise ValueError(f"duplicate persona id generated: {profile.agent_id}")
    seen_ids.add(profile.agent_id)
    return profile
