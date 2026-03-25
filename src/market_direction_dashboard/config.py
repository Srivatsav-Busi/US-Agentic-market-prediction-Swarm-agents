from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path


SWARM_DEFAULT_PERSONA_COUNT = 50
SWARM_DEFAULT_ROUNDS = 10
SWARM_DEFAULT_AGENTS_PER_ROUND_MIN = 10
SWARM_DEFAULT_AGENTS_PER_ROUND_MAX = 16
SWARM_DEFAULT_MEMORY_AGENT_ACTIONS = 3
SWARM_DEFAULT_MEMORY_SHARED_ROUNDS = 4
SWARM_DEFAULT_MEMORY_COMMUNITY_REFERENCES = 16
SWARM_DEFAULT_SHARED_REFERENCE_CAP = 24
SWARM_DEFAULT_DECISION_TRACE_CAP = 4
SWARM_DEFAULT_EVENT_HISTORY_CAP = 12
SWARM_DEFAULT_GRAPH_DELTA_CAP = 12
SWARM_DEFAULT_ACTIVE_HISTORY_CAP = 10
SWARM_DEFAULT_PROMPT_SHARED_ROUNDS = 3
SWARM_DEFAULT_PROMPT_AGENT_ACTIONS = 2
SWARM_DEFAULT_DERIVED_MIN_PARTICIPATION = 0.25
SWARM_DEFAULT_DERIVED_CONSENSUS_THRESHOLD = 0.18
SWARM_DEFAULT_DERIVED_CONFLICT_THRESHOLD = 0.22


def _load_neo4j_credentials_file(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}

    credentials: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        credentials[key.strip()] = value.strip()
    return credentials


_DEFAULT_NEO4J_CREDENTIALS_FILE = os.getenv(
    "MARKET_PREDICTION_GRAPH_NEO4J_CREDENTIALS_FILE",
    "./neo4j-credentials.local.txt",
)
_FILE_NEO4J_DEFAULTS = _load_neo4j_credentials_file(_DEFAULT_NEO4J_CREDENTIALS_FILE)


DEFAULT_CONFIG = {
    "project_name": "US Agentic Market Prediction",
    "results_dir": os.getenv("MARKET_PREDICTION_RESULTS_DIR", "./results"),
    "database_url": os.getenv("MARKET_PREDICTION_DATABASE_URL", "sqlite:///./results/market_intelligence.db"),
    "persist_to_db": os.getenv("MARKET_PREDICTION_PERSIST_TO_DB", "false").lower() == "true",
    "graph_enabled": os.getenv("MARKET_PREDICTION_GRAPH_ENABLED", "false").lower() == "true",
    "graph_backend": os.getenv("MARKET_PREDICTION_GRAPH_BACKEND", "neo4j"),
    "graph_neo4j_credentials_file": _DEFAULT_NEO4J_CREDENTIALS_FILE,
    "graph_neo4j_uri": os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_URI", _FILE_NEO4J_DEFAULTS.get("NEO4J_URI", "bolt://localhost:7687")),
    "graph_neo4j_username": os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_USERNAME", _FILE_NEO4J_DEFAULTS.get("NEO4J_USERNAME", "neo4j")),
    "graph_neo4j_password": os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_PASSWORD", _FILE_NEO4J_DEFAULTS.get("NEO4J_PASSWORD", "")),
    "graph_neo4j_database": os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_DATABASE", _FILE_NEO4J_DEFAULTS.get("NEO4J_DATABASE", "neo4j")),
    "graph_neo4j_pool_size": int(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_POOL_SIZE", "50")),
    "graph_neo4j_connect_timeout_seconds": float(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_CONNECT_TIMEOUT_SECONDS", "15")),
    "graph_neo4j_retry_timeout_seconds": float(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_RETRY_TIMEOUT_SECONDS", "30")),
    "graph_neo4j_round_batch_size": int(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_ROUND_BATCH_SIZE", "1")),
    "graph_neo4j_writeback_enabled": os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_WRITEBACK_ENABLED", "false").lower() == "true",
    "graph_ontology_model": os.getenv("MARKET_PREDICTION_GRAPH_ONTOLOGY_MODEL"),
    "graph_llm_timeout_seconds": int(os.getenv("MARKET_PREDICTION_GRAPH_LLM_TIMEOUT_SECONDS", "20")),
    "llm_provider": os.getenv("MARKET_PREDICTION_LLM_PROVIDER", "openrouter"),
    "llm_model": os.getenv("MARKET_PREDICTION_LLM_MODEL", "moonshotai/kimi-k2.5"),
    "llm_api_key": os.getenv(
        "MARKET_PREDICTION_LLM_API_KEY",
        os.getenv("OPENROUTER_API_KEY", ""),
    ),
    "llm_api_base_url": os.getenv("MARKET_PREDICTION_LLM_API_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"),
    "llm_api_timeout_seconds": int(os.getenv("MARKET_PREDICTION_LLM_API_TIMEOUT_SECONDS", "90")),
    "llm_api_max_attempts": int(os.getenv("MARKET_PREDICTION_LLM_API_MAX_ATTEMPTS", "4")),
    "llm_api_retry_backoff_seconds": float(os.getenv("MARKET_PREDICTION_LLM_API_RETRY_BACKOFF_SECONDS", "3")),
    "llm_api_referrer": os.getenv("MARKET_PREDICTION_LLM_API_REFERRER", "http://localhost"),
    "post_finalize_timeout_seconds": float(os.getenv("MARKET_PREDICTION_POST_FINALIZE_TIMEOUT_SECONDS", "5")),
    "swarm_enabled": os.getenv("MARKET_PREDICTION_SWARM_ENABLED", "true").lower() == "true",
    "swarm_persona_count": int(os.getenv("MARKET_PREDICTION_SWARM_PERSONA_COUNT", str(SWARM_DEFAULT_PERSONA_COUNT))),
    "swarm_dynamic_agent_count": os.getenv("MARKET_PREDICTION_SWARM_DYNAMIC_AGENT_COUNT"),
    "swarm_rounds": int(os.getenv("MARKET_PREDICTION_SWARM_ROUNDS", str(SWARM_DEFAULT_ROUNDS))),
    "swarm_agents_per_round_min": int(os.getenv("MARKET_PREDICTION_SWARM_AGENTS_PER_ROUND_MIN", str(SWARM_DEFAULT_AGENTS_PER_ROUND_MIN))),
    "swarm_agents_per_round_max": int(os.getenv("MARKET_PREDICTION_SWARM_AGENTS_PER_ROUND_MAX", str(SWARM_DEFAULT_AGENTS_PER_ROUND_MAX))),
    "swarm_required_coverage_buckets": ["macro", "market", "sentiment", "policy"],
    "swarm_diversity_min_buckets_per_round": int(os.getenv("MARKET_PREDICTION_SWARM_DIVERSITY_MIN_BUCKETS_PER_ROUND", "4")),
    "swarm_seed_posts": int(os.getenv("MARKET_PREDICTION_SWARM_SEED_POSTS", "3")),
    "swarm_llm_timeout_seconds": int(os.getenv("MARKET_PREDICTION_SWARM_LLM_TIMEOUT_SECONDS", "10")),
    "swarm_parallel_enabled": os.getenv("MARKET_PREDICTION_SWARM_PARALLEL_ENABLED", "true").lower() == "true",
    "swarm_parallel_workers": int(os.getenv("MARKET_PREDICTION_SWARM_PARALLEL_WORKERS", "0")),
    "swarm_parallel_worker_cap": int(os.getenv("MARKET_PREDICTION_SWARM_PARALLEL_WORKER_CAP", "0")),
    "swarm_background_enabled": os.getenv("MARKET_PREDICTION_SWARM_BACKGROUND_ENABLED", "true").lower() == "true",
    "swarm_primary_agent_count": int(os.getenv("MARKET_PREDICTION_SWARM_PRIMARY_AGENT_COUNT", "4")),
    "swarm_secondary_agent_count": int(os.getenv("MARKET_PREDICTION_SWARM_SECONDARY_AGENT_COUNT", "4")),
    "swarm_background_agent_count": int(os.getenv("MARKET_PREDICTION_SWARM_BACKGROUND_AGENT_COUNT", "0")),
    "swarm_primary_max_completion_tokens": int(os.getenv("MARKET_PREDICTION_SWARM_PRIMARY_MAX_COMPLETION_TOKENS", "96")),
    "swarm_secondary_max_completion_tokens": int(os.getenv("MARKET_PREDICTION_SWARM_SECONDARY_MAX_COMPLETION_TOKENS", "48")),
    "swarm_secondary_prompt_style": os.getenv("MARKET_PREDICTION_SWARM_SECONDARY_PROMPT_STYLE", "compressed"),
    "swarm_enable_batching": os.getenv("MARKET_PREDICTION_SWARM_ENABLE_BATCHING", "false").lower() == "true",
    "swarm_persist_round_logs": os.getenv("MARKET_PREDICTION_SWARM_PERSIST_ROUND_LOGS", "true").lower() == "true",
    "swarm_max_influence": float(os.getenv("MARKET_PREDICTION_SWARM_MAX_INFLUENCE", "0.12")),
    "swarm_random_seed": int(os.getenv("MARKET_PREDICTION_SWARM_RANDOM_SEED", "42")),
    "swarm_start_hour": int(os.getenv("MARKET_PREDICTION_SWARM_START_HOUR", "8")),
    "swarm_minutes_per_round": int(os.getenv("MARKET_PREDICTION_SWARM_MINUTES_PER_ROUND", "120")),
    "swarm_peak_hours": [9, 10, 11, 14, 15],
    "swarm_offpeak_hours": [6, 7, 12, 13, 16, 17, 18],
    "swarm_peak_activity_multiplier": float(os.getenv("MARKET_PREDICTION_SWARM_PEAK_ACTIVITY_MULTIPLIER", "1.2")),
    "swarm_offpeak_activity_multiplier": float(os.getenv("MARKET_PREDICTION_SWARM_OFFPEAK_ACTIVITY_MULTIPLIER", "0.78")),
    "swarm_memory_agent_action_cap": int(os.getenv("MARKET_PREDICTION_SWARM_MEMORY_AGENT_ACTION_CAP", str(SWARM_DEFAULT_MEMORY_AGENT_ACTIONS))),
    "swarm_memory_shared_round_cap": int(os.getenv("MARKET_PREDICTION_SWARM_MEMORY_SHARED_ROUND_CAP", str(SWARM_DEFAULT_MEMORY_SHARED_ROUNDS))),
    "swarm_memory_community_reference_cap": int(os.getenv("MARKET_PREDICTION_SWARM_MEMORY_COMMUNITY_REFERENCE_CAP", str(SWARM_DEFAULT_MEMORY_COMMUNITY_REFERENCES))),
    "swarm_shared_reference_cap": int(os.getenv("MARKET_PREDICTION_SWARM_SHARED_REFERENCE_CAP", str(SWARM_DEFAULT_SHARED_REFERENCE_CAP))),
    "swarm_decision_trace_cap": int(os.getenv("MARKET_PREDICTION_SWARM_DECISION_TRACE_CAP", str(SWARM_DEFAULT_DECISION_TRACE_CAP))),
    "swarm_event_history_cap": int(os.getenv("MARKET_PREDICTION_SWARM_EVENT_HISTORY_CAP", str(SWARM_DEFAULT_EVENT_HISTORY_CAP))),
    "swarm_graph_delta_cap": int(os.getenv("MARKET_PREDICTION_SWARM_GRAPH_DELTA_CAP", str(SWARM_DEFAULT_GRAPH_DELTA_CAP))),
    "swarm_active_history_cap": int(os.getenv("MARKET_PREDICTION_SWARM_ACTIVE_HISTORY_CAP", str(SWARM_DEFAULT_ACTIVE_HISTORY_CAP))),
    "swarm_prompt_shared_round_cap": int(os.getenv("MARKET_PREDICTION_SWARM_PROMPT_SHARED_ROUND_CAP", str(SWARM_DEFAULT_PROMPT_SHARED_ROUNDS))),
    "swarm_prompt_agent_action_cap": int(os.getenv("MARKET_PREDICTION_SWARM_PROMPT_AGENT_ACTION_CAP", str(SWARM_DEFAULT_PROMPT_AGENT_ACTIONS))),
    "swarm_derived_min_participation": float(os.getenv("MARKET_PREDICTION_SWARM_DERIVED_MIN_PARTICIPATION", str(SWARM_DEFAULT_DERIVED_MIN_PARTICIPATION))),
    "swarm_derived_consensus_threshold": float(os.getenv("MARKET_PREDICTION_SWARM_DERIVED_CONSENSUS_THRESHOLD", str(SWARM_DEFAULT_DERIVED_CONSENSUS_THRESHOLD))),
    "swarm_derived_conflict_threshold": float(os.getenv("MARKET_PREDICTION_SWARM_DERIVED_CONFLICT_THRESHOLD", str(SWARM_DEFAULT_DERIVED_CONFLICT_THRESHOLD))),
    "enable_sector_scrape": os.getenv("MARKET_PREDICTION_ENABLE_SECTOR_SCRAPE", "false").lower() == "true",
    "timezone": "America/New_York",
    "default_target": "S&P 500",
    "default_ticker": "^GSPC",
    "source_limit": 5,
    "history_lookback_days": 180,
    "bootstrap_history_years": 2,
    "minimum_history_rows": 45,
    "neutral_band_base": 0.0025,
    "neutral_band_volatility_multiplier": 0.45,
    "freshness_threshold_hours": {
        "market": 48,
        "macro": 72,
        "news": 18,
        "events": 72,
        "sentiment": 18,
    },
    "market_symbols": {
        "S&P 500": "^GSPC",
        "NASDAQ 100": "^NDX",
        "DOW JONES": "^DJI",
        "VIX": "^VIX",
        "US 10 YR TREASURY": "^TNX",
        "US 2 YR TREASURY": "^IRX",
        "US 30 YR TREASURY": "^TYX",
        "DXY": "DX-Y.NYB",
        "WTI CRUDE OIL": "CL=F",
        "GOLD": "GC=F",
        "SILVER": "SI=F",
        "COPPER": "HG=F",
        "NATURAL GAS": "NG=F",
        "HIGH YIELD CREDIT": "HYG",
        "INVESTMENT GRADE CREDIT": "LQD",
        "BTC": "BTC-USD",
        "RUSSELL 2000": "^RUT",
    },
    "sector_symbols": {
        "MATERIALS": "XLB",
        "ENERGY": "XLE",
        "FINANCIALS": "XLF",
        "HEALTH CARE": "XLV",
        "INDUSTRIALS": "XLI",
        "CONSUMER STAPLES": "XLP",
        "TECHNOLOGY": "XLK",
        "UTILITIES": "XLU",
        "REAL ESTATE": "XLRE",
        "CONSUMER DISCRETIONARY": "XLY",
        "COMMUNICATION SERVICES": "XLC",
    },
    "macro_series": {
        "FEDFUNDS": {"display_name": "Fed Funds Rate", "frequency": "monthly"},
        "CPIAUCSL": {"display_name": "CPI", "frequency": "monthly"},
        "CPILFESL": {"display_name": "Core CPI", "frequency": "monthly"},
        "PPIACO": {"display_name": "PPI", "frequency": "monthly"},
        "UNRATE": {"display_name": "Unemployment Rate", "frequency": "monthly"},
        "PAYEMS": {"display_name": "Nonfarm Payrolls", "frequency": "monthly"},
        "UMCSENT": {"display_name": "Consumer Sentiment", "frequency": "monthly"},
        "RSAFS": {"display_name": "Retail Sales", "frequency": "monthly"},
        "HOUST": {"display_name": "Housing Starts", "frequency": "monthly"},
        "ICSA": {"display_name": "Initial Claims", "frequency": "weekly"},
        "DGS2": {"display_name": "2Y Treasury Constant Maturity", "frequency": "daily"},
        "DGS10": {"display_name": "10Y Treasury Constant Maturity", "frequency": "daily"},
        "DGS30": {"display_name": "30Y Treasury Constant Maturity", "frequency": "daily"},
        "T10YIE": {"display_name": "10Y Breakeven Inflation", "frequency": "daily"},
    },
    "rss_queries": {
        "economic": [
            "Federal Reserve interest rates",
            "US inflation economy",
        ],
        "political": [
            "US politics markets",
            "geopolitics oil prices",
        ],
        "social": [
            "wall street sentiment stocks",
            "retail investor sentiment",
        ],
    },
}


def _default_swarm_dynamic_agent_count(persona_count: int) -> int:
    return max(1, persona_count // 4)


def _normalize_swarm_config(config: dict) -> dict:
    persona_raw = config.get("swarm_persona_count")
    rounds_raw = config.get("swarm_rounds")
    min_active_raw = config.get("swarm_agents_per_round_min")
    max_active_raw = config.get("swarm_agents_per_round_max")
    persona_count = SWARM_DEFAULT_PERSONA_COUNT if persona_raw in (None, "") else int(persona_raw)
    rounds = SWARM_DEFAULT_ROUNDS if rounds_raw in (None, "") else int(rounds_raw)
    min_active_agents = SWARM_DEFAULT_AGENTS_PER_ROUND_MIN if min_active_raw in (None, "") else int(min_active_raw)
    max_active_agents = SWARM_DEFAULT_AGENTS_PER_ROUND_MAX if max_active_raw in (None, "") else int(max_active_raw)
    diversity_min_buckets = int(config.get("swarm_diversity_min_buckets_per_round", 4) or 4)
    primary_agent_count = int(config.get("swarm_primary_agent_count", 4) or 4)
    secondary_agent_count = int(config.get("swarm_secondary_agent_count", 4) or 4)
    dynamic_raw = config.get("swarm_dynamic_agent_count")
    dynamic_agent_count = (
        _default_swarm_dynamic_agent_count(persona_count)
        if dynamic_raw in (None, "")
        else int(dynamic_raw)
    )

    if rounds < 1:
        raise ValueError("swarm_rounds must be >= 1.")
    if min_active_agents > max_active_agents:
        raise ValueError("swarm_agents_per_round_min must be <= swarm_agents_per_round_max.")
    if persona_count < max_active_agents:
        raise ValueError("swarm_persona_count must be >= swarm_agents_per_round_max.")
    if dynamic_agent_count < 0:
        raise ValueError("swarm_dynamic_agent_count must be >= 0.")
    if diversity_min_buckets < 1:
        raise ValueError("swarm_diversity_min_buckets_per_round must be >= 1.")
    if primary_agent_count < 1:
        raise ValueError("swarm_primary_agent_count must be >= 1.")
    if secondary_agent_count < 0:
        raise ValueError("swarm_secondary_agent_count must be >= 0.")
    for key in (
        "swarm_memory_agent_action_cap",
        "swarm_memory_shared_round_cap",
        "swarm_memory_community_reference_cap",
        "swarm_shared_reference_cap",
        "swarm_decision_trace_cap",
        "swarm_event_history_cap",
        "swarm_graph_delta_cap",
        "swarm_active_history_cap",
        "swarm_prompt_shared_round_cap",
        "swarm_prompt_agent_action_cap",
    ):
        if int(config.get(key, 0) or 0) < 1:
            raise ValueError(f"{key} must be >= 1.")
    for key in (
        "swarm_derived_min_participation",
        "swarm_derived_consensus_threshold",
        "swarm_derived_conflict_threshold",
    ):
        value = float(config.get(key, 0.0) or 0.0)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{key} must be between 0 and 1.")

    normalized = deepcopy(config)
    normalized["swarm_persona_count"] = persona_count
    normalized["swarm_rounds"] = rounds
    normalized["swarm_agents_per_round_min"] = min_active_agents
    normalized["swarm_agents_per_round_max"] = max_active_agents
    normalized["swarm_dynamic_agent_count"] = dynamic_agent_count
    normalized["swarm_diversity_min_buckets_per_round"] = diversity_min_buckets
    normalized["swarm_primary_agent_count"] = primary_agent_count
    normalized["swarm_secondary_agent_count"] = secondary_agent_count
    normalized["swarm_memory_agent_action_cap"] = int(config.get("swarm_memory_agent_action_cap", SWARM_DEFAULT_MEMORY_AGENT_ACTIONS) or SWARM_DEFAULT_MEMORY_AGENT_ACTIONS)
    normalized["swarm_memory_shared_round_cap"] = int(config.get("swarm_memory_shared_round_cap", SWARM_DEFAULT_MEMORY_SHARED_ROUNDS) or SWARM_DEFAULT_MEMORY_SHARED_ROUNDS)
    normalized["swarm_memory_community_reference_cap"] = int(config.get("swarm_memory_community_reference_cap", SWARM_DEFAULT_MEMORY_COMMUNITY_REFERENCES) or SWARM_DEFAULT_MEMORY_COMMUNITY_REFERENCES)
    normalized["swarm_shared_reference_cap"] = int(config.get("swarm_shared_reference_cap", SWARM_DEFAULT_SHARED_REFERENCE_CAP) or SWARM_DEFAULT_SHARED_REFERENCE_CAP)
    normalized["swarm_decision_trace_cap"] = int(config.get("swarm_decision_trace_cap", SWARM_DEFAULT_DECISION_TRACE_CAP) or SWARM_DEFAULT_DECISION_TRACE_CAP)
    normalized["swarm_event_history_cap"] = int(config.get("swarm_event_history_cap", SWARM_DEFAULT_EVENT_HISTORY_CAP) or SWARM_DEFAULT_EVENT_HISTORY_CAP)
    normalized["swarm_graph_delta_cap"] = int(config.get("swarm_graph_delta_cap", SWARM_DEFAULT_GRAPH_DELTA_CAP) or SWARM_DEFAULT_GRAPH_DELTA_CAP)
    normalized["swarm_active_history_cap"] = int(config.get("swarm_active_history_cap", SWARM_DEFAULT_ACTIVE_HISTORY_CAP) or SWARM_DEFAULT_ACTIVE_HISTORY_CAP)
    normalized["swarm_prompt_shared_round_cap"] = int(config.get("swarm_prompt_shared_round_cap", SWARM_DEFAULT_PROMPT_SHARED_ROUNDS) or SWARM_DEFAULT_PROMPT_SHARED_ROUNDS)
    normalized["swarm_prompt_agent_action_cap"] = int(config.get("swarm_prompt_agent_action_cap", SWARM_DEFAULT_PROMPT_AGENT_ACTIONS) or SWARM_DEFAULT_PROMPT_AGENT_ACTIONS)
    normalized["swarm_derived_min_participation"] = float(config.get("swarm_derived_min_participation", SWARM_DEFAULT_DERIVED_MIN_PARTICIPATION) or SWARM_DEFAULT_DERIVED_MIN_PARTICIPATION)
    normalized["swarm_derived_consensus_threshold"] = float(config.get("swarm_derived_consensus_threshold", SWARM_DEFAULT_DERIVED_CONSENSUS_THRESHOLD) or SWARM_DEFAULT_DERIVED_CONSENSUS_THRESHOLD)
    normalized["swarm_derived_conflict_threshold"] = float(config.get("swarm_derived_conflict_threshold", SWARM_DEFAULT_DERIVED_CONFLICT_THRESHOLD) or SWARM_DEFAULT_DERIVED_CONFLICT_THRESHOLD)
    return normalized


def load_config(overrides: dict | None = None) -> dict:
    config = deepcopy(DEFAULT_CONFIG)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                config[key] = value
    return _normalize_swarm_config(config)


def list_tracked_instruments(config: dict) -> list[dict]:
    instruments = []
    for name, symbol in config.get("market_symbols", {}).items():
        instruments.append(
            {
                "symbol": symbol,
                "display_name": name,
                "asset_class": "market",
                "category": "broad_market",
            }
        )
    for name, symbol in config.get("sector_symbols", {}).items():
        instruments.append(
            {
                "symbol": symbol,
                "display_name": name,
                "asset_class": "sector",
                "category": "sector_etf",
            }
        )
    for series_name, meta in config.get("macro_series", {}).items():
        instruments.append(
            {
                "symbol": series_name,
                "display_name": meta.get("display_name", series_name),
                "asset_class": "macro",
                "category": meta.get("frequency", "macro"),
            }
        )
    return instruments
