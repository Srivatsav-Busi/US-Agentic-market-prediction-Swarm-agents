from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_credentials_file(path_value: str | None) -> dict[str, str]:
    if not path_value:
        return {}
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        return {}

    credentials: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        credentials[key.strip()] = value.strip()
    return credentials


@dataclass(frozen=True)
class Neo4jAuraConfig:
    uri: str
    username: str
    password: str
    database: str = "neo4j"
    max_connection_pool_size: int = 50
    connection_timeout_seconds: float = 15.0
    max_transaction_retry_time_seconds: float = 30.0
    round_batch_size: int = 1
    writeback_enabled: bool = False
    enabled: bool = False

    @property
    def is_configured(self) -> bool:
        return bool(self.uri and self.username and self.password)

    @classmethod
    def from_mapping(cls, mapping: dict | None = None) -> "Neo4jAuraConfig":
        return neo4j_aura_config_from_config(mapping or {})


AuraConnectionSettings = Neo4jAuraConfig
Neo4jAuraSettings = Neo4jAuraConfig


def neo4j_aura_config_from_env(prefix: str = "NEO4J_") -> Neo4jAuraConfig:
    credentials_file = os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_CREDENTIALS_FILE") or os.getenv(f"{prefix}CREDENTIALS_FILE")
    file_values = _load_credentials_file(credentials_file)
    uri = os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_URI") or os.getenv(f"{prefix}URI") or file_values.get("NEO4J_URI", "")
    username = os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_USERNAME") or os.getenv(f"{prefix}USERNAME") or file_values.get("NEO4J_USERNAME", "")
    password = os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_PASSWORD") or os.getenv(f"{prefix}PASSWORD") or file_values.get("NEO4J_PASSWORD", "")
    database = os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_DATABASE") or os.getenv(f"{prefix}DATABASE") or file_values.get("NEO4J_DATABASE", "neo4j")
    return Neo4jAuraConfig(
        uri=uri,
        username=username,
        password=password,
        database=database,
        max_connection_pool_size=int(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_POOL_SIZE", "50")),
        connection_timeout_seconds=float(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_CONNECT_TIMEOUT_SECONDS", "15")),
        max_transaction_retry_time_seconds=float(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_RETRY_TIMEOUT_SECONDS", "30")),
        round_batch_size=int(os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_ROUND_BATCH_SIZE", "1")),
        writeback_enabled=os.getenv("MARKET_PREDICTION_GRAPH_NEO4J_WRITEBACK_ENABLED", "false").lower() == "true",
        enabled=(os.getenv("MARKET_PREDICTION_GRAPH_ENABLED", "false").lower() == "true") or bool(uri),
    )


def neo4j_aura_config_from_config(config: dict) -> Neo4jAuraConfig:
    env_config = neo4j_aura_config_from_env()
    uri = str(config.get("graph_neo4j_uri") or config.get("neo4j_aura_uri") or env_config.uri or "")
    username = str(config.get("graph_neo4j_username") or config.get("neo4j_aura_username") or env_config.username or "")
    password = str(config.get("graph_neo4j_password") or config.get("neo4j_aura_password") or env_config.password or "")
    database = str(config.get("graph_neo4j_database") or config.get("neo4j_aura_database") or env_config.database or "neo4j")
    return Neo4jAuraConfig(
        uri=uri,
        username=username,
        password=password,
        database=database,
        max_connection_pool_size=int(config.get("graph_neo4j_pool_size") or env_config.max_connection_pool_size),
        connection_timeout_seconds=float(config.get("graph_neo4j_connect_timeout_seconds") or env_config.connection_timeout_seconds),
        max_transaction_retry_time_seconds=float(config.get("graph_neo4j_retry_timeout_seconds") or env_config.max_transaction_retry_time_seconds),
        round_batch_size=int(config.get("graph_neo4j_round_batch_size") or env_config.round_batch_size),
        writeback_enabled=bool(config.get("graph_neo4j_writeback_enabled", env_config.writeback_enabled)),
        enabled=bool(config.get("graph_enabled", env_config.enabled)),
    )


def load_aura_settings(config: dict | None = None, *, env: dict | None = None) -> Neo4jAuraConfig:
    _ = env  # Environment is resolved from process env for compatibility.
    return neo4j_aura_config_from_config(config or {})


def load_neo4j_aura_settings(config: dict | None = None, *, env: dict | None = None) -> Neo4jAuraConfig:
    return load_aura_settings(config, env=env)
