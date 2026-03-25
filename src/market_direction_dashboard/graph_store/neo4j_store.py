from __future__ import annotations

from contextlib import AbstractContextManager
from inspect import signature
from time import perf_counter
from typing import Any, Callable

from .config import Neo4jAuraConfig

try:
    from neo4j import GraphDatabase
except ImportError:  # pragma: no cover
    GraphDatabase = None


class Neo4jGraphStore(AbstractContextManager["Neo4jGraphStore"]):
    def __init__(self, settings: Neo4jAuraConfig, *, driver: Any | None = None) -> None:
        self.settings = settings
        self._driver = driver
        self._metrics = {
            "write_calls": 0,
            "read_calls": 0,
            "write_failures": 0,
            "retryable_failures": 0,
            "last_write_latency_ms": 0.0,
            "last_read_latency_ms": 0.0,
        }

    def _driver_instance(self):
        if self._driver is not None:
            return self._driver
        if GraphDatabase is None:
            raise RuntimeError("Neo4j driver is not installed. Add the 'neo4j' package to enable graph operations.")
        if not self.settings.is_configured:
            raise RuntimeError("Neo4j Aura configuration is incomplete.")
        self._driver = GraphDatabase.driver(
            self.settings.uri,
            auth=(self.settings.username, self.settings.password),
            max_connection_pool_size=self.settings.max_connection_pool_size,
            connection_timeout=self.settings.connection_timeout_seconds,
            max_transaction_retry_time=self.settings.max_transaction_retry_time_seconds,
        )
        return self._driver

    def verify_connectivity(self) -> None:
        self._driver_instance().verify_connectivity()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def execute_write(self, callback: Callable[[Any], Any] | str, parameters: dict[str, Any] | None = None) -> Any:
        start = perf_counter()
        self._metrics["write_calls"] += 1
        try:
            with self._driver_instance().session(database=self.settings.database) as session:
                if callable(callback):
                    callback_parameters = parameters or {}
                    try:
                        expects_parameters = len(signature(callback).parameters) > 1
                    except (TypeError, ValueError):
                        expects_parameters = False
                    result = session.execute_write(lambda tx: callback(tx, callback_parameters) if expects_parameters else callback(tx))
                else:
                    result = session.execute_write(lambda tx: tx.run(callback, **(parameters or {})).consume())
            self._metrics["last_write_latency_ms"] = round((perf_counter() - start) * 1000.0, 3)
            return result
        except Exception:
            self._metrics["write_failures"] += 1
            self._metrics["last_write_latency_ms"] = round((perf_counter() - start) * 1000.0, 3)
            raise

    def execute_read(self, callback: Callable[[Any], Any] | str, parameters: dict[str, Any] | None = None) -> Any:
        start = perf_counter()
        self._metrics["read_calls"] += 1
        with self._driver_instance().session(database=self.settings.database) as session:
            if callable(callback):
                callback_parameters = parameters or {}
                try:
                    expects_parameters = len(signature(callback).parameters) > 1
                except (TypeError, ValueError):
                    expects_parameters = False
                result = session.execute_read(lambda tx: callback(tx, callback_parameters) if expects_parameters else callback(tx))
            else:
                result = session.execute_read(lambda tx: tx.run(callback, **(parameters or {})).data())
        self._metrics["last_read_latency_ms"] = round((perf_counter() - start) * 1000.0, 3)
        return result

    def metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
