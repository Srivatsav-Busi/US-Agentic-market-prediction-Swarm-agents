from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .graph_store import Neo4jGraphStore, neo4j_aura_config_from_config, project_simulation_graph_deltas
from .simulation_state import normalize_simulation_state
from .swarm_simulation import SimulationEnvironment, SwarmRoundResult, SwarmSimulationResult, run_swarm_from_environment


@dataclass
class SimulationExecutionOptions:
    execution_mode: str = "sync"
    parallel_workers: int = 0
    persist_round_logs: bool = True
    base_results_dir: str = ""
    graph_writeback_enabled: bool = False
    retry_budget: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationRunState:
    simulation_id: str
    status: str
    mode: str
    started_at: str
    updated_at: str
    completed_at: str | None = None
    base_run_stem: str = ""
    prediction_date: str = ""
    target: str = ""
    current_round: int = 0
    total_rounds: int = 0
    result_path: str = ""
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SimulationRunner:
    def __init__(self, results_root: str | Path):
        self.results_root = Path(results_root)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._worker_pool: ThreadPoolExecutor | None = None

    def _ensure_worker_pool(self, worker_count: int) -> ThreadPoolExecutor:
        if self._worker_pool is None:
            self._worker_pool = ThreadPoolExecutor(max_workers=max(1, worker_count), thread_name_prefix="simulation-worker")
        return self._worker_pool

    def run_sync(
        self,
        *,
        simulation_id: str,
        environment: SimulationEnvironment,
        items: list,
        snapshot: dict,
        features: list,
        llm_client,
        config: dict,
        target: str,
        state_path: str | Path,
        actions_log_path: str | Path,
        result_path: str | Path = "",
        mode: str = "sync",
        persist_round_logs: bool = True,
        progress_callback: Callable[[SimulationRunState], None] | None = None,
    ) -> SwarmSimulationResult:
        state_path = Path(state_path)
        actions_log_path = Path(actions_log_path)
        result_path = Path(result_path) if result_path else Path("")
        state_path.parent.mkdir(parents=True, exist_ok=True)
        actions_log_path.parent.mkdir(parents=True, exist_ok=True)
        if actions_log_path.exists():
            actions_log_path.unlink()
        state_trace_path = state_path.parent / "simulation_state_trace.jsonl"
        simulation_state_path = state_path.parent / "simulation_state.json"
        if state_trace_path.exists():
            state_trace_path.unlink()
        if simulation_state_path.exists():
            simulation_state_path.unlink()

        state = self._make_state(
            simulation_id=simulation_id,
            environment=environment,
            status="running",
            mode=mode,
            result_path=str(result_path) if str(result_path) else "",
        )
        self._write_state(state_path=state_path, state=state, progress_callback=progress_callback)
        self._append_state_trace(state_trace_path, self._make_trace_state(state=state, event="started"))

        def on_round(round_result: SwarmRoundResult) -> None:
            if persist_round_logs:
                self._append_round_actions(actions_log_path, round_result)
            state.current_round = round_result.round_index + 1
            state.updated_at = _iso_now()
            self._write_state(state_path=state_path, state=state, progress_callback=progress_callback)
            self._append_state_trace(
                state_trace_path,
                self._make_trace_state(state=state, event="round_completed", round_result=round_result),
            )

        try:
            swarm_result = run_swarm_from_environment(
                environment=environment,
                items=items,
                snapshot=snapshot,
                features=features,
                llm_client=llm_client,
                config=config,
                target=target,
                round_callback=on_round,
            )
        except Exception as exc:
            state.status = "failed"
            state.error_message = str(exc)
            state.updated_at = _iso_now()
            state.completed_at = state.updated_at
            self._write_state(state_path=state_path, state=state, progress_callback=progress_callback)
            raise

        state.status = "complete"
        state.updated_at = _iso_now()
        state.completed_at = state.updated_at
        state.current_round = state.total_rounds
        self._write_state(state_path=state_path, state=state, progress_callback=progress_callback)
        self._append_state_trace(state_trace_path, self._make_trace_state(state=state, event="completed"))
        memory_snapshot_path = state_path.parent / "simulation_memory_snapshot.json"
        memory_snapshot_path.write_text(
            json.dumps(swarm_result.diagnostics.get("memory", {}), indent=2),
            encoding="utf-8",
        )
        simulation_state_path.write_text(json.dumps(normalize_simulation_state(swarm_result.simulation_state), indent=2), encoding="utf-8")
        self._project_graph_deltas_if_enabled(
            config=config,
            simulation_id=simulation_id,
            state=state,
            simulation_state=swarm_result.simulation_state,
        )
        return swarm_result

    def start_background(
        self,
        *,
        simulation_id: str,
        environment: SimulationEnvironment,
        items: list,
        snapshot: dict,
        features: list,
        llm_client,
        config: dict,
        target: str,
        state_path: str | Path,
        actions_log_path: str | Path,
        result_path: str | Path,
        result_builder: Callable[[SwarmSimulationResult], dict[str, Any]] | None = None,
        mode: str = "background",
        persist_round_logs: bool = True,
        progress_callback: Callable[[SimulationRunState], None] | None = None,
        idempotency_key: str | None = None,
        max_retries: int = 1,
    ) -> SimulationRunState:
        state_path = Path(state_path)
        result_path = Path(result_path)
        state = self._make_state(
            simulation_id=simulation_id,
            environment=environment,
            status="pending",
            mode=mode,
            result_path=str(result_path),
        )
        queue_dir = state_path.parent
        queue_dir.mkdir(parents=True, exist_ok=True)
        queue_record_path = queue_dir / "queue_job.json"
        execution_options = SimulationExecutionOptions(
            execution_mode=mode,
            parallel_workers=int(config.get("simulation_worker_pool_size", config.get("swarm_parallel_workers", 0)) or 0),
            persist_round_logs=persist_round_logs,
            base_results_dir=str(self.results_root),
            graph_writeback_enabled=bool(config.get("graph_neo4j_writeback_enabled", False)),
            retry_budget=max_retries,
        ).to_dict()
        worker_pool_size = int(config.get("simulation_worker_pool_size", 2) or 2)
        now = _iso_now()
        queue_payload = {
            "simulation_id": simulation_id,
            "status": "queued",
            "idempotency_key": idempotency_key or simulation_id,
            "retry_count": 0,
            "max_retries": max_retries,
            "retry_budget": max_retries,
            "state_path": str(state_path),
            "result_path": str(result_path),
            "queued_at": now,
            "started_at": None,
            "completed_at": None,
            "last_heartbeat_at": now,
            "worker_pool_size": worker_pool_size,
            "execution_options": execution_options,
            "current_round": 0,
            "total_rounds": environment.time_config.total_rounds,
            "latest_round_summary": "",
        }
        existing_queue_record = json.loads(queue_record_path.read_text(encoding="utf-8")) if queue_record_path.exists() else None

        def target_fn() -> None:
            attempts = 0
            while attempts <= max_retries:
                try:
                    queue_payload["status"] = "running"
                    queue_payload["retry_count"] = attempts
                    queue_payload["started_at"] = queue_payload.get("started_at") or _iso_now()
                    queue_payload["last_heartbeat_at"] = _iso_now()
                    queue_record_path.write_text(json.dumps(queue_payload, indent=2), encoding="utf-8")
                    callback = progress_callback

                    def on_progress(run_state: SimulationRunState) -> None:
                        queue_payload["current_round"] = run_state.current_round
                        queue_payload["total_rounds"] = run_state.total_rounds
                        queue_payload["last_heartbeat_at"] = _iso_now()
                        queue_payload["latest_round_summary"] = f"round {run_state.current_round}/{run_state.total_rounds}"
                        queue_record_path.write_text(json.dumps(queue_payload, indent=2), encoding="utf-8")
                        if callback is not None:
                            callback(run_state)

                    swarm_result = self.run_sync(
                        simulation_id=simulation_id,
                        environment=environment,
                        items=items,
                        snapshot=snapshot,
                        features=features,
                        llm_client=llm_client,
                        config=config,
                        target=target,
                        state_path=state_path,
                        actions_log_path=actions_log_path,
                        result_path=result_path,
                        mode=mode,
                        persist_round_logs=persist_round_logs,
                        progress_callback=on_progress,
                    )
                    if result_builder is not None:
                        payload = result_builder(swarm_result)
                        result_path.parent.mkdir(parents=True, exist_ok=True)
                        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    queue_payload["status"] = "completed"
                    queue_payload["completed_at"] = _iso_now()
                    queue_payload["last_heartbeat_at"] = queue_payload["completed_at"]
                    queue_record_path.write_text(json.dumps(queue_payload, indent=2), encoding="utf-8")
                    return
                except Exception as exc:
                    attempts += 1
                    queue_payload["status"] = "retrying" if attempts <= max_retries else "failed"
                    queue_payload["retry_count"] = attempts
                    queue_payload["error_message"] = str(exc)
                    queue_payload["last_heartbeat_at"] = _iso_now()
                    if attempts > max_retries:
                        queue_payload["completed_at"] = queue_payload["last_heartbeat_at"]
                    queue_record_path.write_text(json.dumps(queue_payload, indent=2), encoding="utf-8")
                    if attempts > max_retries:
                        return

        worker_pool = self._ensure_worker_pool(worker_pool_size)
        with self._lock:
            existing_job = self._jobs.get(simulation_id)
            if existing_job and not existing_job["future"].done():
                raise RuntimeError(f"Simulation {simulation_id} is already running.")
            if existing_queue_record and existing_queue_record.get("idempotency_key") == queue_payload["idempotency_key"]:
                payload = self.get_status(simulation_id, state_path=state_path)
                if payload is not None:
                    return SimulationRunState(**payload)
            queue_record_path.write_text(json.dumps(queue_payload, indent=2), encoding="utf-8")
            future = worker_pool.submit(target_fn)
            self._jobs[simulation_id] = {
                "future": future,
                "state_path": str(state_path),
                "result_path": str(result_path),
                "queue_record_path": str(queue_record_path),
            }
        self._write_state(state_path=state_path, state=state, progress_callback=progress_callback)
        return state

    def get_status(self, simulation_id: str, state_path: str | Path | None = None) -> dict[str, Any] | None:
        resolved_state_path = self._resolve_state_path(simulation_id=simulation_id, state_path=state_path)
        if not resolved_state_path.exists():
            return None
        return json.loads(resolved_state_path.read_text(encoding="utf-8"))

    def load_result(self, simulation_id: str, result_path: str | Path | None = None) -> dict[str, Any] | None:
        resolved_result_path = self._resolve_result_path(simulation_id=simulation_id, result_path=result_path)
        if not resolved_result_path.exists():
            return None
        return json.loads(resolved_result_path.read_text(encoding="utf-8"))

    def resolve_simulation_dir(self, simulation_id: str) -> Path:
        matches = sorted(self.results_root.glob(f"simulations/*/{simulation_id}"))
        if matches:
            return matches[-1]
        return self.results_root / "simulations" / simulation_id

    def find_latest_simulation_id(self) -> str | None:
        candidates = sorted(
            (path for path in self.results_root.glob("simulations/*/*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
        )
        if not candidates:
            return None
        return candidates[-1].name

    def load_environment(self, simulation_id: str) -> dict[str, Any] | None:
        environment_path = self.resolve_simulation_dir(simulation_id) / "simulation_environment.json"
        if not environment_path.exists():
            return None
        return json.loads(environment_path.read_text(encoding="utf-8"))

    def load_actions(self, simulation_id: str) -> list[dict[str, Any]]:
        actions_path = self.resolve_simulation_dir(simulation_id) / "actions.jsonl"
        if not actions_path.exists():
            return []
        payloads: list[dict[str, Any]] = []
        with actions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payloads.append(json.loads(line))
        return payloads

    def load_state_trace(self, simulation_id: str) -> list[dict[str, Any]]:
        trace_path = self.resolve_simulation_dir(simulation_id) / "simulation_state_trace.jsonl"
        if not trace_path.exists():
            return []
        payloads: list[dict[str, Any]] = []
        with trace_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payloads.append(json.loads(line))
        return payloads

    def load_memory_snapshot(self, simulation_id: str) -> dict[str, Any] | None:
        memory_path = self.resolve_simulation_dir(simulation_id) / "simulation_memory_snapshot.json"
        if not memory_path.exists():
            return None
        return json.loads(memory_path.read_text(encoding="utf-8"))

    def load_simulation_state(self, simulation_id: str) -> dict[str, Any] | None:
        state_path = self.resolve_simulation_dir(simulation_id) / "simulation_state.json"
        if not state_path.exists():
            return None
        return normalize_simulation_state(json.loads(state_path.read_text(encoding="utf-8")))

    def load_queue_record(self, simulation_id: str) -> dict[str, Any] | None:
        queue_path = self.resolve_simulation_dir(simulation_id) / "queue_job.json"
        if not queue_path.exists():
            return None
        return json.loads(queue_path.read_text(encoding="utf-8"))

    def load_interaction_bundle(self, simulation_id: str | None = None) -> dict[str, Any] | None:
        resolved_simulation_id = simulation_id or self.find_latest_simulation_id()
        if not resolved_simulation_id:
            return None
        simulation_dir = self.resolve_simulation_dir(resolved_simulation_id)
        if not simulation_dir.exists():
            return None
        return {
            "simulation_id": resolved_simulation_id,
            "simulation_dir": simulation_dir,
            "environment": self.load_environment(resolved_simulation_id),
            "run_state": self.get_status(resolved_simulation_id),
            "result": self.load_result(resolved_simulation_id),
            "actions": self.load_actions(resolved_simulation_id),
            "state_trace": self.load_state_trace(resolved_simulation_id),
            "memory_snapshot": self.load_memory_snapshot(resolved_simulation_id),
            "simulation_state": self.load_simulation_state(resolved_simulation_id),
            "queue_record": self.load_queue_record(resolved_simulation_id),
        }

    def load_runtime_metadata(self, simulation_id: str | None = None) -> dict[str, Any] | None:
        bundle = self.load_interaction_bundle(simulation_id)
        if bundle is None:
            return None
        run_state = dict(bundle.get("run_state") or {})
        queue_record = dict(bundle.get("queue_record") or {})
        current_round = int(queue_record.get("current_round", run_state.get("current_round", 0)) or 0)
        total_rounds = max(1, int(queue_record.get("total_rounds", run_state.get("total_rounds", 0)) or 1))
        return {
            "simulation_id": bundle["simulation_id"],
            "status": run_state.get("status"),
            "run_state": run_state,
            "queue_record": queue_record,
            "progress": {
                "current_round": current_round,
                "total_rounds": total_rounds,
                "percent_complete": round(min(100.0, (current_round / total_rounds) * 100.0), 2),
            },
        }

    def _make_state(
        self,
        *,
        simulation_id: str,
        environment: SimulationEnvironment,
        status: str,
        mode: str,
        result_path: str,
    ) -> SimulationRunState:
        now = _iso_now()
        return SimulationRunState(
            simulation_id=simulation_id,
            status=status,
            mode=mode,
            started_at=now,
            updated_at=now,
            base_run_stem=environment.base_run_stem,
            prediction_date=environment.prediction_date,
            target=environment.target,
            current_round=0,
            total_rounds=environment.time_config.total_rounds,
            result_path=result_path,
        )

    def _write_state(
        self,
        *,
        state_path: Path,
        state: SimulationRunState,
        progress_callback: Callable[[SimulationRunState], None] | None,
    ) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
        if progress_callback is not None:
            progress_callback(state)

    def _append_round_actions(self, actions_log_path: Path, round_result: SwarmRoundResult) -> None:
        payload = {
            "round_index": round_result.round_index,
            "summary": round_result.summary,
            "active_agent_ids": round_result.active_agent_ids,
            "consensus_score": round_result.consensus_score,
            "conflict_score": round_result.conflict_score,
            "actions": [action.to_dict() for action in round_result.actions],
        }
        with actions_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _append_state_trace(self, state_trace_path: Path, payload: dict[str, Any]) -> None:
        with state_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def _make_trace_state(
        self,
        *,
        state: SimulationRunState,
        event: str,
        round_result: SwarmRoundResult | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "event": event,
            "simulation_id": state.simulation_id,
            "status": state.status,
            "mode": state.mode,
            "current_round": state.current_round,
            "total_rounds": state.total_rounds,
            "updated_at": state.updated_at,
            "base_run_stem": state.base_run_stem,
            "prediction_date": state.prediction_date,
            "target": state.target,
        }
        if round_result is not None:
            payload.update(
                {
                    "round_index": round_result.round_index,
                    "active_agent_ids": round_result.active_agent_ids,
                    "summary": round_result.summary,
                    "consensus_score": round_result.consensus_score,
                    "conflict_score": round_result.conflict_score,
                    "action_count": len(round_result.actions),
                }
            )
        return payload

    def _project_graph_deltas_if_enabled(
        self,
        *,
        config: dict[str, Any],
        simulation_id: str,
        state: SimulationRunState,
        simulation_state: dict[str, Any],
    ) -> None:
        if not config.get("graph_neo4j_writeback_enabled"):
            return
        settings = neo4j_aura_config_from_config(config)
        if not settings.is_configured:
            return
        graph_deltas = list((normalize_simulation_state(simulation_state).get("graph_deltas") or []))
        if not graph_deltas:
            return
        with Neo4jGraphStore(settings) as store:
            project_simulation_graph_deltas(
                store,
                simulation_id=simulation_id,
                base_run_stem=state.base_run_stem,
                target=state.target,
                graph_deltas=graph_deltas,
                round_batch_size=int(config.get("graph_neo4j_round_batch_size", settings.round_batch_size) or settings.round_batch_size),
            )

    def _resolve_state_path(self, *, simulation_id: str, state_path: str | Path | None) -> Path:
        if state_path is not None:
            return Path(state_path)
        with self._lock:
            job = self._jobs.get(simulation_id)
        if job:
            return Path(job["state_path"])
        matches = sorted(self.results_root.glob(f"simulations/*/{simulation_id}/run_state.json"))
        if matches:
            return matches[-1]
        return self.results_root / "simulations" / simulation_id / "run_state.json"

    def _resolve_result_path(self, *, simulation_id: str, result_path: str | Path | None) -> Path:
        if result_path is not None:
            return Path(result_path)
        with self._lock:
            job = self._jobs.get(simulation_id)
        if job:
            return Path(job["result_path"])
        matches = sorted(self.results_root.glob(f"simulations/*/{simulation_id}/result.json"))
        if matches:
            return matches[-1]
        return self.results_root / "simulations" / simulation_id / "result.json"

def _iso_now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()
