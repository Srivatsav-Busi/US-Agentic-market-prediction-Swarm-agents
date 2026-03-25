from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from .config import load_config
from .domain import KnowledgeGraph, knowledge_graph_from_serialized
from .graph_store import Neo4jGraphRepository, Neo4jGraphStore, neo4j_aura_config_from_config
from .llm_clients import OpenRouterClient, create_llm_client, with_timeout
from .storage.db import create_schema, database_session
from .storage.models import (
    DailyPredictionRunModel,
    FeatureSnapshotModel,
    GraphBuildTaskRecord,
    GraphProjectRecord,
    GraphSnapshotRecord,
    NewsEvidenceModel,
    ProjectedPathModel,
    SectorOutlookModel,
)
from .storage.repositories import MarketRepository


DEFAULT_ONTOLOGY = {
    "entity_types": [
        {
            "name": "PredictionRun",
            "description": "A single published market prediction run.",
            "properties": ["run_id", "prediction_date", "prediction_label", "run_health", "target"],
        },
        {
            "name": "Instrument",
            "description": "A market or sector instrument referenced by the run.",
            "properties": ["instrument", "category", "latest_value", "pct_change"],
        },
        {
            "name": "Evidence",
            "description": "A news or market evidence item used during reasoning.",
            "properties": ["evidence_id", "source", "category", "direction", "impact_score", "credibility_score"],
        },
        {
            "name": "SignalFeature",
            "description": "A structured feature derived from market or evidence inputs.",
            "properties": ["feature_name", "direction", "strength", "category"],
        },
        {
            "name": "SourceReport",
            "description": "A source-specific reasoning report.",
            "properties": ["source", "score", "source_confidence", "source_regime_fit"],
        },
        {
            "name": "CategoryReport",
            "description": "A desk-level market reasoning report.",
            "properties": ["category", "score", "confidence", "dominant_regime_label"],
        },
        {
            "name": "SwarmPersona",
            "description": "A simulation persona participating in the swarm.",
            "properties": ["agent_id", "archetype", "stance_bias", "influence_weight"],
        },
        {
            "name": "SwarmAction",
            "description": "A post or interaction produced during a swarm round.",
            "properties": ["action_id", "action_type", "direction", "strength", "round_index"],
        },
        {
            "name": "ProjectionPoint",
            "description": "A forecast point on the market projection path.",
            "properties": ["forecast_date", "horizon_day", "scenario_type", "predicted_price", "predicted_return"],
        },
        {
            "name": "SectorOutlook",
            "description": "A sector-level ranking and recommendation row.",
            "properties": ["sector_symbol", "recommendation_label", "ranking_score", "expected_return_30d"],
        },
    ],
    "edge_types": [
        {"name": "USES_EVIDENCE", "description": "Run or report consumes an evidence item.", "source_targets": [["PredictionRun", "Evidence"], ["SourceReport", "Evidence"], ["CategoryReport", "Evidence"]]},
        {"name": "GENERATED_FEATURE", "description": "Evidence or instrument generated a feature.", "source_targets": [["Evidence", "SignalFeature"], ["Instrument", "SignalFeature"]]},
        {"name": "INFLUENCES_RUN", "description": "Feature or report influences the run outcome.", "source_targets": [["SignalFeature", "PredictionRun"], ["SourceReport", "PredictionRun"], ["CategoryReport", "PredictionRun"]]},
        {"name": "REFERENCES_INSTRUMENT", "description": "An entity references an instrument.", "source_targets": [["Evidence", "Instrument"], ["ProjectionPoint", "Instrument"], ["PredictionRun", "Instrument"]]},
        {"name": "AUTHORED", "description": "A persona authored a swarm action.", "source_targets": [["SwarmPersona", "SwarmAction"]]},
        {"name": "MENTIONS", "description": "A swarm action mentions evidence or features.", "source_targets": [["SwarmAction", "Evidence"], ["SwarmAction", "SignalFeature"]]},
        {"name": "PROJECTS_TO", "description": "A run projects to forecast points and sector views.", "source_targets": [["PredictionRun", "ProjectionPoint"], ["PredictionRun", "SectorOutlook"]]},
    ],
    "analysis_summary": "Finance run graph centered on explainability across evidence, structured features, agent reports, swarm behavior, and forecast outputs.",
}

REQUIRED_ENTITY_TYPES = {item["name"] for item in DEFAULT_ONTOLOGY["entity_types"]}
REQUIRED_EDGE_TYPES = {item["name"] for item in DEFAULT_ONTOLOGY["edge_types"]}
RESERVED_PROPERTY_NAMES = {"id", "node_id", "edge_id", "project_id", "graph_backend"}


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


class GraphTaskManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}

    def create(self, task_id: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._tasks[task_id] = payload.copy()

    def update(self, task_id: str, **updates: Any) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.update(updates)
            return task.copy()

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return task.copy() if task else None


GRAPH_TASK_MANAGER = GraphTaskManager()


@dataclass(frozen=True)
class GraphBuildContext:
    project_id: str
    task_id: str
    run_id: str
    artifact_path: str


def _start_graph_worker(*, config: dict, project_id: str, task_id: str, run_id: str, artifact_path: str) -> None:
    worker = threading.Thread(
        target=_build_graph_background,
        kwargs={
            "config": config,
            "project_id": project_id,
            "task_id": task_id,
            "run_id": run_id,
            "artifact_path": artifact_path,
        },
        daemon=True,
    )
    worker.start()


def build_graph_for_run(
    *,
    run_id: str,
    config: dict | None = None,
    results_dir: str | Path | None = None,
    artifact_path: str | Path | None = None,
) -> GraphBuildContext:
    config = load_config(config)
    results_root = Path(results_dir or config["results_dir"])
    artifact = Path(artifact_path) if artifact_path else find_artifact_for_run(results_root, run_id)
    if artifact is None:
        raise FileNotFoundError(f"No result artifact found for run_id={run_id}")

    task_id = f"graph-task-{uuid4().hex[:12]}"
    now = utc_now_iso()
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        existing = repo.get_graph_project(run_id=run_id)
        project_id = existing["project_id"] if existing else f"graph-project-{uuid4().hex[:12]}"
        repo.upsert_graph_project(
            GraphProjectRecord(
                project_id=project_id,
                run_id=run_id,
                status="queued",
                graph_backend=str(config.get("graph_backend", "neo4j")),
                backend_graph_ref=existing["backend_graph_ref"] if existing else None,
                source_artifact_path=str(artifact),
                ontology_json=existing["ontology_json"] if existing else None,
                created_at=existing["created_at"] if existing else now,
                updated_at=now,
            )
        )
        repo.upsert_graph_build_task(
            GraphBuildTaskRecord(
                task_id=task_id,
                project_id=project_id,
                status="queued",
                progress_stage="queued",
                started_at=now,
                stage_started_at=now,
                last_progress_at=now,
            )
        )

    GRAPH_TASK_MANAGER.create(
        task_id,
        {
            "task_id": task_id,
            "project_id": project_id,
            "run_id": run_id,
            "status": "queued",
            "progress_stage": "queued",
            "progress_detail": "Queued for background graph build",
            "error_message": None,
            "started_at": now,
            "stage_started_at": now,
            "last_progress_at": now,
            "telemetry": None,
            "completed_at": None,
        },
    )
    _start_graph_worker(
        config=config,
        project_id=project_id,
        task_id=task_id,
        run_id=run_id,
        artifact_path=str(artifact),
    )
    return GraphBuildContext(project_id=project_id, task_id=task_id, run_id=run_id, artifact_path=str(artifact))


def maybe_queue_graph_build(
    *,
    result: dict,
    config: dict,
    output_dir: str | Path,
    artifact_path: str | Path,
) -> dict | None:
    if not config.get("graph_enabled"):
        return None
    run_id = str(result.get("run_id") or "").strip()
    if not run_id:
        return None
    context = build_graph_for_run(run_id=run_id, config=config, results_dir=output_dir, artifact_path=artifact_path)
    return {
        "graph_project_id": context.project_id,
        "graph_task_id": context.task_id,
        "graph_status": "queued",
        "graph_summary": {
            "project_id": context.project_id,
            "task_id": context.task_id,
            "status": "queued",
            "node_count": 0,
            "edge_count": 0,
        },
    }


def resume_pending_graph_builds(*, config: dict | None = None, results_dir: str | Path | None = None) -> list[GraphBuildContext]:
    config = load_config(config)
    results_root = Path(results_dir or config["results_dir"])
    resumed: list[GraphBuildContext] = []
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        for project in repo.list_graph_projects():
            if project["status"] not in {"queued", "running"}:
                continue
            snapshot = repo.get_graph_snapshot(project_id=project["project_id"])
            if snapshot is not None:
                continue
            task = repo.get_latest_graph_task(project["project_id"])
            if task is None:
                continue
            if GRAPH_TASK_MANAGER.get(task["task_id"]):
                continue
            artifact_path = project.get("source_artifact_path") or ""
            if not artifact_path:
                artifact = find_artifact_for_run(results_root, project["run_id"])
                if artifact is None:
                    continue
                artifact_path = str(artifact)
            GRAPH_TASK_MANAGER.create(
                task["task_id"],
                {
                    "task_id": task["task_id"],
                    "project_id": project["project_id"],
                    "run_id": project["run_id"],
                    "status": "running",
                    "progress_stage": task.get("progress_stage") or "queued",
                    "progress_detail": task.get("progress_detail"),
                    "error_message": task.get("error_message"),
                    "started_at": task.get("started_at") or utc_now_iso(),
                    "stage_started_at": task.get("stage_started_at") or task.get("started_at") or utc_now_iso(),
                    "last_progress_at": task.get("last_progress_at") or task.get("started_at") or utc_now_iso(),
                    "telemetry": task.get("telemetry"),
                    "completed_at": task.get("completed_at"),
                },
            )
            _start_graph_worker(
                config=config,
                project_id=project["project_id"],
                task_id=task["task_id"],
                run_id=project["run_id"],
                artifact_path=artifact_path,
            )
            resumed.append(
                GraphBuildContext(
                    project_id=project["project_id"],
                    task_id=task["task_id"],
                    run_id=project["run_id"],
                    artifact_path=artifact_path,
                )
            )
    return resumed


def _build_graph_background(*, config: dict, project_id: str, task_id: str, run_id: str, artifact_path: str) -> None:
    try:
        _update_graph_state(
            config,
            project_id,
            task_id,
            status="running",
            progress_stage="assembling_corpus",
            progress_detail="Loading run artifact and database rows",
        )
        corpus_builder = GraphCorpusBuilder(config=config, results_dir=config["results_dir"])
        corpus = corpus_builder.build(run_id=run_id, artifact_path=artifact_path)

        _update_graph_state(
            config,
            project_id,
            task_id,
            status="running",
            progress_stage="generating_ontology",
            progress_detail="Preparing compact ontology prompt",
        )
        ontology_generator = OntologyGenerator(config)
        ontology, ontology_diagnostics = ontology_generator.generate_with_diagnostics(
            corpus,
            progress_callback=lambda detail, telemetry=None: _update_graph_state(
                config,
                project_id,
                task_id,
                status="running",
                progress_stage="generating_ontology",
                progress_detail=detail,
                telemetry=telemetry,
            ),
        )

        _update_graph_state(
            config,
            project_id,
            task_id,
            status="running",
            progress_stage="building_graph",
            progress_detail="Persisting graph payload to Neo4j",
            telemetry={"ontology": ontology_diagnostics},
        )
        graph_payload = build_graph_payload(corpus, ontology, project_id=project_id)
        backend = Neo4jGraphBuilder(config)
        backend_result = backend.build_graph(project_id=project_id, graph_payload=graph_payload)

        now = utc_now_iso()
        with database_session(config["database_url"]) as session:
            create_schema(session)
            repo = MarketRepository(session)
            project = repo.get_graph_project(project_id=project_id)
            repo.upsert_graph_project(
                GraphProjectRecord(
                    project_id=project_id,
                    run_id=run_id,
                    status="completed",
                    graph_backend=str(config.get("graph_backend", "neo4j")),
                    backend_graph_ref=backend_result["backend_graph_ref"],
                    source_artifact_path=artifact_path,
                    ontology_json=json.dumps(ontology, sort_keys=True),
                    created_at=project["created_at"] if project else now,
                    updated_at=now,
                )
            )
            repo.upsert_graph_build_task(
                GraphBuildTaskRecord(
                    task_id=task_id,
                    project_id=project_id,
                    status="completed",
                    progress_stage="completed",
                    started_at=GRAPH_TASK_MANAGER.get(task_id)["started_at"] if GRAPH_TASK_MANAGER.get(task_id) else now,
                    progress_detail="Graph build completed",
                    stage_started_at=GRAPH_TASK_MANAGER.get(task_id)["stage_started_at"] if GRAPH_TASK_MANAGER.get(task_id) else now,
                    last_progress_at=now,
                    telemetry_json=json.dumps({"ontology": ontology_diagnostics}, sort_keys=True),
                    completed_at=now,
                )
            )
            repo.upsert_graph_snapshot(
                GraphSnapshotRecord(
                    snapshot_id=f"graph-snapshot-{uuid4().hex[:12]}",
                    project_id=project_id,
                    run_id=run_id,
                    status="completed",
                    node_count=len(graph_payload["nodes"]),
                    edge_count=len(graph_payload["edges"]),
                    graph_json=json.dumps(graph_payload, sort_keys=True),
                    highlights_json=json.dumps(graph_payload.get("highlights", {}), sort_keys=True),
                    created_at=now,
                    updated_at=now,
                )
            )
        GRAPH_TASK_MANAGER.update(
            task_id,
            status="completed",
            progress_stage="completed",
            progress_detail="Graph build completed",
            last_progress_at=now,
            telemetry={"ontology": ontology_diagnostics},
            completed_at=now,
        )
    except Exception as exc:
        now = utc_now_iso()
        with database_session(config["database_url"]) as session:
            create_schema(session)
            repo = MarketRepository(session)
            project = repo.get_graph_project(project_id=project_id)
            repo.upsert_graph_project(
                GraphProjectRecord(
                    project_id=project_id,
                    run_id=run_id,
                    status="failed",
                    graph_backend=str(config.get("graph_backend", "neo4j")),
                    backend_graph_ref=project["backend_graph_ref"] if project else None,
                    source_artifact_path=artifact_path,
                    ontology_json=project["ontology_json"] if project else None,
                    created_at=project["created_at"] if project else now,
                    updated_at=now,
                )
            )
            repo.upsert_graph_build_task(
                GraphBuildTaskRecord(
                    task_id=task_id,
                    project_id=project_id,
                    status="failed",
                    progress_stage="failed",
                    progress_detail="Graph build failed",
                    error_message=str(exc),
                    started_at=GRAPH_TASK_MANAGER.get(task_id)["started_at"] if GRAPH_TASK_MANAGER.get(task_id) else now,
                    stage_started_at=GRAPH_TASK_MANAGER.get(task_id)["stage_started_at"] if GRAPH_TASK_MANAGER.get(task_id) else now,
                    last_progress_at=now,
                    telemetry_json=json.dumps((GRAPH_TASK_MANAGER.get(task_id) or {}).get("telemetry"), sort_keys=True)
                    if (GRAPH_TASK_MANAGER.get(task_id) or {}).get("telemetry") is not None
                    else None,
                    completed_at=now,
                )
            )
        GRAPH_TASK_MANAGER.update(
            task_id,
            status="failed",
            progress_stage="failed",
            progress_detail="Graph build failed",
            error_message=str(exc),
            last_progress_at=now,
            completed_at=now,
        )


def _update_graph_state(
    config: dict,
    project_id: str,
    task_id: str,
    *,
    status: str,
    progress_stage: str,
    progress_detail: str | None = None,
    telemetry: dict[str, Any] | None = None,
) -> None:
    now = utc_now_iso()
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        project = repo.get_graph_project(project_id=project_id)
        repo.upsert_graph_project(
            GraphProjectRecord(
                project_id=project_id,
                run_id=project["run_id"],
                status=status,
                graph_backend=project["graph_backend"],
                backend_graph_ref=project["backend_graph_ref"],
                source_artifact_path=project["source_artifact_path"],
                ontology_json=project["ontology_json"],
                created_at=project["created_at"],
                updated_at=now,
            )
        )
        task = repo.get_graph_task(task_id)
        stage_started_at = now if not task or task.get("progress_stage") != progress_stage else task.get("stage_started_at") or now
        merged_telemetry = _merge_graph_task_telemetry(task.get("telemetry") if task else None, telemetry)
        merged_telemetry = _append_stage_history(
            merged_telemetry,
            status=status,
            progress_stage=progress_stage,
            progress_detail=progress_detail or (task.get("progress_detail") if task else None),
            at=now,
        )
        repo.upsert_graph_build_task(
            GraphBuildTaskRecord(
                task_id=task_id,
                project_id=project_id,
                status=status,
                progress_stage=progress_stage,
                progress_detail=progress_detail or (task.get("progress_detail") if task else None),
                error_message=task["error_message"] if task else None,
                started_at=task["started_at"] if task else now,
                stage_started_at=stage_started_at,
                last_progress_at=now,
                telemetry_json=json.dumps(merged_telemetry, sort_keys=True) if merged_telemetry is not None else None,
                completed_at=task["completed_at"] if task else None,
            )
        )
    GRAPH_TASK_MANAGER.update(
        task_id,
        status=status,
        progress_stage=progress_stage,
        progress_detail=progress_detail,
        stage_started_at=stage_started_at,
        last_progress_at=now,
        telemetry=merged_telemetry,
    )


class GraphCorpusBuilder:
    def __init__(self, *, config: dict, results_dir: str | Path) -> None:
        self.config = config
        self.results_dir = Path(results_dir)

    def build(self, *, run_id: str, artifact_path: str | Path) -> dict[str, Any]:
        artifact = Path(artifact_path)
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        knowledge_graph_payload = payload.get("knowledge_graph") if isinstance(payload.get("knowledge_graph"), dict) else None
        db_rows = self._load_db_rows(run_id=run_id, payload=payload)
        return {
            "run_id": run_id,
            "artifact_path": str(artifact),
            "artifact": payload,
            "knowledge_graph": knowledge_graph_payload,
            "db_rows": db_rows,
            "corpus_summary": {
                "evidence_count": len(payload.get("sources", [])),
                "feature_count": len(payload.get("signal_features", [])),
                "source_report_count": len(payload.get("source_agent_reports", [])),
                "swarm_agent_count": len(payload.get("swarm_agents", [])),
                "swarm_round_count": len(payload.get("swarm_rounds", [])),
            },
        }

    def _load_db_rows(self, *, run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        prediction_date = payload.get("prediction_date")
        target = payload.get("target")
        rows: dict[str, Any] = {
            "prediction_run": None,
            "news_evidence": [],
            "feature_snapshots": [],
            "projected_paths": [],
            "sector_outlook": [],
        }
        with database_session(self.config["database_url"]) as session:
            create_schema(session)
            prediction = session.get(DailyPredictionRunModel, run_id)
            if prediction is not None:
                rows["prediction_run"] = {
                    "run_id": prediction.run_id,
                    "run_date": prediction.run_date,
                    "target": prediction.target,
                    "prediction_label": prediction.prediction_label,
                    "confidence": prediction.confidence,
                    "run_health": prediction.run_health,
                }
            evidence_rows = session.query(NewsEvidenceModel).filter(NewsEvidenceModel.run_date == prediction_date).all()
            rows["news_evidence"] = [
                {
                    "evidence_id": row.evidence_id,
                    "title": row.title,
                    "source": row.source,
                    "category": row.category,
                    "summary": row.summary,
                }
                for row in evidence_rows
            ]
            feature_rows = (
                session.query(FeatureSnapshotModel)
                .filter(FeatureSnapshotModel.snapshot_date == prediction_date, FeatureSnapshotModel.target_scope == target)
                .limit(50)
                .all()
            )
            rows["feature_snapshots"] = [
                {
                    "snapshot_date": row.snapshot_date,
                    "feature_name": row.feature_name,
                    "feature_value": row.feature_value,
                    "feature_group": row.feature_group,
                }
                for row in feature_rows
            ]
            projection_rows = session.query(ProjectedPathModel).filter(ProjectedPathModel.run_id == run_id).all()
            rows["projected_paths"] = [
                {
                    "forecast_date": row.forecast_date,
                    "horizon_day": row.horizon_day,
                    "scenario_type": row.scenario_type,
                    "predicted_price": row.predicted_price,
                    "predicted_return": row.predicted_return,
                }
                for row in projection_rows
            ]
            sector_rows = session.query(SectorOutlookModel).filter(SectorOutlookModel.run_id == run_id).all()
            rows["sector_outlook"] = [
                {
                    "sector_symbol": row.sector_symbol,
                    "sector_name": row.sector_name,
                    "recommendation_label": row.recommendation_label,
                    "ranking_score": row.ranking_score,
                }
                for row in sector_rows
            ]
        return rows


def _merge_graph_task_telemetry(existing: dict[str, Any] | None, updates: dict[str, Any] | None) -> dict[str, Any] | None:
    if not existing and not updates:
        return None
    merged = dict(existing or {})
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def _append_stage_history(
    telemetry: dict[str, Any] | None,
    *,
    status: str,
    progress_stage: str,
    progress_detail: str | None,
    at: str,
) -> dict[str, Any] | None:
    telemetry = dict(telemetry or {})
    history = list(telemetry.get("stage_history") or [])
    entry = {
        "status": status,
        "progress_stage": progress_stage,
        "progress_detail": progress_detail,
        "at": at,
    }
    if not history or history[-1] != entry:
        history.append(entry)
    telemetry["stage_history"] = history[-20:]
    return telemetry


class OntologyGenerator:
    def __init__(self, config: dict) -> None:
        llm_config = dict(config)
        if llm_config.get("graph_ontology_model"):
            llm_config["llm_model"] = llm_config["graph_ontology_model"]
        llm = create_llm_client(llm_config)
        llm = with_timeout(llm, int(config.get("graph_llm_timeout_seconds", 20) or 20))
        if isinstance(llm, OpenRouterClient):
            llm = replace(
                llm,
                request_group="ontology_generation",
                max_attempts=1,
                max_completion_tokens=min(600, int(config.get("graph_ontology_max_completion_tokens", 600) or 600)),
            )
        self.llm_client = llm

    def generate(self, corpus: dict[str, Any]) -> dict[str, Any]:
        ontology, _ = self.generate_with_diagnostics(corpus)
        return ontology

    def generate_with_diagnostics(
        self,
        corpus: dict[str, Any],
        progress_callback: Callable[[str, dict[str, Any] | None], None] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_prompt = json.dumps(self._build_prompt_payload(corpus), separators=(",", ":"), sort_keys=True)
        fallback = json.dumps(DEFAULT_ONTOLOGY)
        system_prompt = (
            "You design compact JSON ontologies for finance explainability graphs. "
            "Return JSON only with keys entity_types, edge_types, analysis_summary. "
            "Preserve every required type from base_schema. "
            "Add at most 2 entity types and 2 edge types. "
            "Keep analysis_summary under 240 characters. "
            "Keep each description to one short sentence and each properties list to at most 8 items."
        )
        diagnostics = {
            "timeout_seconds": int(getattr(self.llm_client, "timeout_seconds", 0) or 0),
            "max_attempts": int(getattr(self.llm_client, "max_attempts", 0) or 0),
            "max_completion_tokens": getattr(self.llm_client, "max_completion_tokens", None),
            "prompt_bytes": len(user_prompt.encode("utf-8")),
            "fallback_used": False,
            "fallback_reason": None,
        }
        if progress_callback is not None:
            progress_callback("Calling ontology LLM", {"ontology": diagnostics})

        llm_started = time.perf_counter()
        response = self.llm_client.summarize(system_prompt, user_prompt, fallback)
        diagnostics["llm_seconds"] = round(time.perf_counter() - llm_started, 3)
        diagnostics["response_bytes"] = len((response or "").encode("utf-8"))
        diagnostics["response_chars"] = len(response or "")
        diagnostics["llm"] = self.llm_client.diagnostics()

        if progress_callback is not None:
            progress_callback("Parsing ontology response", {"ontology": diagnostics})

        parse_started = time.perf_counter()
        parsed = _safe_json_loads(response, DEFAULT_ONTOLOGY)
        diagnostics["json_parse_seconds"] = round(time.perf_counter() - parse_started, 6)

        normalize_started = time.perf_counter()
        normalized = normalize_ontology(parsed)
        diagnostics["normalize_seconds"] = round(time.perf_counter() - normalize_started, 6)
        if response == fallback:
            diagnostics["fallback_used"] = True
            diagnostics["fallback_reason"] = "llm_fallback_response"
        elif parsed == DEFAULT_ONTOLOGY and response != fallback:
            diagnostics["fallback_used"] = True
            diagnostics["fallback_reason"] = "malformed_json_response"
        diagnostics["entity_type_count"] = len(normalized["entity_types"])
        diagnostics["edge_type_count"] = len(normalized["edge_types"])
        return normalized, diagnostics

    def _build_prompt_payload(self, corpus: dict[str, Any]) -> dict[str, Any]:
        artifact = corpus["artifact"]
        db_rows = corpus["db_rows"]
        return {
            "run_id": corpus["run_id"],
            "target": artifact.get("target"),
            "prediction_label": artifact.get("prediction_label"),
            "run_health": artifact.get("run_health"),
            "corpus_summary": corpus.get("corpus_summary", {}),
            "market_context": {
                "summary": str(artifact.get("summary") or "")[:800],
                "top_drivers": [str(item)[:160] for item in (artifact.get("top_drivers", []) or [])[:5]],
                "signals": [
                    {
                        "name": str(item.get("name") or "")[:80],
                        "category": item.get("category"),
                        "direction": item.get("direction"),
                        "strength": item.get("strength"),
                    }
                    for item in (artifact.get("signal_features", []) or [])[:6]
                ],
            },
            "evidence_samples": [
                {
                    "id": item.get("id"),
                    "source": item.get("source"),
                    "category": item.get("category"),
                    "direction": item.get("direction"),
                    "title": str(item.get("title") or "")[:160],
                }
                for item in (artifact.get("sources", []) or [])[:4]
            ],
            "source_report_samples": [
                {
                    "source": item.get("source"),
                    "score": item.get("score"),
                    "confidence": item.get("source_confidence"),
                    "regime_fit": item.get("source_regime_fit"),
                }
                for item in (artifact.get("source_agent_reports", []) or [])[:3]
            ],
            "db_summary": {
                "prediction_run": db_rows.get("prediction_run"),
                "news_evidence_titles": [str(item.get("title") or "")[:120] for item in db_rows.get("news_evidence", [])[:4]],
                "feature_snapshot_names": [item.get("feature_name") for item in db_rows.get("feature_snapshots", [])[:8]],
                "projection_sample": db_rows.get("projected_paths", [])[:6],
                "sector_outlook_sample": db_rows.get("sector_outlook", [])[:6],
            },
            "knowledge_graph_summary": self._summarize_knowledge_graph(corpus.get("knowledge_graph")),
            "base_schema": DEFAULT_ONTOLOGY,
        }

    @staticmethod
    def _summarize_knowledge_graph(payload: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        entities = payload.get("entities") if isinstance(payload.get("entities"), list) else []
        relationships = payload.get("relationships") if isinstance(payload.get("relationships"), list) else []
        return {
            "graph_id": payload.get("graph_id"),
            "entity_count": len(entities),
            "relationship_count": len(relationships),
            "entity_types": sorted({str(item.get("entity_type") or "") for item in entities if isinstance(item, dict) and item.get("entity_type")})[:12],
            "relationship_types": sorted(
                {str(item.get("relationship_type") or "") for item in relationships if isinstance(item, dict) and item.get("relationship_type")}
            )[:12],
        }


def _safe_json_loads(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else fallback
    except json.JSONDecodeError:
        return fallback


def normalize_ontology(raw: dict[str, Any]) -> dict[str, Any]:
    entity_types = raw.get("entity_types") if isinstance(raw.get("entity_types"), list) else []
    edge_types = raw.get("edge_types") if isinstance(raw.get("edge_types"), list) else []
    normalized_entities: list[dict[str, Any]] = []
    normalized_edges: list[dict[str, Any]] = []

    for item in entity_types[:12]:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        properties = []
        for prop in item.get("properties", []):
            prop_name = str(prop).strip()
            if not prop_name:
                continue
            if prop_name in RESERVED_PROPERTY_NAMES:
                prop_name = f"graph_{prop_name}"
            properties.append(prop_name)
        normalized_entities.append(
            {
                "name": str(item["name"]).strip(),
                "description": str(item.get("description", "")).strip(),
                "properties": sorted(set(properties)),
            }
        )

    for required in DEFAULT_ONTOLOGY["entity_types"]:
        if required["name"] not in {item["name"] for item in normalized_entities}:
            normalized_entities.append(required)

    entity_names = {item["name"] for item in normalized_entities}
    for item in edge_types[:20]:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        pairs = []
        for pair in item.get("source_targets", []):
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            source_name, target_name = str(pair[0]).strip(), str(pair[1]).strip()
            if source_name in entity_names and target_name in entity_names:
                pairs.append([source_name, target_name])
        if not pairs:
            continue
        normalized_edges.append(
            {
                "name": str(item["name"]).strip(),
                "description": str(item.get("description", "")).strip(),
                "source_targets": pairs,
            }
        )

    for required in DEFAULT_ONTOLOGY["edge_types"]:
        if required["name"] not in {item["name"] for item in normalized_edges}:
            normalized_edges.append(required)

    return {
        "entity_types": normalized_entities,
        "edge_types": normalized_edges,
        "analysis_summary": str(raw.get("analysis_summary") or DEFAULT_ONTOLOGY["analysis_summary"]).strip(),
    }


def build_graph_payload(corpus: dict[str, Any], ontology: dict[str, Any], *, project_id: str) -> dict[str, Any]:
    canonical_graph = corpus.get("knowledge_graph")
    if isinstance(canonical_graph, dict) and canonical_graph.get("entities") is not None:
        return _build_graph_payload_from_knowledge_graph(
            knowledge_graph_from_serialized(canonical_graph),
            ontology=ontology,
            project_id=project_id,
            run_id=str(corpus.get("run_id") or ""),
        )

    artifact = corpus["artifact"]
    run_id = corpus["run_id"]
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()
    seen_edges: set[str] = set()

    def add_node(node_id: str, node_type: str, label: str, **properties: Any) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(
            {
                "id": node_id,
                "type": node_type,
                "label": label,
                "group": node_type,
                "project_id": project_id,
                "properties": properties,
                "summary": str(properties.get("summary", "")),
            }
        )

    def add_edge(edge_id: str, source: str, target: str, relation: str, **properties: Any) -> None:
        if source not in seen_nodes or target not in seen_nodes or edge_id in seen_edges:
            return
        seen_edges.add(edge_id)
        edges.append(
            {
                "id": edge_id,
                "source": source,
                "target": target,
                "relation": relation,
                "project_id": project_id,
                "weight": properties.get("weight", 1.0),
                "properties": properties,
            }
        )

    run_node_id = f"run:{run_id}"
    add_node(
        run_node_id,
        "PredictionRun",
        f"{artifact.get('target', 'Market')} {artifact.get('prediction_label', 'UNKNOWN')}",
        run_id=run_id,
        target=artifact.get("target"),
        prediction_date=artifact.get("prediction_date"),
        prediction_label=artifact.get("prediction_label"),
        run_health=artifact.get("run_health"),
        summary=artifact.get("summary"),
    )

    target = artifact.get("target")
    if target:
        instrument_node_id = f"instrument:{target}"
        target_series = artifact.get("market_snapshot", {}).get("series", {}).get(target, {})
        add_node(
            instrument_node_id,
            "Instrument",
            target,
            instrument=target,
            category="target",
            latest_value=target_series.get("latest"),
            pct_change=target_series.get("pct_change"),
        )
        add_edge(f"{run_node_id}:target", run_node_id, instrument_node_id, "REFERENCES_INSTRUMENT")

    for label, series in (artifact.get("market_snapshot", {}).get("series", {}) or {}).items():
        node_id = f"instrument:{label}"
        add_node(
            node_id,
            "Instrument",
            label,
            instrument=label,
            category="market_snapshot",
            latest_value=series.get("latest"),
            pct_change=series.get("pct_change"),
            summary=f"{label} latest move {series.get('pct_change')}",
        )
        add_edge(f"{run_node_id}:instrument:{label}", run_node_id, node_id, "REFERENCES_INSTRUMENT")

    evidence_map = {}
    for item in artifact.get("sources", []) or []:
        evidence_id = str(item.get("id") or uuid4().hex[:12])
        node_id = f"evidence:{evidence_id}"
        evidence_map[evidence_id] = node_id
        add_node(
            node_id,
            "Evidence",
            item.get("title") or evidence_id,
            evidence_id=evidence_id,
            source=item.get("source"),
            category=item.get("category"),
            direction=item.get("direction"),
            impact_score=item.get("impact_score"),
            credibility_score=item.get("credibility_score"),
            summary=item.get("summary"),
        )
        add_edge(f"{run_node_id}:evidence:{evidence_id}", run_node_id, node_id, "USES_EVIDENCE")
        if item.get("instrument"):
            instrument_id = f"instrument:{item['instrument']}"
            add_node(instrument_id, "Instrument", item["instrument"], instrument=item["instrument"], category="referenced")
            add_edge(f"{node_id}:instrument:{item['instrument']}", node_id, instrument_id, "REFERENCES_INSTRUMENT")

    feature_map = {}
    for feature in artifact.get("signal_features", []) or []:
        feature_name = str(feature.get("name"))
        node_id = f"feature:{feature_name}"
        feature_map[feature_name] = node_id
        add_node(
            node_id,
            "SignalFeature",
            feature_name,
            feature_name=feature_name,
            direction=feature.get("direction"),
            strength=feature.get("strength"),
            category=feature.get("category"),
            summary=feature.get("summary"),
        )
        add_edge(f"{node_id}:run", node_id, run_node_id, "INFLUENCES_RUN", weight=feature.get("strength", 1.0))
        for evidence_id in feature.get("supporting_evidence_ids", []) or []:
            source_id = evidence_map.get(evidence_id)
            if source_id:
                add_edge(f"{source_id}:feature:{feature_name}", source_id, node_id, "GENERATED_FEATURE", weight=feature.get("strength", 1.0))

    for report in artifact.get("source_agent_reports", []) or []:
        source_name = str(report.get("source") or "unknown-source")
        node_id = f"source-report:{source_name}"
        add_node(
            node_id,
            "SourceReport",
            source_name,
            source=source_name,
            score=report.get("score"),
            source_confidence=report.get("source_confidence"),
            source_regime_fit=report.get("source_regime_fit"),
            summary=report.get("summary"),
        )
        add_edge(f"{node_id}:run", node_id, run_node_id, "INFLUENCES_RUN", weight=abs(report.get("score", 0.0)))
        for evidence_id in report.get("evidence_ids_used", []) or []:
            evidence_node = evidence_map.get(evidence_id)
            if evidence_node:
                add_edge(f"{node_id}:evidence:{evidence_id}", node_id, evidence_node, "USES_EVIDENCE")

    for category in ("economic", "political", "social", "market"):
        summary = artifact.get(f"{category}_report")
        if not summary:
            continue
        node_id = f"category-report:{category}"
        add_node(node_id, "CategoryReport", category.title(), category=category, summary=summary)
        add_edge(f"{node_id}:run", node_id, run_node_id, "INFLUENCES_RUN")

    persona_map = {}
    for persona in artifact.get("swarm_agents", []) or []:
        agent_id = persona.get("agent_id")
        if not agent_id:
            continue
        node_id = f"persona:{agent_id}"
        persona_map[agent_id] = node_id
        add_node(
            node_id,
            "SwarmPersona",
            persona.get("name") or agent_id,
            agent_id=agent_id,
            archetype=persona.get("archetype"),
            stance_bias=persona.get("stance_bias"),
            influence_weight=persona.get("influence_weight"),
            summary=persona.get("bio") or persona.get("persona"),
        )
    for round_item in artifact.get("swarm_rounds", []) or []:
        for index, action in enumerate(round_item.get("actions", []) or []):
            action_id = f"{action.get('agent_id', 'agent')}:{round_item.get('round_index', 0)}:{index}"
            node_id = f"action:{action_id}"
            add_node(
                node_id,
                "SwarmAction",
                action.get("action_type") or "action",
                action_id=action_id,
                action_type=action.get("action_type"),
                direction=action.get("direction"),
                strength=action.get("strength"),
                round_index=action.get("round_index"),
                summary=action.get("content"),
            )
            agent_node = persona_map.get(action.get("agent_id"))
            if agent_node:
                add_edge(f"{agent_node}:action:{action_id}", agent_node, node_id, "AUTHORED")
            for feature_name in action.get("referenced_feature_names", []) or []:
                feature_node = feature_map.get(feature_name)
                if feature_node:
                    add_edge(f"{node_id}:feature:{feature_name}", node_id, feature_node, "MENTIONS")
            for evidence_id in action.get("referenced_evidence_ids", []) or []:
                evidence_node = evidence_map.get(evidence_id)
                if evidence_node:
                    add_edge(f"{node_id}:evidence:{evidence_id}", node_id, evidence_node, "MENTIONS")

    for scenario_type, points in (artifact.get("market_projection") or {}).items():
        if scenario_type == "confidence_band" or not isinstance(points, list):
            continue
        for point in points[:30]:
            point_id = f"projection:{scenario_type}:{point.get('forecast_date')}:{point.get('horizon_day')}"
            add_node(
                point_id,
                "ProjectionPoint",
                f"{scenario_type} {point.get('horizon_day')}",
                forecast_date=point.get("forecast_date"),
                horizon_day=point.get("horizon_day"),
                scenario_type=scenario_type,
                predicted_price=point.get("predicted_price"),
                predicted_return=point.get("predicted_return"),
            )
            add_edge(f"{run_node_id}:{point_id}", run_node_id, point_id, "PROJECTS_TO")
            if point.get("target_symbol"):
                instrument_id = f"instrument:{point['target_symbol']}"
                add_node(instrument_id, "Instrument", point["target_symbol"], instrument=point["target_symbol"], category="projection_target")
                add_edge(f"{point_id}:instrument:{point['target_symbol']}", point_id, instrument_id, "REFERENCES_INSTRUMENT")

    for item in artifact.get("sector_outlook", []) or []:
        symbol = item.get("sector_symbol") or item.get("ticker")
        if not symbol:
            continue
        node_id = f"sector:{symbol}"
        add_node(
            node_id,
            "SectorOutlook",
            item.get("sector_name") or symbol,
            sector_symbol=symbol,
            recommendation_label=item.get("recommendation_label") or item.get("prediction"),
            ranking_score=item.get("ranking_score"),
            expected_return_30d=item.get("expected_return_30d"),
            summary=item.get("rationale"),
        )
        add_edge(f"{run_node_id}:{node_id}", run_node_id, node_id, "PROJECTS_TO")

    highlights = {
        "top_features": [node for node in nodes if node["type"] == "SignalFeature"][:5],
        "top_evidence": [node for node in nodes if node["type"] == "Evidence"][:5],
        "analysis_summary": ontology.get("analysis_summary"),
    }
    return {
        "project_id": project_id,
        "run_id": run_id,
        "ontology": ontology,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "highlights": highlights,
    }


def _build_graph_payload_from_knowledge_graph(
    graph: KnowledgeGraph,
    *,
    ontology: dict[str, Any],
    project_id: str,
    run_id: str,
) -> dict[str, Any]:
    nodes = []
    edges = []
    seen_node_ids: set[str] = set()
    seen_edge_ids: set[str] = set()

    for entity in graph.entities:
        if entity.entity_id in seen_node_ids:
            continue
        seen_node_ids.add(entity.entity_id)
        nodes.append(
            {
                "id": entity.entity_id,
                "type": entity.entity_type,
                "label": entity.name,
                "group": entity.entity_type,
                "project_id": project_id,
                "properties": {
                    **entity.properties,
                    "canonical_name": entity.canonical_name,
                    "source_document_ids": entity.source_document_ids,
                    **entity.metadata,
                },
                "summary": entity.summary,
            }
        )

    for relationship in graph.relationships:
        if relationship.relationship_id in seen_edge_ids:
            continue
        if relationship.source_entity_id not in seen_node_ids or relationship.target_entity_id not in seen_node_ids:
            continue
        seen_edge_ids.add(relationship.relationship_id)
        edges.append(
            {
                "id": relationship.relationship_id,
                "source": relationship.source_entity_id,
                "target": relationship.target_entity_id,
                "relation": relationship.relationship_type,
                "project_id": project_id,
                "weight": relationship.weight,
                "properties": {
                    **relationship.properties,
                    "evidence_document_ids": relationship.evidence_document_ids,
                    **relationship.metadata,
                },
            }
        )

    highlights = {
        "top_features": [node for node in nodes if node["type"] == "SignalFeature"][:5],
        "top_evidence": [node for node in nodes if node["type"] == "Evidence"][:5],
        "analysis_summary": ontology.get("analysis_summary"),
    }
    return {
        "project_id": project_id,
        "run_id": run_id,
        "ontology": ontology,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "highlights": highlights,
        "knowledge_graph": graph.to_dict(),
    }


class Neo4jGraphBuilder:
    def __init__(self, config: dict) -> None:
        self.settings = neo4j_aura_config_from_config(config)

    def build_graph(self, *, project_id: str, graph_payload: dict[str, Any]) -> dict[str, Any]:
        with Neo4jGraphStore(self.settings) as store:
            store.verify_connectivity()
            repository = Neo4jGraphRepository(store)
            repository.ensure_schema()
            knowledge_graph_payload = graph_payload.get("knowledge_graph")
            if isinstance(knowledge_graph_payload, dict):
                repository.upsert_knowledge_graph(knowledge_graph_payload)
            return repository.replace_projected_graph(project_id=project_id, graph_payload=graph_payload)


def find_artifact_for_run(results_dir: str | Path, run_id: str) -> Path | None:
    results_root = Path(results_dir)
    candidates = list(results_root.glob("*.json")) + list(results_root.glob("runs/*/*.json"))
    for path in sorted(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") == str(run_id):
            return path
    return None


def find_latest_graph_snapshot(config: dict | None = None) -> dict[str, Any] | None:
    config = load_config(config)
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        projects = repo.list_graph_projects()
        for project in projects:
            snapshot = repo.get_graph_snapshot(project_id=project["project_id"])
            if snapshot:
                return merge_graph_records(project, snapshot, repo.get_latest_graph_task(project["project_id"]))
    return None


def get_graph_for_run(run_id: str, config: dict | None = None) -> dict[str, Any] | None:
    config = load_config(config)
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        project = repo.get_graph_project(run_id=run_id)
        if not project:
            return None
        snapshot = repo.get_graph_snapshot(project_id=project["project_id"])
        task = repo.get_latest_graph_task(project["project_id"])
        return merge_graph_records(project, snapshot, task)


def list_graph_history(config: dict | None = None) -> list[dict[str, Any]]:
    config = load_config(config)
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        history = []
        for project in repo.list_graph_projects():
            snapshot = repo.get_graph_snapshot(project_id=project["project_id"])
            task = repo.get_latest_graph_task(project["project_id"])
            history.append(merge_graph_records(project, snapshot, task))
        return history


def get_graph_status(*, task_id: str | None = None, run_id: str | None = None, config: dict | None = None) -> dict[str, Any] | None:
    config = load_config(config)
    if task_id:
        task = GRAPH_TASK_MANAGER.get(task_id)
        if task:
            return task
        with database_session(config["database_url"]) as session:
            create_schema(session)
            repo = MarketRepository(session)
            return repo.get_graph_task(task_id)
    if run_id:
        graph = get_graph_for_run(run_id, config=config)
        if graph:
            return {
                "task_id": graph.get("task_id"),
                "project_id": graph.get("project_id"),
                "run_id": graph.get("run_id"),
                "status": graph.get("status"),
                "progress_stage": graph.get("progress_stage"),
                "error_message": graph.get("error_message"),
            }
    return None


def merge_graph_records(project: dict[str, Any], snapshot: dict[str, Any] | None, task: dict[str, Any] | None) -> dict[str, Any]:
    graph_payload = _safe_json_loads(snapshot["graph_json"], {}) if snapshot and snapshot.get("graph_json") else {}
    highlights = _safe_json_loads(snapshot["highlights_json"], {}) if snapshot and snapshot.get("highlights_json") else {}
    ontology = _safe_json_loads(project.get("ontology_json") or "{}", {})
    knowledge_graph_payload = graph_payload.get("knowledge_graph") if isinstance(graph_payload.get("knowledge_graph"), dict) else None
    return {
        "project_id": project["project_id"],
        "run_id": project["run_id"],
        "status": project["status"],
        "graph_backend": project.get("graph_backend"),
        "backend_graph_ref": project.get("backend_graph_ref"),
        "source_artifact_path": project.get("source_artifact_path"),
        "created_at": project.get("created_at"),
        "updated_at": project.get("updated_at"),
        "task_id": task.get("task_id") if task else None,
        "progress_stage": task.get("progress_stage") if task else project["status"],
        "progress_detail": task.get("progress_detail") if task else None,
        "error_message": task.get("error_message") if task else None,
        "stage_started_at": task.get("stage_started_at") if task else None,
        "last_progress_at": task.get("last_progress_at") if task else None,
        "node_count": snapshot.get("node_count", 0) if snapshot else 0,
        "edge_count": snapshot.get("edge_count", 0) if snapshot else 0,
        "ontology": ontology,
        "telemetry": task.get("telemetry") if task else None,
        "nodes": graph_payload.get("nodes", []),
        "edges": graph_payload.get("edges", []),
        "knowledge_graph": knowledge_graph_payload,
        "highlights": highlights,
    }


def backfill_graphs(*, config: dict | None = None, results_dir: str | Path | None = None, limit: int | None = None) -> list[GraphBuildContext]:
    config = load_config(config)
    results_root = Path(results_dir or config["results_dir"])
    queued: list[GraphBuildContext] = []
    candidates = sorted(list(results_root.glob("*.json")) + list(results_root.glob("runs/*/*.json")))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            continue
        queued.append(build_graph_for_run(run_id=run_id, config=config, results_dir=results_root, artifact_path=path))
        if limit is not None and len(queued) >= limit:
            break
    return queued
