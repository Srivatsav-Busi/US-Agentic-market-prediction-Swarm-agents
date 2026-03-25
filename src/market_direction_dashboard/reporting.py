from __future__ import annotations

import html
import json
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import load_config
from .domain import report_context_from_payload
from .graph import get_graph_for_run
from .llm_clients import create_llm_client, is_real_llm_client, with_timeout
from .report_agent import build_report_agent_result
from .simulation_reporting import build_simulation_history_report
from .storage.db import create_schema, database_session
from .storage.models import AnalysisReportRecord, AnalysisReportTaskRecord
from .storage.repositories import MarketRepository


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class ReportTaskManager:
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


REPORT_TASK_MANAGER = ReportTaskManager()


@dataclass(frozen=True)
class ReportGenerationContext:
    report_id: str
    task_id: str
    subject_type: str
    subject_id: str
    status: str
    artifact_dir: str


def queue_analysis_report(
    *,
    config: dict | None = None,
    results_dir: str | Path | None = None,
    run_id: str | None = None,
    simulation_id: str | None = None,
    force_rebuild: bool = False,
) -> ReportGenerationContext:
    subject_type, subject_id = _resolve_subject_identifiers(run_id=run_id, simulation_id=simulation_id)
    config, results_root = _normalize_config(config=config, results_dir=results_dir)
    llm_client = with_timeout(create_llm_client(config), int(config.get("report_llm_timeout_seconds", config.get("llm_api_timeout_seconds", 30)) or 30))
    if not is_real_llm_client(llm_client):
        raise RuntimeError("Report generation requires a configured OpenRouter LLM backend.")

    subject_context = _load_subject_context(subject_type=subject_type, subject_id=subject_id, results_root=results_root, config=config)
    existing: dict[str, Any] | None = None
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        existing = repo.get_latest_analysis_report_for_subject(subject_type=subject_type, subject_id=subject_id)
        if existing and not force_rebuild and existing["status"] in {"queued", "running", "completed"}:
            task = repo.get_latest_analysis_report_task(existing["report_id"])
            return ReportGenerationContext(
                report_id=existing["report_id"],
                task_id=task["task_id"] if task else "",
                subject_type=subject_type,
                subject_id=subject_id,
                status=existing["status"],
                artifact_dir=existing["artifact_dir"],
            )

    report_id = f"report-{uuid4().hex[:12]}"
    task_id = f"report-task-{uuid4().hex[:12]}"
    artifact_dir = results_root / "reports" / report_id
    now = utc_now_iso()
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        repo.upsert_analysis_report(
            AnalysisReportRecord(
                report_id=report_id,
                subject_type=subject_type,
                subject_id=subject_id,
                status="queued",
                artifact_dir=str(artifact_dir),
                source_artifact_path=subject_context["source_artifact_path"],
                graph_project_id=subject_context.get("graph_project_id"),
                created_at=now,
                updated_at=now,
            )
        )
        repo.upsert_analysis_report_task(
            AnalysisReportTaskRecord(
                task_id=task_id,
                report_id=report_id,
                status="queued",
                progress_stage="queued",
                started_at=now,
            )
        )

    REPORT_TASK_MANAGER.create(
        task_id,
        {
            "task_id": task_id,
            "report_id": report_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
            "status": "queued",
            "progress_stage": "queued",
            "error_message": None,
            "started_at": now,
            "completed_at": None,
        },
    )
    worker = threading.Thread(
        target=_generate_report_background,
        kwargs={
            "config": config,
            "results_root": results_root,
            "report_id": report_id,
            "task_id": task_id,
            "subject_type": subject_type,
            "subject_id": subject_id,
        },
        daemon=True,
        name=f"report-{report_id}",
    )
    worker.start()
    return ReportGenerationContext(
        report_id=report_id,
        task_id=task_id,
        subject_type=subject_type,
        subject_id=subject_id,
        status="queued",
        artifact_dir=str(artifact_dir),
    )


def get_analysis_report_status(
    *,
    report_id: str | None = None,
    run_id: str | None = None,
    simulation_id: str | None = None,
    config: dict | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    config, results_root = _normalize_config(config=config, results_dir=results_dir)
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        if report_id:
            report = repo.get_analysis_report(report_id)
        else:
            subject_type, subject_id = _resolve_subject_identifiers(run_id=run_id, simulation_id=simulation_id)
            report = repo.get_latest_analysis_report_for_subject(subject_type=subject_type, subject_id=subject_id)
        if report is None:
            return None
        task = repo.get_latest_analysis_report_task(report["report_id"])
    return _merge_report_records(report=report, task=task, results_root=results_root)


def get_analysis_report_result(
    *,
    report_id: str,
    config: dict | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    status = get_analysis_report_status(report_id=report_id, config=config, results_dir=results_dir)
    if status is None:
        return None
    artifact_dir = Path(status["artifact_dir"])
    outline = _load_json_if_exists(artifact_dir / "outline.json", {})
    progress = _load_json_if_exists(artifact_dir / "progress.json", {})
    meta = _load_json_if_exists(artifact_dir / "meta.json", {})
    structured_summary = _load_json_if_exists(artifact_dir / "structured_summary.json", {})
    markdown = (artifact_dir / "full_report.md").read_text(encoding="utf-8") if (artifact_dir / "full_report.md").exists() else ""
    return {
        **status,
        "meta": meta,
        "outline": outline,
        "progress": progress,
        "structured_summary": structured_summary,
        "markdown": markdown,
    }


def summarize_analysis_report(
    *,
    run_id: str | None = None,
    simulation_id: str | None = None,
    report_id: str | None = None,
    config: dict | None = None,
    results_dir: str | Path | None = None,
) -> dict[str, Any] | None:
    if not any([run_id, simulation_id, report_id]):
        return None
    status = get_analysis_report_status(
        report_id=report_id,
        run_id=run_id,
        simulation_id=simulation_id,
        config=config,
        results_dir=results_dir,
    )
    if status is None:
        return None
    artifact_dir = Path(status["artifact_dir"])
    outline = _load_json_if_exists(artifact_dir / "outline.json", {})
    progress = _load_json_if_exists(artifact_dir / "progress.json", {})
    return {
        "report_id": status["report_id"],
        "status": status["status"],
        "subject_type": status["subject_type"],
        "subject_id": status["subject_id"],
        "task_id": status.get("task_id"),
        "progress_stage": status.get("progress_stage"),
        "summary": outline.get("summary"),
        "title": outline.get("title"),
        "section_count": len(outline.get("sections", [])),
        "completed_sections": progress.get("completed_sections", 0),
        "structured_summary": _load_json_if_exists(artifact_dir / "structured_summary.json", {}),
        "markdown_path": status.get("markdown_path"),
        "html_path": status.get("html_path"),
        "updated_at": status.get("updated_at"),
        "error_message": status.get("error_message"),
    }


def _generate_report_background(*, config: dict, results_root: Path, report_id: str, task_id: str, subject_type: str, subject_id: str) -> None:
    artifact_dir = results_root / "reports" / report_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    try:
        _update_report_state(config, report_id, task_id, status="running", progress_stage="loading_context")
        context = _load_subject_context(subject_type=subject_type, subject_id=subject_id, results_root=results_root, config=config)
        llm_client = with_timeout(create_llm_client(config), int(config.get("report_llm_timeout_seconds", config.get("llm_api_timeout_seconds", 30)) or 30))

        _write_json(artifact_dir / "meta.json", _build_meta(context=context, report_id=report_id))
        _update_progress(artifact_dir, total_sections=0, completed_sections=0, stage="planning")
        _update_report_state(config, report_id, task_id, status="running", progress_stage="planning")

        outline = _plan_outline(context=context, llm_client=llm_client)
        _write_json(artifact_dir / "outline.json", outline)

        sections_output: list[dict[str, Any]] = []
        total_sections = len(outline["sections"])
        _update_progress(artifact_dir, total_sections=total_sections, completed_sections=0, stage="writing_sections")
        _append_log(artifact_dir / "agent_log.jsonl", {"event": "outline_created", "outline": outline})

        for index, section in enumerate(outline["sections"], start=1):
            _update_report_state(config, report_id, task_id, status="running", progress_stage=f"section_{index}")
            retrievals = _build_section_evidence(section=section, context=context, llm_client=llm_client, artifact_dir=artifact_dir)
            content = _write_section(
                outline=outline,
                section=section,
                retrievals=retrievals,
                prior_sections=sections_output,
                llm_client=llm_client,
            )
            section_payload = {
                "title": section["title"],
                "body": content.strip(),
                "tools_used": [item["tool"] for item in retrievals],
            }
            sections_output.append(section_payload)
            (artifact_dir / f"section_{index:02d}.md").write_text(
                f"## {section['title']}\n\n{section_payload['body']}\n",
                encoding="utf-8",
            )
            _append_log(artifact_dir / "agent_log.jsonl", {"event": "section_completed", "index": index, "section": section_payload})
            _update_progress(artifact_dir, total_sections=total_sections, completed_sections=index, stage="writing_sections")

        _update_report_state(config, report_id, task_id, status="running", progress_stage="assembling")
        markdown = _assemble_markdown(outline=outline, sections=sections_output)
        html_report = _render_report_html(outline["title"], markdown)
        structured_summary = _build_structured_summary(context=context, outline=outline)
        (artifact_dir / "full_report.md").write_text(markdown, encoding="utf-8")
        (artifact_dir / "full_report.html").write_text(html_report, encoding="utf-8")
        _write_json(artifact_dir / "structured_summary.json", structured_summary)
        _update_progress(artifact_dir, total_sections=total_sections, completed_sections=total_sections, stage="completed")
        completed_at = utc_now_iso()
        with (artifact_dir / "console_log.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{completed_at} completed report generation\n")
        _complete_report_state(config, report_id, task_id, status="completed", error_message=None, completed_at=completed_at)
    except Exception as exc:
        completed_at = utc_now_iso()
        with (artifact_dir / "console_log.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{completed_at} error: {exc}\n")
        _complete_report_state(config, report_id, task_id, status="failed", error_message=str(exc), completed_at=completed_at)


def _resolve_subject_identifiers(*, run_id: str | None, simulation_id: str | None) -> tuple[str, str]:
    if bool(run_id) == bool(simulation_id):
        raise ValueError("Exactly one of run_id or simulation_id is required.")
    if run_id:
        return "run", run_id
    return "simulation", str(simulation_id)


def _normalize_config(*, config: dict | None, results_dir: str | Path | None) -> tuple[dict[str, Any], Path]:
    loaded = load_config(config)
    results_root = Path(results_dir or loaded["results_dir"])
    resolved_overrides = {
        "results_dir": str(results_root),
    }
    default_database_url = str(load_config()["database_url"])
    if not config or not config.get("database_url") or loaded.get("database_url") == default_database_url:
        resolved_overrides["database_url"] = f"sqlite:///{results_root / 'market_intelligence.db'}"
    loaded = load_config({**loaded, **resolved_overrides})
    return loaded, results_root


def _load_subject_context(*, subject_type: str, subject_id: str, results_root: Path, config: dict) -> dict[str, Any]:
    if subject_type == "run":
        artifact_path = _find_run_artifact(results_root, subject_id)
        if artifact_path is None:
            raise FileNotFoundError(f"No result artifact found for run_id={subject_id}")
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        graph = get_graph_for_run(subject_id, config={**config, "results_dir": str(results_root)})
        report_context = report_context_from_payload(
            subject_type=subject_type,
            subject_id=subject_id,
            artifact=artifact,
            graph_payload=graph.get("knowledge_graph") if graph else artifact.get("knowledge_graph"),
        )
        return {
            "subject_type": subject_type,
            "subject_id": subject_id,
            "artifact": artifact,
            "artifact_path": str(artifact_path),
            "source_artifact_path": str(artifact_path),
            "graph": graph,
            "report_context": report_context,
            "graph_project_id": graph.get("project_id") if graph else None,
            "base_run_id": subject_id,
        }

    result_path = _find_simulation_artifact(results_root, subject_id, "result.json")
    if result_path is None:
        raise FileNotFoundError(f"No simulation result found for simulation_id={subject_id}")
    simulation_dir = result_path.parent
    artifact = json.loads(result_path.read_text(encoding="utf-8"))
    run_state = _load_json_if_exists(simulation_dir / "run_state.json", {})
    actions = _load_jsonl_if_exists(simulation_dir / "actions.jsonl")
    environment = _load_json_if_exists(simulation_dir / "simulation_environment.json", {})
    simulation_state = _load_json_if_exists(simulation_dir / "simulation_state.json", {})
    memory_snapshot = _load_json_if_exists(simulation_dir / "simulation_memory_snapshot.json", {})
    state_trace = _load_jsonl_if_exists(simulation_dir / "simulation_state_trace.jsonl")
    queue_record = _load_json_if_exists(simulation_dir / "queue_job.json", {})
    simulation_bundle = {
        "simulation_id": subject_id,
        "simulation_dir": str(simulation_dir),
        "result": artifact,
        "run_state": run_state,
        "actions": actions,
        "environment": environment,
        "simulation_state": simulation_state,
        "memory_snapshot": memory_snapshot,
        "state_trace": state_trace,
        "queue_record": queue_record,
    }
    simulation_bundle["analytics"] = build_simulation_history_report(simulation_bundle)
    base_run_stem = run_state.get("base_run_stem") or simulation_dir.parent.name
    base_run_artifact = _find_base_run_artifact(results_root, base_run_stem)
    base_run_payload = _load_json_if_exists(base_run_artifact, {}) if base_run_artifact else {}
    base_run_id = str(base_run_payload.get("run_id") or "")
    graph = get_graph_for_run(base_run_id, config={**config, "results_dir": str(results_root)}) if base_run_id else None
    report_context = report_context_from_payload(
        subject_type=subject_type,
        subject_id=subject_id,
        artifact=artifact,
        graph_payload=graph.get("knowledge_graph") if graph else base_run_payload.get("knowledge_graph"),
        simulation_payload=environment,
    )
    return {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "artifact": artifact,
        "artifact_path": str(result_path),
        "source_artifact_path": str(result_path),
        "run_state": run_state,
        "simulation_state": simulation_state,
        "simulation_actions": actions,
        "simulation_environment": environment,
        "simulation_memory_snapshot": memory_snapshot,
        "simulation_state_trace": state_trace,
        "simulation_queue_record": queue_record,
        "simulation_bundle": simulation_bundle,
        "base_run_artifact": base_run_payload,
        "base_run_artifact_path": str(base_run_artifact) if base_run_artifact else None,
        "graph": graph,
        "report_context": report_context,
        "graph_project_id": graph.get("project_id") if graph else None,
        "base_run_id": base_run_id or None,
    }


def _build_meta(*, context: dict[str, Any], report_id: str) -> dict[str, Any]:
    artifact = context["artifact"]
    report_context = context.get("report_context")
    simulation_analytics = ((context.get("simulation_bundle") or {}).get("analytics") or {})
    return {
        "report_id": report_id,
        "subject_type": context["subject_type"],
        "subject_id": context["subject_id"],
        "prediction_date": artifact.get("prediction_date"),
        "target": artifact.get("target"),
        "prediction_label": artifact.get("prediction_label"),
        "run_health": artifact.get("run_health"),
        "source_artifact_path": context.get("source_artifact_path"),
        "graph_project_id": context.get("graph_project_id"),
        "graph_entity_count": len(report_context.graph.entities) if report_context else 0,
        "graph_relationship_count": len(report_context.graph.relationships) if report_context else 0,
        "simulation_round_count": ((simulation_analytics.get("summary") or {}).get("round_count")),
        "simulation_graph_delta_count": ((simulation_analytics.get("graph") or {}).get("count")),
        "generated_at": utc_now_iso(),
    }


def _build_structured_summary(*, context: dict[str, Any], outline: dict[str, Any]) -> dict[str, Any]:
    artifact = context["artifact"]
    simulation_bundle = context.get("simulation_bundle")
    graph_summary = context.get("graph")
    return build_report_agent_result(
        artifact=artifact,
        outline=outline,
        simulation_bundle=simulation_bundle,
        graph_summary=graph_summary,
    ).to_dict()


def _plan_outline(*, context: dict[str, Any], llm_client) -> dict[str, Any]:
    artifact = context["artifact"]
    simulation_analytics = ((context.get("simulation_bundle") or {}).get("analytics") or {})
    fallback = {
        "title": f"{artifact.get('target', 'Market')} Analysis Report",
        "summary": artifact.get("summary") or "Generated market analysis report.",
        "sections": [
            {"title": "Executive Summary", "focus": "decision, confidence, and primary outlook"},
            {"title": "Evidence Base", "focus": "top drivers, features, and source quality"},
            {"title": "Risk Map", "focus": "challenge signals, failure modes, and downside cases"},
        ],
    }
    if context["subject_type"] == "simulation":
        fallback["sections"].append({"title": "Simulation Readthrough", "focus": "swarm behavior, scenario response, and sector impact"})
    if context.get("graph"):
        fallback["sections"].append({"title": "Graph Signals", "focus": "connected entities, graph highlights, and structural explainability"})
    fallback["sections"] = fallback["sections"][:5]
    prompt = {
        "subject_type": context["subject_type"],
        "subject_id": context["subject_id"],
        "prediction_label": artifact.get("prediction_label"),
        "confidence": artifact.get("confidence"),
        "summary": artifact.get("summary"),
        "top_drivers": artifact.get("top_drivers", [])[:4],
        "signal_features": artifact.get("signal_features", [])[:6],
        "graph_available": bool(context.get("graph")),
        "simulation_available": context["subject_type"] == "simulation",
        "simulation_summary": simulation_analytics.get("summary"),
        "simulation_graph": simulation_analytics.get("graph"),
    }
    raw = llm_client.complete(
        "Return JSON only with keys title, summary, sections. Sections must be an array of 3 to 5 objects with title and focus. Write for an analyst audience and stay grounded in the provided subject.",
        json.dumps(prompt, indent=2),
        json.dumps(fallback),
    )
    parsed = _safe_json_loads(raw, fallback)
    sections = parsed.get("sections") if isinstance(parsed.get("sections"), list) else fallback["sections"]
    normalized_sections = []
    for item in sections[:5]:
        if isinstance(item, dict) and item.get("title"):
            normalized_sections.append({"title": str(item["title"]).strip(), "focus": str(item.get("focus") or "").strip()})
    if len(normalized_sections) < 3:
        normalized_sections = fallback["sections"]
    return {
        "title": str(parsed.get("title") or fallback["title"]).strip(),
        "summary": str(parsed.get("summary") or fallback["summary"]).strip(),
        "sections": normalized_sections,
    }


def _build_section_evidence(*, section: dict[str, Any], context: dict[str, Any], llm_client, artifact_dir: Path) -> list[dict[str, Any]]:
    retrievals = [_artifact_summary(context=context, section=section)]
    if context.get("graph"):
        retrievals.append(_graph_context(context=context, section=section))
    if context["subject_type"] == "simulation":
        retrievals.append(_simulation_trace(context=context, section=section))
    if any(keyword in section["title"].lower() for keyword in ("risk", "desk", "signal", "evidence", "summary")):
        retrievals.append(_desk_interview(context=context, section=section, llm_client=llm_client))
    if len(retrievals) < 2:
        retrievals.append(_desk_interview(context=context, section=section, llm_client=llm_client))
    for retrieval in retrievals:
        _append_log(artifact_dir / "agent_log.jsonl", {"event": "tool_call", "tool": retrieval["tool"], "section": section["title"], "result": retrieval["content"]})
    return retrievals


def _artifact_summary(*, context: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    artifact = context["artifact"]
    top_drivers = ", ".join(item.get("name", "driver") for item in artifact.get("top_drivers", [])[:4]) or "No explicit top drivers published"
    feature_names = ", ".join(item.get("name", "feature") for item in artifact.get("signal_features", [])[:6]) or "No structured features"
    confidence_notes = "; ".join(artifact.get("confidence_notes", [])[:4]) or "No explicit confidence notes"
    content = (
        f"Prediction={artifact.get('prediction_label')} confidence={artifact.get('confidence')} run_health={artifact.get('run_health')}. "
        f"Summary={artifact.get('summary')}. Focus={section['focus']}. "
        f"Top drivers: {top_drivers}. Features: {feature_names}. Confidence notes: {confidence_notes}."
    )
    return {"tool": "artifact_summary", "content": content}


def _graph_context(*, context: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    graph = context.get("graph") or {}
    highlights = graph.get("highlights") or {}
    top_features = ", ".join(item.get("label", "feature") for item in (highlights.get("top_features") or [])[:4]) or "No graph highlight features"
    top_evidence = ", ".join(item.get("label", "evidence") for item in (highlights.get("top_evidence") or [])[:3]) or "No graph highlight evidence"
    content = (
        f"Graph status={graph.get('status')} nodes={graph.get('node_count', 0)} edges={graph.get('edge_count', 0)}. "
        f"Ontology summary={graph.get('ontology', {}).get('analysis_summary', 'n/a')}. "
        f"Top graph features: {top_features}. Top graph evidence: {top_evidence}. Focus={section['focus']}."
    )
    return {"tool": "graph_context", "content": content}


def _simulation_trace(*, context: dict[str, Any], section: dict[str, Any]) -> dict[str, Any]:
    run_state = context.get("run_state") or {}
    analytics = ((context.get("simulation_bundle") or {}).get("analytics") or {})
    latest = dict(analytics.get("latest") or {})
    graph = dict(analytics.get("graph") or {})
    action_bits = []
    latest_event = latest.get("event") or {}
    latest_delta = latest.get("graph_delta") or {}
    if latest_event:
        action_bits.append(f"latest event={latest_event.get('event')} round={latest_event.get('round_index')}")
    if latest_delta:
        action_bits.append(f"latest delta actions={latest_delta.get('new_action_nodes', 0)}")
    content = (
        f"Simulation status={run_state.get('status')} rounds={run_state.get('current_round')}/{run_state.get('total_rounds')}. "
        f"Scenario summary={context['artifact'].get('summary')}. Action trace: {'; '.join(action_bits) or 'No persisted action log excerpts'}."
        f" Graph deltas={graph.get('count', 0)}. Focus={section['focus']}."
    )
    return {"tool": "simulation_trace", "content": content}


def _desk_interview(*, context: dict[str, Any], section: dict[str, Any], llm_client) -> dict[str, Any]:
    artifact = context["artifact"]
    category = _pick_interview_category(section["title"])
    report_text = artifact.get(f"{category}_report") or artifact.get("market_context_report") or artifact.get("summary") or "No desk report available."
    sources = [item for item in artifact.get("sources", []) if item.get("category") == category][:4]
    evidence = "\n".join(f"- {item.get('title')}: {item.get('summary')}" for item in sources) or "- No category-specific evidence"
    fallback = f"{title_case(category)} desk remains anchored to: {report_text}"
    response = llm_client.complete(
        f"You are the {category} market desk. Answer in 2 to 3 sentences and stay evidence-backed.",
        f"Question: What matters most for the section '{section['title']}'?\nDesk report: {report_text}\nEvidence:\n{evidence}",
        fallback,
    )
    return {"tool": "desk_interview", "content": response.strip() or fallback}


def _write_section(*, outline: dict[str, Any], section: dict[str, Any], retrievals: list[dict[str, Any]], prior_sections: list[dict[str, Any]], llm_client) -> str:
    evidence_text = "\n".join(f"[{item['tool']}] {item['content']}" for item in retrievals)
    prior_titles = ", ".join(item["title"] for item in prior_sections) or "None"
    fallback_lines = [
        f"{outline['summary']}",
        f"This section focuses on {section['focus']}.",
        *(item["content"] for item in retrievals[:3]),
    ]
    fallback = "\n\n".join(fallback_lines)
    return llm_client.complete(
        "Write one markdown section body for an analyst report. Use 2 to 4 short paragraphs, avoid bullets, avoid hype, and synthesize the evidence. Do not include the section heading.",
        (
            f"Report title: {outline['title']}\n"
            f"Report summary: {outline['summary']}\n"
            f"Section title: {section['title']}\n"
            f"Section focus: {section['focus']}\n"
            f"Prior sections: {prior_titles}\n"
            f"Evidence:\n{evidence_text}"
        ),
        fallback,
    ).strip()


def _assemble_markdown(*, outline: dict[str, Any], sections: list[dict[str, Any]]) -> str:
    parts = [f"# {outline['title']}", "", outline["summary"], ""]
    for section in sections:
        parts.extend([f"## {section['title']}", "", section["body"], ""])
    return "\n".join(parts).strip() + "\n"


def _render_report_html(title: str, markdown: str) -> str:
    lines = []
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        escaped = html.escape(line)
        if line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        else:
            lines.append(f"<p>{escaped}</p>")
    body = "\n".join(lines)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title>"
        "<style>body{font-family:Georgia,serif;max-width:860px;margin:40px auto;padding:0 20px;line-height:1.6;color:#1b1b1b;background:#f7f3ea}"
        "h1,h2{font-family:'Times New Roman',serif;color:#7a2f24}p{margin:0 0 1rem}h2{margin-top:2rem}</style>"
        f"</head><body>{body}</body></html>"
    )


def _update_report_state(config: dict, report_id: str, task_id: str, *, status: str, progress_stage: str) -> None:
    now = utc_now_iso()
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        report = repo.get_analysis_report(report_id)
        task = repo.get_analysis_report_task(task_id)
        repo.upsert_analysis_report(
            AnalysisReportRecord(
                report_id=report_id,
                subject_type=report["subject_type"],
                subject_id=report["subject_id"],
                status=status,
                source_artifact_path=report.get("source_artifact_path"),
                graph_project_id=report.get("graph_project_id"),
                artifact_dir=report["artifact_dir"],
                created_at=report["created_at"],
                updated_at=now,
                completed_at=report.get("completed_at"),
                error_message=report.get("error_message"),
            )
        )
        repo.upsert_analysis_report_task(
            AnalysisReportTaskRecord(
                task_id=task_id,
                report_id=report_id,
                status=status,
                progress_stage=progress_stage,
                error_message=task.get("error_message") if task else None,
                started_at=task["started_at"] if task else now,
                completed_at=task.get("completed_at") if task else None,
            )
        )
    REPORT_TASK_MANAGER.update(task_id, status=status, progress_stage=progress_stage)


def _complete_report_state(config: dict, report_id: str, task_id: str, *, status: str, error_message: str | None, completed_at: str) -> None:
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        report = repo.get_analysis_report(report_id)
        task = repo.get_analysis_report_task(task_id)
        repo.upsert_analysis_report(
            AnalysisReportRecord(
                report_id=report_id,
                subject_type=report["subject_type"],
                subject_id=report["subject_id"],
                status=status,
                source_artifact_path=report.get("source_artifact_path"),
                graph_project_id=report.get("graph_project_id"),
                artifact_dir=report["artifact_dir"],
                created_at=report["created_at"],
                updated_at=completed_at,
                completed_at=completed_at if status == "completed" else None,
                error_message=error_message,
            )
        )
        repo.upsert_analysis_report_task(
            AnalysisReportTaskRecord(
                task_id=task_id,
                report_id=report_id,
                status=status,
                progress_stage=status,
                error_message=error_message,
                started_at=task["started_at"] if task else completed_at,
                completed_at=completed_at,
            )
        )
    REPORT_TASK_MANAGER.update(task_id, status=status, progress_stage=status, error_message=error_message, completed_at=completed_at)


def _merge_report_records(*, report: dict[str, Any], task: dict[str, Any] | None, results_root: Path) -> dict[str, Any]:
    artifact_dir = Path(report["artifact_dir"])
    markdown_path = _to_report_url(path=artifact_dir / "full_report.md", results_root=results_root)
    html_path = _to_report_url(path=artifact_dir / "full_report.html", results_root=results_root)
    structured_summary_path = _to_report_url(path=artifact_dir / "structured_summary.json", results_root=results_root)
    progress = _load_json_if_exists(artifact_dir / "progress.json", {})
    return {
        "report_id": report["report_id"],
        "subject_type": report["subject_type"],
        "subject_id": report["subject_id"],
        "status": report["status"],
        "source_artifact_path": report.get("source_artifact_path"),
        "graph_project_id": report.get("graph_project_id"),
        "artifact_dir": report["artifact_dir"],
        "created_at": report.get("created_at"),
        "updated_at": report.get("updated_at"),
        "completed_at": report.get("completed_at"),
        "error_message": report.get("error_message"),
        "task_id": task.get("task_id") if task else None,
        "progress_stage": task.get("progress_stage") if task else report["status"],
        "markdown_path": markdown_path,
        "html_path": html_path,
        "structured_summary_path": structured_summary_path,
        "structured_summary": _load_json_if_exists(artifact_dir / "structured_summary.json", {}),
        "progress": progress,
    }


def _find_run_artifact(results_root: Path, run_id: str) -> Path | None:
    candidates = sorted(results_root.glob("*.json"))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") == run_id:
            return path
    run_dir_candidates = sorted(results_root.glob("runs/*/*.json"))
    for path in run_dir_candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("run_id") or "") == run_id:
            return path
    return None


def _find_base_run_artifact(results_root: Path, base_run_stem: str) -> Path | None:
    if not base_run_stem:
        return None
    direct = results_root / f"{base_run_stem}.json"
    if direct.exists():
        return direct
    nested = results_root / "runs" / base_run_stem / f"{base_run_stem}.json"
    if nested.exists():
        return nested
    return None


def _find_simulation_artifact(results_root: Path, simulation_id: str, file_name: str) -> Path | None:
    matches = sorted(results_root.glob(f"simulations/*/{simulation_id}/{file_name}"))
    return matches[-1] if matches else None


def _to_report_url(*, path: Path, results_root: Path) -> str | None:
    if not path.exists():
        return None
    resolved_path = path.resolve()
    resolved_results_root = results_root.resolve()
    generated_root = (resolved_results_root / "reports").resolve()
    if generated_root in resolved_path.parents:
        relative = resolved_path.relative_to(generated_root)
    else:
        relative = resolved_path.relative_to(resolved_results_root)
    return "/reports/" + "/".join(relative.parts)


def _load_json_if_exists(path: Path, fallback: Any) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _load_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _safe_json_loads(raw: str, fallback: dict[str, Any]) -> dict[str, Any]:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("\n")
        cleaned = "\n".join(parts[1:-1]) if len(parts) > 2 else cleaned.strip("`")
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback
    return parsed if isinstance(parsed, dict) else fallback


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": utc_now_iso(), **payload}) + "\n")


def _update_progress(artifact_dir: Path, *, total_sections: int, completed_sections: int, stage: str) -> None:
    _write_json(
        artifact_dir / "progress.json",
        {
            "status": stage,
            "total_sections": total_sections,
            "completed_sections": completed_sections,
            "updated_at": utc_now_iso(),
        },
    )


def _pick_interview_category(title: str) -> str:
    lowered = title.lower()
    if "policy" in lowered or "politic" in lowered:
        return "political"
    if "sentiment" in lowered or "social" in lowered:
        return "social"
    if "market" in lowered or "graph" in lowered:
        return "market"
    return "economic"


def title_case(value: str) -> str:
    return value.replace("_", " ").title()
