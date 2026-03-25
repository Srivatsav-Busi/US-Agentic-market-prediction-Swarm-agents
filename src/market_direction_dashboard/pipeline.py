from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from .agents import run_research_agents, run_source_agents, synthesize_prediction, run_sector_agents
from .calendar import is_market_closed
from .config import load_config
from .domain import simulation_state_from_environment
from .forecasting.hybrid_ml import run_weekly_retraining
from .graph import maybe_queue_graph_build
from .graph_features import build_graph_delta_summary, build_graph_feature_vector, build_graph_prediction_context
from .ingestion import build_graph_first_ingestion, enrich_knowledge_graph
from .pipelines.bootstrap_history import persist_daily_run
from .storage.db import create_schema, database_session
from .storage.models import SchedulerRunRecord
from .storage.repositories import MarketRepository
from .dashboard import render_dashboard
from .live_features import extract_signal_features
from .llm_clients import LLMProgressTracker, create_llm_client, with_progress_tracker, with_request_group
from .models import DataQualitySummary
from .scheduler import run_scheduler
from .simulation_runner import SimulationRunner
from .sources import apply_graph_quality_layer, collect_sources
from .swarm_simulation import prepare_swarm_environment


def run_daily_prediction(
    output_dir: str | Path,
    target: str = "S&P 500",
    prediction_date: str | None = None,
    config_overrides: dict | None = None,
    scheduler_context: dict | None = None,
) -> dict:
    config = load_config(config_overrides)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _ensure_runtime_layout(output_dir)

    if not prediction_date:
        prediction_date = datetime.now(ZoneInfo(config["timezone"])).date().isoformat()

    _record_runtime_status(
        output_dir,
        {
            "status": "running",
            "prediction_date": prediction_date,
            "target": target,
            "updated_at": datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat(),
        },
    )
    runtime_state = {
        "status": "running",
        "prediction_date": prediction_date,
        "target": target,
        "updated_at": datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat(),
    }
    finalization_trace: list[dict[str, object]] = []
    terminal_state_written = False

    def _write_runtime_state(**updates: object) -> None:
        runtime_state.update({key: value for key, value in updates.items() if value is not None})
        runtime_state["updated_at"] = datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat()
        _record_runtime_status(output_dir, dict(runtime_state))

    def _mark_finalization(stage: str, **details: object) -> None:
        entry = {
            "stage": stage,
            "timestamp": datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat(),
        }
        entry.update({key: value for key, value in details.items() if value is not None})
        finalization_trace.append(entry)
        _write_runtime_state(
            finalization_stage=stage,
            finalization_trace=list(finalization_trace),
        )
        _append_runtime_event(
            output_dir,
            {
                "event": "finalization_stage",
                "stage": stage,
                "prediction_date": prediction_date,
                "target": target,
                **{key: value for key, value in details.items() if value is not None},
                "logged_at": entry["timestamp"],
            },
        )

    scheduler_run = _prepare_scheduler_run(
        config=config,
        target=target,
        prediction_date=prediction_date,
        scheduler_context=scheduler_context,
    )
    
    # Fast path for weekends and market holidays
    if is_market_closed(prediction_date):
        stem = f"{prediction_date}_{_slug(target)}"
        html_path = output_dir / f"{stem}.html"
        json_path = output_dir / f"{stem}.json"
        
        from .forecasting.baseline_30d import apply_hybrid_overlay, build_baseline_forecast
        from .forecasting.hybrid_ml import build_hybrid_ml_overlay
        
        snapshot = {}
        json_files = sorted(output_dir.glob(f"*_{_slug(target)}.json"))
        for json_file in reversed(json_files):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if "market_snapshot" in data and "history" in data["market_snapshot"]:
                        snapshot = data["market_snapshot"]
                        break
            except Exception:
                pass

        baseline_forecast = build_baseline_forecast(target=target, snapshot=snapshot, config=config)
        ml_overlay = build_hybrid_ml_overlay(target=target, snapshot=snapshot, config=config)
        hybrid = apply_hybrid_overlay(baseline_forecast, ml_overlay.to_dict(), 0.0, {})

        result = {
            "prediction_label": "NEUTRAL",
            "confidence": 1.0,
            "summary": "Market is closed. No live prediction generated for this session.",
            "run_health": "HEALTHY",
            "prediction_date": prediction_date,
            "project_name": config["project_name"],
            "target": target,
            "html_path": str(html_path),
            "json_path": str(json_path),
            "market_projection": hybrid.projection,
            "sector_outlook": hybrid.sectors,
            "top_drivers": hybrid.top_drivers,
            "confidence_notes": hybrid.confidence_notes,
            "pipeline_stage_status": {
                "ingestion": "skipped_market_closed",
                "normalization": "skipped_market_closed",
                "feature_extraction": "complete",
                "statistical_decision": "skipped_market_closed",
                "challenge": "skipped_market_closed",
                "publish": "complete",
            },
            "stage_diagnostics": {
                "market_calendar": {"status": "closed", "prediction_date": prediction_date},
                "fallback_snapshot": {"history_series": len(snapshot.get("history", {}))},
            },
            "feature_snapshot_version": "daily_feature_snapshot:v2",
            "model_stack_version": "market_closed_fast_path:v1",
            "calibration_version": "confidence_calibration:v2",
            "agreement_features": {},
            "regime_slice": hybrid.regime_label,
            "forecast_summary": {
                "regime_label": hybrid.regime_label,
                "expected_return_30d": round(hybrid.expected_return_30d, 6),
                "expected_volatility_30d": round(hybrid.expected_volatility_30d, 6),
                "horizon_days": hybrid.horizon_days
            }
        }
        if scheduler_run:
            result["run_id"] = scheduler_run["prediction_run_id"]
            _complete_scheduler_run(
                config=config,
                scheduler_run=scheduler_run,
                target=target,
                prediction_date=prediction_date,
                run_id=result.get("run_id"),
                status="complete",
            )
        
        html_content = render_dashboard(result)
        json_content = json.dumps(result, indent=2)
        _atomic_write(html_path, html_content)
        _atomic_write(json_path, json_content)
        _persist_runtime_artifacts(output_dir=output_dir, stem=stem, result=result, html_content=html_content, json_content=json_content)
        graph_context = maybe_queue_graph_build(result=result, config=config, output_dir=output_dir, artifact_path=json_path)
        if graph_context:
            result.update(graph_context)
            _rewrite_result_artifacts(output_dir=output_dir, stem=stem, json_path=json_path, result=result)
        _finalize_runtime_success(
            output_dir=output_dir,
            stem=stem,
            result=result,
            runtime_state=dict(runtime_state),
            finalization_trace=list(finalization_trace),
        )
        terminal_state_written = True
        return result

    try:
        _write_runtime_state(
            status="running",
            prediction_date=prediction_date,
            target=target,
            llm_stage="source_collection",
            source_collection_status="running",
        )
        source_collection_started = time.monotonic()
        items, snapshot, warnings, source_diagnostics = collect_sources(
            prediction_date=prediction_date,
            target=target,
            config=config,
        )
        source_collection_duration = round(time.monotonic() - source_collection_started, 3)
        _write_runtime_state(
            source_collection_status="complete",
            source_collection_duration_seconds=source_collection_duration,
            source_item_count=len(items),
            quote_series_count=len(snapshot.get("series", {})),
            history_series_count=len(snapshot.get("history", {})),
        )
        ingestion_run_id = scheduler_run["prediction_run_id"] if scheduler_run else f"ingestion-{uuid4().hex[:12]}"
        ingestion_result = build_graph_first_ingestion(
            items=items,
            snapshot=snapshot,
            target=target,
            prediction_date=prediction_date,
            run_id=ingestion_run_id,
        )
        llm_progress = LLMProgressTracker()

        def _on_llm_progress(snapshot_payload: dict) -> None:
            _write_runtime_state(llm_progress=snapshot_payload)

        llm_progress.on_update = _on_llm_progress
        llm_client = with_progress_tracker(create_llm_client(config), llm_progress)
        signal_features = extract_signal_features(items, snapshot, graph=ingestion_result.graph)
        pre_simulation_graph = enrich_knowledge_graph(
            ingestion_result.graph,
            {
                "run_id": ingestion_run_id,
                "target": target,
                "prediction_date": prediction_date,
                "signal_features": [feature.to_dict() for feature in signal_features],
            },
        )
        graph_prediction_context = build_graph_prediction_context(pre_simulation_graph)
        items, quality_summary, source_diagnostics = apply_graph_quality_layer(
            items,
            DataQualitySummary(**source_diagnostics.get("data_quality_summary", {})),
            source_diagnostics,
            graph_prediction_context,
        )
        graph_feature_vector = build_graph_feature_vector(
            pre_simulation_graph,
            prediction_date=prediction_date,
            target=target,
        )
        previous_graph_summary = None
        if config.get("database_url"):
            try:
                with database_session(config["database_url"]) as session:
                    create_schema(session)
                    previous_graph_summary = MarketRepository(session).load_latest_daily_graph_summary(
                        target=target,
                        before_prediction_date=prediction_date,
                    )
            except Exception:
                previous_graph_summary = None
        graph_delta_summary = build_graph_delta_summary(
            prediction_date=prediction_date,
            target=target,
            current_features=graph_feature_vector.features,
            previous_summary=previous_graph_summary,
        )
        graph_feature_vector = graph_feature_vector.with_delta_summary(graph_delta_summary)
        stem = f"{prediction_date}_{_slug(target)}"
        prior_memory_snapshot = _load_latest_memory_snapshot(output_dir=output_dir, target=target, exclude_stem=stem)
        simulation_environment = prepare_swarm_environment(
            items=items,
            snapshot=snapshot,
            features=signal_features,
            llm_client=llm_client,
            config=config,
            target=target,
            graph=pre_simulation_graph,
            memory_snapshot=prior_memory_snapshot,
            mode="daily_run",
            base_run_stem=stem,
            prediction_date=prediction_date,
        )
        simulation_environment_path = _persist_simulation_environment(
            output_dir=output_dir,
            stem=stem,
            environment_payload=simulation_environment.to_dict(),
        )
        simulation_runner = SimulationRunner(output_dir)
        simulation_run_state_path = output_dir / "runs" / stem / "simulation_run_state.json"
        simulation_actions_path = output_dir / "runs" / stem / "simulation_actions.jsonl"

        def _simulation_progress(state) -> None:
            _write_runtime_state(
                status="running",
                prediction_date=prediction_date,
                target=target,
                simulation_id=state.simulation_id,
                simulation_status=state.status,
                simulation_round=state.current_round,
                simulation_total_rounds=state.total_rounds,
                llm_stage="swarm",
                llm_progress=llm_progress.snapshot(),
            )

        swarm_result = simulation_runner.run_sync(
            simulation_id=simulation_environment.environment_id,
            environment=simulation_environment,
            items=items,
            snapshot=snapshot,
            features=signal_features,
            llm_client=with_request_group(llm_client, "swarm"),
            config=config,
            target=target,
            state_path=simulation_run_state_path,
            actions_log_path=simulation_actions_path,
            mode="daily_run",
            persist_round_logs=bool(config.get("swarm_persist_round_logs", True)),
            progress_callback=_simulation_progress,
        )
        signal_features.extend(swarm_result.derived_features)
        _write_runtime_state(
            simulation_status="complete",
            simulation_round=swarm_result.setup.time_config.total_rounds,
            simulation_total_rounds=swarm_result.setup.time_config.total_rounds,
            llm_stage="source_agents",
            llm_progress=llm_progress.snapshot(),
        )
        source_agent_reports = run_source_agents(items, signal_features, with_request_group(llm_client, "source_agents"), target)
        _write_runtime_state(llm_stage="research_agents", llm_progress=llm_progress.snapshot())
        reports = run_research_agents(items, source_agent_reports, signal_features, with_request_group(llm_client, "research_agents"), target)
        _write_runtime_state(llm_stage="challenge_and_final_summary", llm_progress=llm_progress.snapshot())
        artifacts = synthesize_prediction(
            prediction_date=prediction_date,
            target=target,
            config=config,
            reports=reports,
            items=items,
            snapshot=snapshot,
            warnings=warnings,
            source_diagnostics=source_diagnostics,
            source_agent_reports=source_agent_reports,
            llm_client=llm_client,
            challenge_llm_client=with_request_group(llm_client, "challenge"),
            final_summary_llm_client=with_request_group(llm_client, "final_summary"),
            features=signal_features,
            backend_diagnostics={
                **llm_client.diagnostics(),
                "progress": llm_progress.snapshot(),
            },
            swarm_priors=swarm_result.priors,
            swarm_diagnostics=swarm_result.diagnostics,
            swarm_summary=swarm_result.summary_metrics,
            swarm_setup=swarm_result.setup.to_dict(),
            swarm_rounds=[round_result.to_dict() for round_result in swarm_result.rounds],
            swarm_agents=[profile.to_dict() for profile in swarm_result.profiles],
            simulation_environment_summary=simulation_environment.summary(),
            simulation_environment_path=simulation_environment_path,
            simulation_id=simulation_environment.environment_id,
            graph_prediction_context=graph_prediction_context,
            graph_feature_vector=graph_feature_vector,
            graph_delta_summary=graph_delta_summary.to_dict(),
        )
        _mark_finalization("final_summary_completed")
        
        if "sector_outlook" in snapshot and snapshot["sector_outlook"]:
            artifacts.sector_outlook = run_sector_agents(snapshot["sector_outlook"], llm_client)

        html_path = output_dir / f"{stem}.html"
        json_path = output_dir / f"{stem}.json"
        result = artifacts.to_dict()
        result["run_id"] = scheduler_run["prediction_run_id"] if scheduler_run else ingestion_run_id
        retraining = None
        if config.get("persist_to_db"):
            with database_session(config["database_url"]) as session:
                create_schema(session)
                repo = MarketRepository(session)
                retraining = run_weekly_retraining(
                    repo,
                    target=target,
                    snapshot=snapshot,
                    config=config,
                    as_of_date=prediction_date,
                    model_name="hybrid_v1",
                    horizon_days=int(result.get("forecast_summary", {}).get("horizon_days", 30) or 30),
                )
                result["model_version"] = retraining["active_model_version"]
        if "model_version" not in result:
            result["model_version"] = result.get("ensemble_diagnostics", {}).get("mode", "baseline-30d-v1")
        result["project_name"] = config["project_name"]
        result["html_path"] = str(html_path)
        result["json_path"] = str(json_path)
        result["simulation_environment_path"] = simulation_environment_path
        result["simulation_environment_summary"] = simulation_environment.summary()
        result["simulation_id"] = simulation_environment.environment_id
        result["simulation_run_state_path"] = str(simulation_run_state_path)
        result["simulation_actions_path"] = str(simulation_actions_path)
        knowledge_graph = enrich_knowledge_graph(pre_simulation_graph, result)
        result["knowledge_graph"] = knowledge_graph.to_dict()
        result["ingestion_graph"] = ingestion_result.to_dict()
        result["pipeline_stage_status"]["graph_ingestion"] = "complete"
        result["pipeline_stage_status"]["graph_priors"] = "complete"
        result["stage_diagnostics"]["graph_ingestion"] = ingestion_result.stage_diagnostics
        result["stage_diagnostics"]["source_collection"] = {
            "status": "complete",
            "duration_seconds": source_collection_duration,
            "source_item_count": len(items),
            "quote_series_count": len(snapshot.get("series", {})),
            "history_series_count": len(snapshot.get("history", {})),
        }
        result["stage_diagnostics"]["graph_priors"] = {
            "status": "complete",
            "influence_weight": graph_prediction_context.priors.influence_weight,
            "sparse_graph": graph_prediction_context.priors.sparse_graph,
            "evidence_nodes": graph_prediction_context.feature_summary.get("evidence_node_count", 0),
            "contradiction_score": graph_prediction_context.priors.contradiction_score,
        }
        result["stage_diagnostics"]["graph_quality"] = {
            "status": "complete",
            "graph_quality_score": quality_summary.graph_quality_summary.get("graph_quality_score"),
            "graph_adjusted_item_count": quality_summary.graph_adjusted_item_count,
            "severe_graph_risk": quality_summary.graph_quality_summary.get("severe_graph_risk"),
            "cluster_count": len(quality_summary.cluster_quality_summary),
        }
        result["stage_diagnostics"]["retrieval_assisted_features"] = {
            "status": "complete",
            "pass_1_feature_count": sum(
                1 for feature in result.get("signal_features", []) if feature.get("provenance", {}).get("feature_pass") == "pass_1"
            ),
            "pass_2_feature_count": sum(
                1 for feature in result.get("signal_features", []) if feature.get("provenance", {}).get("feature_pass") == "pass_2"
            ),
            "compound_feature_names": [
                feature.get("name")
                for feature in result.get("signal_features", [])
                if feature.get("provenance", {}).get("feature_pass") == "pass_2"
            ],
        }
        result["graph_feature_summary"] = {
            **(result.get("graph_feature_summary") or {}),
            **graph_feature_vector.to_summary_dict(),
        }
        result["graph_quality_summary"] = quality_summary.graph_quality_summary
        result["graph_delta_summary"] = graph_delta_summary.to_summary_dict()
        result["stage_diagnostics"]["graph_ml_features"] = {
            "status": "complete",
            "schema_version": graph_feature_vector.schema_version,
            "feature_count": len(graph_feature_vector.features),
            "sparse_graph": graph_feature_vector.sparse_graph,
            "availability_flag": graph_feature_vector.features.get("graph__feature_available", 0.0),
        }
        result["pipeline_stage_status"]["graph_temporal_deltas"] = "complete"
        result["stage_diagnostics"]["graph_temporal_deltas"] = {
            "status": "complete",
            "delta_available": graph_delta_summary.delta_available,
            "previous_prediction_date": graph_delta_summary.previous_prediction_date,
            "theme_acceleration": graph_delta_summary.theme_acceleration,
            "delta_strength": graph_delta_summary.delta_strength,
            "narrative_reversal_flag": graph_delta_summary.narrative_reversal_flag,
        }
        result["simulation_state"] = simulation_state_from_environment(
            simulation_environment.to_dict(),
            graph=knowledge_graph,
            snapshot=snapshot,
        ).to_dict()
        _write_runtime_state(llm_stage="publish", llm_progress=llm_progress.snapshot())

        serialization_started = time.monotonic()
        _mark_finalization("result_serialization_started")
        html_content = render_dashboard(result)
        json_content = json.dumps(result, indent=2)
        _mark_finalization(
            "result_serialization_completed",
            duration_seconds=round(time.monotonic() - serialization_started, 3),
            json_bytes=len(json_content.encode("utf-8")),
            html_bytes=len(html_content.encode("utf-8")),
        )
        _atomic_write(html_path, html_content)
        _atomic_write(json_path, json_content)
        _persist_runtime_artifacts(
            output_dir=output_dir,
            stem=stem,
            result=result,
            html_content=html_content,
            json_content=json_content,
            simulation_environment_path=simulation_environment_path,
            simulation_environment_id=simulation_environment.environment_id,
        )
        
        if config.get("persist_to_db"):
            db_started = time.monotonic()
            _mark_finalization("db_persistence_started")
            result["db_persistence"] = persist_daily_run(
                config=config,
                prediction_date=prediction_date,
                snapshot=snapshot,
                items=items,
                target=target,
                artifacts=result,
                graph_feature_vector={
                    **graph_feature_vector.to_dict(),
                    "generation_run_id": str(result.get("run_id") or ingestion_run_id),
                },
                graph_delta_summary={
                    **graph_delta_summary.to_dict(),
                    "generation_run_id": str(result.get("run_id") or ingestion_run_id),
                },
                run_evaluation=True,
            )
            result["db_persistence"]["retraining"] = retraining
            _mark_finalization(
                "db_persistence_completed",
                duration_seconds=round(time.monotonic() - db_started, 3),
            )
        if scheduler_run:
            _complete_scheduler_run(
                config=config,
                scheduler_run=scheduler_run,
                target=target,
                prediction_date=prediction_date,
                run_id=result.get("run_id"),
                status="complete",
            )
        _mark_finalization("runtime_state_final_update_started")
        _finalize_runtime_success(
            output_dir=output_dir,
            stem=stem,
            result=result,
            runtime_state=dict(runtime_state),
            finalization_trace=list(finalization_trace),
        )
        runtime_state.update(
            {
                "status": "complete",
                "run_id": result.get("run_id"),
                "prediction_label": result.get("prediction_label"),
                "run_health": result.get("run_health"),
            }
        )
        terminal_state_written = True
        _mark_finalization("runtime_state_final_update_completed")

        graph_timeout_seconds = float(config.get("post_finalize_timeout_seconds", 5.0) or 5.0)
        _mark_finalization("graph_queue_started", timeout_seconds=graph_timeout_seconds)
        graph_context, graph_error = _run_optional_with_timeout(
            lambda: maybe_queue_graph_build(
                result=result,
                config=config,
                output_dir=output_dir,
                artifact_path=json_path,
            ),
            timeout_seconds=graph_timeout_seconds,
            step_name="graph_queue",
        )
        if graph_context:
            result.update(graph_context)
            _rewrite_result_artifacts(output_dir=output_dir, stem=stem, json_path=json_path, result=result)
            _mark_finalization("graph_queue_completed")
        elif graph_error:
            _mark_finalization("graph_queue_failed", error_message=graph_error)
        else:
            _mark_finalization("graph_queue_skipped")
        _mark_finalization("function_return")
        return result
    except Exception as exc:
        if scheduler_run:
            _complete_scheduler_run(
                config=config,
                scheduler_run=scheduler_run,
                target=target,
                prediction_date=prediction_date,
                run_id=None,
                status="failed",
                error_message=str(exc),
            )
        _record_runtime_status(
            output_dir,
            {
                **runtime_state,
                "status": "failed",
                "prediction_date": prediction_date,
                "target": target,
                "error_message": str(exc),
                "updated_at": datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat(),
                "finalization_trace": list(finalization_trace),
            },
        )
        terminal_state_written = True
        _append_runtime_event(
            output_dir,
            {
                "event": "prediction_failed",
                "prediction_date": prediction_date,
                "target": target,
                "error_message": str(exc),
                "logged_at": datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat(),
            },
        )
        raise
    finally:
        if not terminal_state_written:
            try:
                _record_runtime_status(
                    output_dir,
                    {
                        **runtime_state,
                        "status": runtime_state.get("status", "failed"),
                        "prediction_date": prediction_date,
                        "target": target,
                        "updated_at": datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat(),
                        "finalization_trace": list(finalization_trace),
                    },
                )
            except Exception:
                pass


def run_prediction_scheduler(
    output_dir: str | Path,
    target: str = "S&P 500",
    run_time: str = "08:30",
    timezone_name: str = "America/New_York",
    max_runs: int | None = None,
    config_overrides: dict | None = None,
    run_now: bool = False,
    prediction_date: str | None = None,
) -> list[dict]:
    return run_scheduler(
        lambda: run_daily_prediction(
            output_dir=output_dir,
            target=target,
            prediction_date=prediction_date,
            config_overrides=(config_overrides or {}) | {"timezone": timezone_name},
            scheduler_context={
                "job_name": "daily_prediction",
                "scheduled_for": prediction_date or datetime.now(ZoneInfo(timezone_name)).replace(microsecond=0).isoformat(),
                "idempotency_key": f"daily_prediction:{target}:{prediction_date or datetime.now(ZoneInfo(timezone_name)).date().isoformat()}",
            },
        ),
        run_time=run_time,
        timezone_name=timezone_name,
        max_runs=max_runs,
        run_now=run_now,
    )


def run_dashboard_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    index_column: str,
    date_column: str | None = None,
    live_as_of: str | None = None,
    overrides: dict[str, float] | None = None,
) -> dict:
    del input_path, date_column, overrides
    output_path = Path(output_path)
    result = run_daily_prediction(
        output_dir=output_path.parent,
        target=index_column,
        prediction_date=live_as_of,
    )
    rendered_path = Path(result["html_path"])
    if rendered_path != output_path:
        _atomic_write(output_path, rendered_path.read_text(encoding="utf-8"))
        result["html_path"] = str(output_path)
    return result


def _slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _atomic_write(path: str | Path, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.{uuid4().hex[:8]}.tmp")
    tmp_path.write_text(content, encoding="utf-8")
    os.replace(tmp_path, path)


def _run_optional_with_timeout(
    operation,
    *,
    timeout_seconds: float,
    step_name: str,
) -> tuple[object | None, str | None]:
    result: dict[str, object] = {}
    error: dict[str, str] = {}

    def _runner() -> None:
        try:
            result["value"] = operation()
        except Exception as exc:  # pragma: no cover - validated via caller behavior
            error["message"] = f"{step_name}_error: {exc}"

    worker = threading.Thread(target=_runner, daemon=True, name=f"optional-{step_name}")
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        return None, f"{step_name}_timeout_after_{timeout_seconds:.1f}s"
    if "message" in error:
        return None, error["message"]
    return result.get("value"), None


def _ensure_runtime_layout(output_dir: Path) -> None:
    (output_dir / "runs").mkdir(parents=True, exist_ok=True)
    (output_dir / "_state").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)


def _persist_runtime_artifacts(
    *,
    output_dir: Path,
    stem: str,
    result: dict,
    html_content: str,
    json_content: str,
    simulation_environment_path: str = "",
    simulation_environment_id: str = "",
) -> None:
    run_dir = output_dir / "runs" / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    run_html = run_dir / f"{stem}.html"
    run_json = run_dir / f"{stem}.json"
    manifest_path = run_dir / "artifact_manifest.json"

    _atomic_write(run_html, html_content)
    _atomic_write(run_json, json_content)

    manifest = {
        "stem": stem,
        "prediction_date": result.get("prediction_date"),
        "target": result.get("target"),
        "run_id": result.get("run_id"),
        "prediction_label": result.get("prediction_label"),
        "run_health": result.get("run_health"),
        "generated_at": datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "root_json_path": str(output_dir / f"{stem}.json"),
        "root_html_path": str(output_dir / f"{stem}.html"),
        "run_json_path": str(run_json),
        "run_html_path": str(run_html),
        "db_path": result.get("db_persistence", {}).get("database_url"),
        "source_file": f"{stem}.json",
        "report_file": f"{stem}.html",
        "simulation_environment_path": simulation_environment_path,
        "simulation_environment_id": simulation_environment_id,
        "simulation_run_state_path": result.get("simulation_run_state_path"),
        "simulation_actions_path": result.get("simulation_actions_path"),
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=2))


def _rewrite_result_artifacts(*, output_dir: Path, stem: str, json_path: Path, result: dict) -> None:
    json_content = json.dumps(result, indent=2)
    _atomic_write(json_path, json_content)
    run_json = output_dir / "runs" / stem / f"{stem}.json"
    _atomic_write(run_json, json_content)


def _persist_simulation_environment(*, output_dir: Path, stem: str, environment_payload: dict) -> str:
    run_dir = output_dir / "runs" / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    environment_path = run_dir / "simulation_environment.json"
    _atomic_write(environment_path, json.dumps(environment_payload, indent=2))
    return str(environment_path)


def _load_latest_memory_snapshot(*, output_dir: Path, target: str, exclude_stem: str) -> dict:
    target_slug = _slug(target)
    candidates = sorted(
        path
        for path in (output_dir / "runs").glob(f"*_{target_slug}/simulation_memory_snapshot.json")
        if path.parent.name != exclude_stem
    )
    if not candidates:
        return {}
    latest = candidates[-1]
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def _record_runtime_status(output_dir: Path, payload: dict, filename: str = "runtime_status.json") -> None:
    _atomic_write(output_dir / "_state" / filename, json.dumps(payload, indent=2))


def _append_runtime_event(output_dir: Path, payload: dict) -> None:
    log_path = output_dir / "logs" / "pipeline-events.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _finalize_runtime_success(
    *,
    output_dir: Path,
    stem: str,
    result: dict,
    runtime_state: dict | None = None,
    finalization_trace: list[dict] | None = None,
) -> None:
    timestamp = datetime.now().astimezone().replace(microsecond=0).isoformat()
    latest_payload = {
        "prediction_date": result.get("prediction_date"),
        "target": result.get("target"),
        "run_id": result.get("run_id"),
        "prediction_label": result.get("prediction_label"),
        "run_health": result.get("run_health"),
        "root_json_path": str(output_dir / f"{stem}.json"),
        "root_html_path": str(output_dir / f"{stem}.html"),
        "run_dir": str(output_dir / "runs" / stem),
        "manifest_path": str(output_dir / "runs" / stem / "artifact_manifest.json"),
        "updated_at": timestamp,
    }
    if finalization_trace:
        latest_payload["finalization_trace"] = finalization_trace
    _record_runtime_status(output_dir, latest_payload, filename="latest_run.json")
    terminal_payload = {
        **(runtime_state or {}),
        "status": "complete",
        "prediction_date": result.get("prediction_date"),
        "target": result.get("target"),
        "run_id": result.get("run_id"),
        "prediction_label": result.get("prediction_label"),
        "run_health": result.get("run_health"),
        "updated_at": timestamp,
    }
    if finalization_trace:
        terminal_payload["finalization_trace"] = finalization_trace
    _record_runtime_status(output_dir, terminal_payload)
    _append_runtime_event(
        output_dir,
        {
            "event": "prediction_completed",
            "prediction_date": result.get("prediction_date"),
            "target": result.get("target"),
            "run_id": result.get("run_id"),
            "prediction_label": result.get("prediction_label"),
            "run_health": result.get("run_health"),
            "logged_at": timestamp,
        },
    )


def _prepare_scheduler_run(*, config: dict, target: str, prediction_date: str, scheduler_context: dict | None) -> dict | None:
    if not scheduler_context or not config.get("persist_to_db"):
        return None
    started_at = datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat()
    idempotency_key = scheduler_context.get("idempotency_key") or f"daily_prediction:{target}:{prediction_date}"
    prediction_run_id = f"{_slug(target)}:{prediction_date}"
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        existing = repo.get_scheduler_run_by_key(idempotency_key)
        attempt_count = int(existing["attempt_count"]) + 1 if existing else 1
        scheduler_run_id = existing["scheduler_run_id"] if existing else f"scheduler:{uuid4().hex[:12]}"
        repo.upsert_scheduler_run(
            SchedulerRunRecord(
                scheduler_run_id=scheduler_run_id,
                job_name=scheduler_context.get("job_name", "daily_prediction"),
                scheduled_for=scheduler_context.get("scheduled_for"),
                started_at=started_at,
                completed_at=None,
                status="running",
                target=target,
                prediction_date=prediction_date,
                run_id=existing["run_id"] if existing else None,
                error_message=None,
                attempt_count=attempt_count,
                idempotency_key=idempotency_key,
            )
        )
    return {
        "scheduler_run_id": scheduler_run_id,
        "idempotency_key": idempotency_key,
        "prediction_run_id": prediction_run_id,
    }


def _complete_scheduler_run(
    *,
    config: dict,
    scheduler_run: dict,
    target: str,
    prediction_date: str,
    run_id: str | None,
    status: str,
    error_message: str | None = None,
) -> None:
    completed_at = datetime.now(ZoneInfo(config["timezone"])).replace(microsecond=0).isoformat()
    with database_session(config["database_url"]) as session:
        create_schema(session)
        repo = MarketRepository(session)
        existing = repo.get_scheduler_run_by_key(scheduler_run["idempotency_key"])
        attempt_count = int(existing["attempt_count"]) if existing else 1
        repo.upsert_scheduler_run(
            SchedulerRunRecord(
                scheduler_run_id=scheduler_run["scheduler_run_id"],
                job_name="daily_prediction",
                scheduled_for=existing["scheduled_for"] if existing else prediction_date,
                started_at=existing["started_at"] if existing else completed_at,
                completed_at=completed_at,
                status=status,
                target=target,
                prediction_date=prediction_date,
                run_id=run_id,
                error_message=error_message,
                attempt_count=attempt_count,
                idempotency_key=scheduler_run["idempotency_key"],
            )
        )
