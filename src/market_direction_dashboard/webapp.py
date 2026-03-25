from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

from .agents import synthesize_prediction
from .graph import build_graph_for_run, get_graph_for_run, get_graph_status, list_graph_history, resume_pending_graph_builds
from .llm_clients import create_llm_client, is_real_llm_client, with_timeout
from .models import AgentReport, SignalFeature, SourceAgentReport, SourceItem
from .reporting import get_analysis_report_result, get_analysis_report_status, queue_analysis_report, summarize_analysis_report
from .simulation_reporting import build_simulation_history_report
from .simulation_runner import SimulationRunner
from .swarm_simulation import prepare_swarm_environment


def _interpret_scenario_fast(scenario: str) -> SignalFeature | None:
    scenario_lower = scenario.lower().strip()
    if not scenario_lower:
        return None

    def make_feature(name: str, category: str, direction: str, strength: float, summary: str) -> SignalFeature:
        return SignalFeature(
            name=name,
            category=category,
            direction=direction,
            strength=strength,
            summary=summary,
            supporting_evidence_ids=["sim-fast"],
        )

    if "inflation" in scenario_lower or "hot" in scenario_lower:
        return make_feature(
            "simulated_inflation_shock",
            "economic",
            "bearish",
            0.95,
            f"Scenario Injection: {scenario}. Massive spike in price pressure detected.",
        )
    if "cut" in scenario_lower or "dovish" in scenario_lower:
        return make_feature(
            "simulated_monetary_easing",
            "economic",
            "bullish",
            0.88,
            f"Scenario Injection: {scenario}. Central bank pivots to aggressive support.",
        )

    bullish_score = 0.0
    bearish_score = 0.0
    category = "market"

    keyword_map = [
        ("yield falls", "economic", "bullish", 0.32),
        ("yields fall", "economic", "bullish", 0.32),
        ("bond yields fall", "economic", "bullish", 0.34),
        ("treasury yield falls", "economic", "bullish", 0.34),
        ("oil reverses", "market", "bullish", 0.18),
        ("oil falls", "market", "bullish", 0.18),
        ("inflation cools", "economic", "bullish", 0.30),
        ("spreads tighten", "market", "bullish", 0.26),
        ("vix falls", "market", "bullish", 0.22),
        ("vix drops", "market", "bullish", 0.22),
        ("dollar weakens", "market", "bullish", 0.18),
        ("earnings beat", "market", "bullish", 0.24),
        ("yield spikes", "economic", "bearish", 0.36),
        ("yields jump", "economic", "bearish", 0.36),
        ("oil spikes", "market", "bearish", 0.22),
        ("inflation reaccelerates", "economic", "bearish", 0.34),
        ("growth scare", "market", "bearish", 0.28),
        ("credit spreads widen", "market", "bearish", 0.28),
        ("vix spikes", "market", "bearish", 0.24),
        ("dollar breaks out", "market", "bearish", 0.16),
        ("geopolitical escalation", "political", "bearish", 0.26),
        ("policy shock", "political", "bearish", 0.22),
    ]

    for phrase, mapped_category, direction, score in keyword_map:
        if phrase in scenario_lower:
            category = mapped_category
            if direction == "bullish":
                bullish_score += score
            else:
                bearish_score += score

    if "treasury" in scenario_lower or "bond" in scenario_lower or "yield" in scenario_lower:
        category = "economic"
    elif "election" in scenario_lower or "tariff" in scenario_lower or "policy" in scenario_lower:
        category = "political"
    elif "sentiment" in scenario_lower or "retail" in scenario_lower:
        category = "social"

    if bullish_score == 0.0 and bearish_score == 0.0:
        return make_feature(
            "custom_scenario_balanced",
            category,
            "neutral",
            0.0,
            f"Scenario Injection: {scenario}. No strong directional rule matched, so the base artifact was preserved.",
        )

    if bullish_score >= bearish_score:
        return make_feature(
            "custom_scenario_bullish",
            category,
            "bullish",
            min(0.92, round(0.45 + bullish_score, 2)),
            f"Scenario Injection: {scenario}. Fast parser identified a net bullish market impulse.",
        )

    return make_feature(
        "custom_scenario_bearish",
        category,
        "bearish",
        min(0.92, round(0.45 + bearish_score, 2)),
        f"Scenario Injection: {scenario}. Fast parser identified a net bearish market impulse.",
    )


def _persist_simulation_environment(*, results_root: Path, base_stem: str, simulation_id: str, environment_payload: dict) -> str:
    simulation_dir = results_root / "simulations" / base_stem / simulation_id
    simulation_dir.mkdir(parents=True, exist_ok=True)
    environment_path = simulation_dir / "simulation_environment.json"
    environment_path.write_text(json.dumps(environment_payload, indent=2), encoding="utf-8")
    return str(environment_path)


def _simulation_dir(results_root: Path, base_stem: str, simulation_id: str) -> Path:
    return results_root / "simulations" / base_stem / simulation_id


def _load_simulation_context(results_root: Path, payload: dict) -> dict:
    latest_json_path = _find_latest_result(results_root)
    if latest_json_path is None:
        raise FileNotFoundError("No base result found to simulate from")
    base_data = json.loads(latest_json_path.read_text(encoding="utf-8"))
    scenario = str(payload.get("scenario") or "")
    simulated_features = [SignalFeature(**f) if isinstance(f, dict) else f for f in base_data.get("signal_features", [])]
    config = _load_runtime_config()
    llm_client = with_timeout(
        create_llm_client(config),
        int(config.get("swarm_llm_timeout_seconds", config.get("llm_api_timeout_seconds", 30)) or 10),
    )
    if not is_real_llm_client(llm_client):
        raise RuntimeError("Simulation requires a configured OpenRouter LLM backend.")
    interpreted_feature = _interpret_scenario_fast(scenario)
    if interpreted_feature:
        simulated_features.insert(0, interpreted_feature)
    items = [SourceItem(**item) for item in base_data.get("sources", [])]
    source_agent_reports = [SourceAgentReport(**report) for report in base_data.get("source_agent_reports", [])]
    reports = {
        cat: AgentReport(
            name=f"{cat}_agent",
            category=cat,
            summary=base_data.get(f"{cat}_report", "Baseline"),
            bullish_points=[],
            bearish_points=[],
            score=0.0,
            confidence=0.5,
            dominant_regime_label="mixed",
        )
        for cat in ["economic", "political", "social", "market"]
    }
    return {
        "latest_json_path": latest_json_path,
        "latest_stem": latest_json_path.stem,
        "base_data": base_data,
        "scenario": scenario,
        "config": config,
        "llm_client": llm_client,
        "items": items,
        "source_agent_reports": source_agent_reports,
        "reports": reports,
        "simulated_features": simulated_features,
    }


def _build_simulation_prediction_payload(
    *,
    base_data: dict,
    scenario: str,
    config: dict,
    llm_client,
    items: list[SourceItem],
    reports: dict[str, AgentReport],
    source_agent_reports: list[SourceAgentReport],
    simulated_features: list[SignalFeature],
    simulation_environment,
    simulation_environment_path: str,
    simulation_id: str,
    swarm_result,
) -> dict:
    combined_features = list(simulated_features) + list(swarm_result.derived_features)
    artifacts = synthesize_prediction(
        prediction_date=base_data["prediction_date"],
        target=base_data["target"],
        config=config,
        reports=reports,
        items=items,
        snapshot=base_data["market_snapshot"],
        warnings=base_data.get("warnings", []) + [f"SIMULATED SCENARIO: {scenario}"],
        source_diagnostics=base_data["source_diagnostics"],
        source_agent_reports=source_agent_reports,
        llm_client=llm_client,
        features=combined_features,
        backend_diagnostics=llm_client.diagnostics(),
        swarm_priors=swarm_result.priors,
        swarm_diagnostics=swarm_result.diagnostics,
        swarm_summary=swarm_result.summary_metrics,
        swarm_setup=swarm_result.setup.to_dict(),
        swarm_rounds=[round_result.to_dict() for round_result in swarm_result.rounds],
        swarm_agents=[profile.to_dict() for profile in swarm_result.profiles],
        simulation_environment_summary=simulation_environment.summary(),
        simulation_environment_path=simulation_environment_path,
        simulation_id=simulation_id,
    )
    result_payload = artifacts.to_dict()
    result_payload["run_health"] = "SIMULATED"
    result_payload["summary"] = f"SIMULATED SCENARIO ANALYSIS: {scenario}\n\n{result_payload['summary']}"
    return result_payload


def _load_runtime_config() -> dict:
    from .config import load_config

    return load_config()


def create_app_handler_class(frontend_dir: str | Path, results_dir: str | Path) -> type[SimpleHTTPRequestHandler]:
    frontend_path = Path(frontend_dir).resolve()
    results_path = Path(results_dir).resolve()
    simulation_runner = SimulationRunner(results_path)

    class AppHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(frontend_path), **kwargs)

        def do_GET(self) -> None:
            parsed_url = urlparse(self.path)
            request_path = parsed_url.path
            query = parse_qs(parsed_url.query)

            if request_path == "/api/latest-result":
                self._handle_latest_result(results_path)
                return
            if request_path == "/api/forecast-history":
                self._handle_forecast_history(results_path)
                return
            if request_path == "/api/sector-outlook":
                self._handle_sector_outlook(results_path)
                return
            if request_path == "/api/market-projection":
                self._handle_market_projection(results_path)
                return
            if request_path == "/api/model-diagnostics":
                self._handle_model_diagnostics(results_path)
                return
            if request_path == "/api/graph/latest":
                self._handle_graph_latest()
                return
            if request_path == "/api/graph/history":
                self._handle_graph_history()
                return
            if request_path == "/api/graph/run":
                self._handle_graph_run(query)
                return
            if request_path == "/api/graph/status":
                self._handle_graph_status(query)
                return
            if request_path == "/api/simulations/status":
                self._handle_simulation_status(query)
                return
            if request_path == "/api/simulations/result":
                self._handle_simulation_result(query)
                return
            if request_path == "/api/simulations/interaction-context":
                self._handle_simulation_interaction_context(query)
                return
            if request_path == "/api/simulations/runtime-inspection":
                self._handle_simulation_runtime_inspection(query)
                return
            if request_path == "/api/simulations/memory":
                self._handle_simulation_memory(query)
                return
            if request_path == "/api/simulations/decision-traces":
                self._handle_simulation_decision_traces(query)
                return
            if request_path == "/api/simulations/social-dynamics":
                self._handle_simulation_social_dynamics(query)
                return
            if request_path == "/api/report/status":
                self._handle_report_status(results_path, query)
                return
            if request_path == "/api/report/result":
                self._handle_report_result(results_path, query)
                return
            if request_path == "/api/report/followups":
                self._handle_report_followups(results_path, query)
                return
            if request_path == "/api/graph/introspection":
                self._handle_graph_introspection(query)
                return

            if request_path.startswith("/reports/"):
                self._handle_report(results_path, request_path)
                return

            if request_path.startswith("/api/"):
                self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
                return

            super().do_GET()

        def do_POST(self) -> None:
            request_path = self.path.split("?", 1)[0]
            if request_path == "/api/simulate":
                self._handle_simulate(results_path)
                return
            if request_path == "/api/interview":
                self._handle_interview(results_path)
                return
            if request_path == "/api/report/chat":
                self._handle_report_chat(results_path)
                return
            if request_path == "/api/report/generate":
                self._handle_report_generate(results_path)
                return
            if request_path == "/api/graph/build":
                self._handle_graph_build(results_path)
                return
            if request_path == "/api/simulations/start":
                self._handle_simulation_start(results_path, simulation_runner)
                return
            if request_path == "/api/simulations/interview":
                self._handle_simulation_interview(simulation_runner)
                return
            if request_path == "/api/simulations/interview/batch":
                self._handle_simulation_interview_batch(simulation_runner)
                return
            if request_path == "/api/simulations/interview/all":
                self._handle_simulation_interview_all(simulation_runner)
                return

            self.send_error(HTTPStatus.NOT_FOUND, "Unknown POST endpoint")

        def send_head(self):
            path = self.translate_path(self.path)
            target = Path(path)

            if target.exists():
                return super().send_head()

            index_path = frontend_path / "index.html"
            if index_path.exists():
                content = index_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                return _BytesIOWithClose(content)

            self.send_error(HTTPStatus.NOT_FOUND, "Frontend build not found")
            return None

        def _handle_latest_result(self, results_root: Path) -> None:
            latest_json = _find_latest_result(results_root)
            if latest_json is None:
                self._send_json({"error": "No result artifacts found"}, status=HTTPStatus.NOT_FOUND)
                return

            payload = _build_frontend_payload(latest_json)
            payload["analysis_report"] = summarize_analysis_report(run_id=payload.get("run_id"), results_dir=results_root)
            self._send_json(payload)

        def _handle_forecast_history(self, results_root: Path) -> None:
            payloads = _load_all_results(results_root)
            history = []
            for payload in payloads:
                summary = payload.get("forecast_summary", {})
                history.append(
                    {
                        "run_date": payload.get("prediction_date"),
                        "prediction_label": payload.get("prediction_label"),
                        "confidence": payload.get("confidence"),
                        "model_version": payload.get("ensemble_diagnostics", {}).get("mode", "baseline"),
                        "expected_return_30d": summary.get("expected_return_30d"),
                        "expected_volatility_30d": summary.get("expected_volatility_30d"),
                        "regime_label": summary.get("regime_label"),
                        "reportPath": payload.get("reportPath"),
                        "run_health": payload.get("run_health"),
                        "target": payload.get("target"),
                        "source_file": payload.get("sourceFile"),
                        "analysis_report": summarize_analysis_report(run_id=payload.get("run_id"), results_dir=results_root),
                    }
                )
            self._send_json({"items": history})

        def _handle_sector_outlook(self, results_root: Path) -> None:
            latest_json = _find_latest_result(results_root)
            if latest_json is None:
                self._send_json({"error": "No result artifacts found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = _build_frontend_payload(latest_json)
            payload["analysis_report"] = summarize_analysis_report(run_id=payload.get("run_id"), results_dir=results_root)
            self._send_json(
                {
                    "run_date": payload.get("prediction_date"),
                    "target": payload.get("target"),
                    "items": payload.get("sector_outlook", []),
                }
            )

        def _handle_market_projection(self, results_root: Path) -> None:
            latest_json = _find_latest_result(results_root)
            if latest_json is None:
                self._send_json({"error": "No result artifacts found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = _build_frontend_payload(latest_json)
            payload["analysis_report"] = summarize_analysis_report(run_id=payload.get("run_id"), results_dir=results_root)
            self._send_json(
                {
                    "run_date": payload.get("prediction_date"),
                    "target": payload.get("target"),
                    "forecast_summary": payload.get("forecast_summary", {}),
                    "market_projection": payload.get("market_projection", {}),
                }
            )

        def _handle_model_diagnostics(self, results_root: Path) -> None:
            latest_json = _find_latest_result(results_root)
            if latest_json is None:
                self._send_json({"error": "No result artifacts found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = _build_frontend_payload(latest_json)
            payload["analysis_report"] = summarize_analysis_report(run_id=payload.get("run_id"), results_dir=results_root)
            self._send_json(
                {
                    "run_date": payload.get("prediction_date"),
                    "target": payload.get("target"),
                    "regime_probabilities": payload.get("regime_probabilities", {}),
                    "expected_volatility": payload.get("expected_volatility"),
                    "history_coverage": payload.get("history_coverage", {}),
                    "statistical_failures": payload.get("statistical_failures", []),
                    "run_health": payload.get("run_health"),
                    "confidence_notes": payload.get("confidence_notes", []),
                    "top_drivers": payload.get("top_drivers", []),
                    "ensemble_diagnostics": payload.get("ensemble_diagnostics", {}),
                    "data_quality_summary": payload.get("data_quality_summary", {}),
                }
            )

        def _handle_graph_latest(self) -> None:
            from .config import load_config

            history = list_graph_history(config=load_config())
            self._send_json({"latest": history[0] if history else None, "items": history})

        def _handle_graph_history(self) -> None:
            from .config import load_config

            self._send_json({"items": list_graph_history(config=load_config())})

        def _handle_graph_run(self, query: dict[str, list[str]]) -> None:
            from .config import load_config

            run_id = (query.get("run_id") or [""])[0]
            if not run_id:
                self._send_json({"error": "run_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            payload = get_graph_for_run(run_id, config=load_config())
            if payload is None:
                self._send_json({"error": "No graph found for run"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)

        def _handle_graph_status(self, query: dict[str, list[str]]) -> None:
            from .config import load_config

            task_id = (query.get("task_id") or [""])[0] or None
            run_id = (query.get("run_id") or [""])[0] or None
            payload = get_graph_status(task_id=task_id, run_id=run_id, config=load_config())
            if payload is None:
                self._send_json({"error": "No graph task found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)

        def _handle_report(self, results_root: Path, request_path: str) -> None:
            report_name = unquote(request_path.removeprefix("/reports/"))
            report_path = (results_root / report_name).resolve()
            if results_root not in report_path.parents and report_path != results_root:
                self.send_error(HTTPStatus.FORBIDDEN, "Invalid report path")
                return

            if not report_path.exists() or not report_path.is_file():
                generated_root = (results_root / "reports").resolve()
                report_path = (generated_root / report_name).resolve()
                if generated_root not in report_path.parents and report_path != generated_root:
                    self.send_error(HTTPStatus.FORBIDDEN, "Invalid report path")
                    return
                if not report_path.exists() or not report_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND, "Report not found")
                    return

            media_type = mimetypes.guess_type(report_path.name)[0] or "application/octet-stream"
            content = report_path.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", media_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _handle_simulate(self, results_root: Path) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json({"error": "Missing JSON body"}, status=HTTPStatus.BAD_REQUEST)
                return
            
            try:
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                context = _load_simulation_context(results_root, payload)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return
            except Exception:
                self._send_json({"error": "Failed to load base result"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if not context["config"].get("swarm_background_enabled", True):
                self._send_json({"error": "Background simulations are disabled."}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return

            simulation_id = f"sim_{uuid4().hex[:12]}"
            simulation_environment = prepare_swarm_environment(
                items=context["items"],
                snapshot=context["base_data"]["market_snapshot"],
                features=context["simulated_features"],
                llm_client=context["llm_client"],
                config=context["config"],
                target=context["base_data"]["target"],
                mode="scenario_simulation",
                base_run_stem=context["latest_stem"],
                prediction_date=context["base_data"]["prediction_date"],
            )
            simulation_environment_path = _persist_simulation_environment(
                results_root=results_root,
                base_stem=context["latest_stem"],
                simulation_id=simulation_id,
                environment_payload=simulation_environment.to_dict(),
            )
            simulation_paths = _simulation_dir(results_root, context["latest_stem"], simulation_id)
            state_path = simulation_paths / "run_state.json"
            actions_path = simulation_paths / "actions.jsonl"
            result_path = simulation_paths / "result.json"

            swarm_result = simulation_runner.run_sync(
                simulation_id=simulation_id,
                environment=simulation_environment,
                items=context["items"],
                snapshot=context["base_data"]["market_snapshot"],
                features=context["simulated_features"],
                llm_client=context["llm_client"],
                config=context["config"],
                target=context["base_data"]["target"],
                state_path=state_path,
                actions_log_path=actions_path,
                result_path=result_path,
                mode="scenario_simulation",
                persist_round_logs=bool(context["config"].get("swarm_persist_round_logs", True)),
            )
            result_payload = _build_simulation_prediction_payload(
                base_data=context["base_data"],
                scenario=context["scenario"],
                config=context["config"],
                llm_client=context["llm_client"],
                items=context["items"],
                reports=context["reports"],
                source_agent_reports=context["source_agent_reports"],
                simulated_features=context["simulated_features"],
                simulation_environment=simulation_environment,
                simulation_environment_path=simulation_environment_path,
                simulation_id=simulation_id,
                swarm_result=swarm_result,
            )
            result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
            self._send_json(result_payload)

        def _handle_simulation_start(self, results_root: Path, runner: SimulationRunner) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                self._send_json({"error": "Missing JSON body"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                raw_body = self.rfile.read(content_length)
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                context = _load_simulation_context(results_root, payload)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return
            except Exception:
                self._send_json({"error": "Failed to load base result"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            simulation_id = f"sim_{uuid4().hex[:12]}"
            simulation_environment = prepare_swarm_environment(
                items=context["items"],
                snapshot=context["base_data"]["market_snapshot"],
                features=context["simulated_features"],
                llm_client=context["llm_client"],
                config=context["config"],
                target=context["base_data"]["target"],
                mode="scenario_simulation",
                base_run_stem=context["latest_stem"],
                prediction_date=context["base_data"]["prediction_date"],
            )
            simulation_environment_path = _persist_simulation_environment(
                results_root=results_root,
                base_stem=context["latest_stem"],
                simulation_id=simulation_id,
                environment_payload=simulation_environment.to_dict(),
            )
            simulation_paths = _simulation_dir(results_root, context["latest_stem"], simulation_id)
            state_path = simulation_paths / "run_state.json"
            actions_path = simulation_paths / "actions.jsonl"
            result_path = simulation_paths / "result.json"

            def build_result(swarm_result) -> dict:
                return _build_simulation_prediction_payload(
                    base_data=context["base_data"],
                    scenario=context["scenario"],
                    config=context["config"],
                    llm_client=context["llm_client"],
                    items=context["items"],
                    reports=context["reports"],
                    source_agent_reports=context["source_agent_reports"],
                    simulated_features=context["simulated_features"],
                    simulation_environment=simulation_environment,
                    simulation_environment_path=simulation_environment_path,
                    simulation_id=simulation_id,
                    swarm_result=swarm_result,
                )

            state = runner.start_background(
                simulation_id=simulation_id,
                environment=simulation_environment,
                items=context["items"],
                snapshot=context["base_data"]["market_snapshot"],
                features=context["simulated_features"],
                llm_client=context["llm_client"],
                config=context["config"],
                target=context["base_data"]["target"],
                state_path=state_path,
                actions_log_path=actions_path,
                result_path=result_path,
                result_builder=build_result,
                mode="scenario_simulation_background",
                persist_round_logs=bool(context["config"].get("swarm_persist_round_logs", True)),
            )
            self._send_json(
                {
                    "simulation_id": simulation_id,
                    "status": state.status,
                    "status_path": str(state_path),
                    "result_path": str(result_path),
                    "simulation_environment_path": simulation_environment_path,
                },
                status=HTTPStatus.ACCEPTED,
            )

        def _handle_simulation_status(self, query: dict[str, list[str]]) -> None:
            simulation_id = (query.get("simulation_id") or [""])[0]
            if not simulation_id:
                self._send_json({"error": "simulation_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            payload = simulation_runner.get_status(simulation_id)
            if payload is None:
                self._send_json({"error": "No simulation found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)

        def _handle_simulation_result(self, query: dict[str, list[str]]) -> None:
            simulation_id = (query.get("simulation_id") or [""])[0]
            if not simulation_id:
                self._send_json({"error": "simulation_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            status_payload = simulation_runner.get_status(simulation_id)
            if status_payload is None:
                self._send_json({"error": "No simulation found"}, status=HTTPStatus.NOT_FOUND)
                return
            if status_payload.get("status") != "complete":
                self._send_json(status_payload, status=HTTPStatus.CONFLICT)
                return
            payload = simulation_runner.load_result(simulation_id)
            if payload is None:
                self._send_json({"error": "Simulation result not found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload["analysis_report"] = summarize_analysis_report(simulation_id=simulation_id, results_dir=results_path)
            self._send_json(payload)

        def _handle_report_generate(self, results_root: Path) -> None:
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                context = queue_analysis_report(
                    results_dir=results_root,
                    run_id=str(payload.get("run_id") or "").strip() or None,
                    simulation_id=str(payload.get("simulation_id") or "").strip() or None,
                    force_rebuild=bool(payload.get("force_rebuild")),
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except RuntimeError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.SERVICE_UNAVAILABLE)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            response_payload = summarize_analysis_report(report_id=context.report_id, results_dir=results_root)
            self._send_json(response_payload or {"report_id": context.report_id}, status=HTTPStatus.ACCEPTED)

        def _handle_report_status(self, results_root: Path, query: dict[str, list[str]]) -> None:
            try:
                payload = get_analysis_report_status(
                    report_id=(query.get("report_id") or [""])[0] or None,
                    run_id=(query.get("run_id") or [""])[0] or None,
                    simulation_id=(query.get("simulation_id") or [""])[0] or None,
                    results_dir=results_root,
                )
            except ValueError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
                return
            if payload is None:
                self._send_json({"error": "No report found"}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(payload)

        def _handle_report_result(self, results_root: Path, query: dict[str, list[str]]) -> None:
            report_id = (query.get("report_id") or [""])[0]
            if not report_id:
                self._send_json({"error": "report_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            payload = get_analysis_report_result(report_id=report_id, results_dir=results_root)
            if payload is None:
                self._send_json({"error": "No report found"}, status=HTTPStatus.NOT_FOUND)
                return
            if payload.get("status") != "completed":
                self._send_json(payload, status=HTTPStatus.CONFLICT)
                return
            self._send_json(payload)

        def _handle_simulation_interaction_context(self, query: dict[str, list[str]]) -> None:
            simulation_id = (query.get("simulation_id") or [""])[0] or None
            bundle = simulation_runner.load_interaction_bundle(simulation_id)
            if bundle is None or not bundle.get("environment"):
                self._send_json({"error": "No simulation interaction context found"}, status=HTTPStatus.NOT_FOUND)
                return
            timeline = _build_simulation_timeline(bundle.get("actions", []))
            self._send_json(
                {
                    "simulation_id": bundle["simulation_id"],
                    "run_state": bundle.get("run_state"),
                    "environment_summary": _build_environment_summary(bundle.get("environment") or {}),
                    "profiles": (bundle.get("environment") or {}).get("profiles", []),
                    "round_count": len(timeline),
                    "action_count": sum(len(round_payload.get("actions", [])) for round_payload in timeline),
                    "ready_for_interview": bool((bundle.get("environment") or {}).get("profiles")),
                }
            )

        def _handle_simulation_runtime_inspection(self, query: dict[str, list[str]]) -> None:
            simulation_id = (query.get("simulation_id") or [""])[0] or None
            bundle = simulation_runner.load_interaction_bundle(simulation_id)
            if bundle is None or not bundle.get("environment"):
                self._send_json({"error": "No simulation interaction context found"}, status=HTTPStatus.NOT_FOUND)
                return
            payload = _build_runtime_inspection_payload(bundle)
            self._send_json(payload)

        def _handle_simulation_memory(self, query: dict[str, list[str]]) -> None:
            bundle = simulation_runner.load_interaction_bundle((query.get("simulation_id") or [""])[0] or None)
            if bundle is None:
                self._send_json({"error": "No simulation found"}, status=HTTPStatus.NOT_FOUND)
                return
            analytics = build_simulation_history_report(bundle)
            self._send_json(
                {
                    "simulation_id": bundle["simulation_id"],
                    "memory_snapshot": bundle.get("memory_snapshot") or {},
                    "memory_state": ((bundle.get("simulation_state") or {}).get("memory_state") or {}),
                    "memory_summary": analytics.get("memory"),
                }
            )

        def _handle_simulation_decision_traces(self, query: dict[str, list[str]]) -> None:
            bundle = simulation_runner.load_interaction_bundle((query.get("simulation_id") or [""])[0] or None)
            if bundle is None:
                self._send_json({"error": "No simulation found"}, status=HTTPStatus.NOT_FOUND)
                return
            decision_traces = (((bundle.get("simulation_state") or {}).get("agent_state") or {}).get("decision_traces") or {})
            analytics = build_simulation_history_report(bundle)
            self._send_json(
                {
                    "simulation_id": bundle["simulation_id"],
                    "decision_traces": decision_traces,
                    "summary": analytics.get("decision_traces"),
                }
            )

        def _handle_simulation_social_dynamics(self, query: dict[str, list[str]]) -> None:
            bundle = simulation_runner.load_interaction_bundle((query.get("simulation_id") or [""])[0] or None)
            if bundle is None:
                self._send_json({"error": "No simulation found"}, status=HTTPStatus.NOT_FOUND)
                return
            analytics = build_simulation_history_report(bundle)
            self._send_json(
                {
                    "simulation_id": bundle["simulation_id"],
                    "social_dynamics": (((bundle.get("simulation_state") or {}).get("agent_state") or {}).get("social_dynamics") or {}),
                    "summary": analytics.get("social_dynamics"),
                }
            )

        def _handle_report_followups(self, results_root: Path, query: dict[str, list[str]]) -> None:
            report_id = (query.get("report_id") or [""])[0] or None
            simulation_id = (query.get("simulation_id") or [""])[0] or None
            run_id = (query.get("run_id") or [""])[0] or None
            payload = None
            if report_id:
                payload = get_analysis_report_result(report_id=report_id, results_dir=results_root)
            else:
                summary = summarize_analysis_report(report_id=None, simulation_id=simulation_id, run_id=run_id, results_dir=results_root)
                if summary and summary.get("report_id"):
                    payload = get_analysis_report_result(report_id=summary["report_id"], results_dir=results_root)
            if payload is None:
                self._send_json({"error": "No report found"}, status=HTTPStatus.NOT_FOUND)
                return
            structured_summary = payload.get("structured_summary") or {}
            self._send_json(
                {
                    "report_id": payload.get("report_id"),
                    "status": payload.get("status"),
                    "follow_up_questions": structured_summary.get("follow_up_questions", []),
                }
            )

        def _handle_graph_introspection(self, query: dict[str, list[str]]) -> None:
            bundle = simulation_runner.load_interaction_bundle((query.get("simulation_id") or [""])[0] or None)
            if bundle is None:
                self._send_json({"error": "No simulation found"}, status=HTTPStatus.NOT_FOUND)
                return
            analytics = build_simulation_history_report(bundle)
            self._send_json(
                {
                    "simulation_id": bundle["simulation_id"],
                    "graph_deltas": ((bundle.get("simulation_state") or {}).get("graph_deltas") or []),
                    "summary": analytics.get("graph"),
                    "queue": analytics.get("queue"),
                }
            )

        def _handle_report_chat(self, results_root: Path) -> None:
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            message = str(payload.get("message") or "").strip()
            if not message:
                self._send_json({"error": "message is required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                response_payload = _run_report_chat(
                    results_root=results_root,
                    message=message,
                    run_id=str(payload.get("run_id") or "").strip() or None,
                    chat_history=payload.get("chat_history") or [],
                    focus_category=str(payload.get("focus_category") or "").strip() or None,
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(response_payload)

        def _handle_interview(self, results_root: Path) -> None:
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            agent_category = str(payload.get("agent_category") or "").strip()
            user_prompt = str(payload.get("prompt") or "").strip()
            if not agent_category or not user_prompt:
                self._send_json({"error": "agent_category and prompt are required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                response_payload = _run_report_chat(
                    results_root=results_root,
                    message=user_prompt,
                    run_id=None,
                    chat_history=payload.get("chat_history") or [],
                    focus_category=agent_category,
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(
                {
                    "agent": agent_category,
                    "response": response_payload["response"],
                    "diagnostics": response_payload["diagnostics"],
                }
            )

        def _handle_simulation_interview(self, runner: SimulationRunner) -> None:
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            simulation_id = str(payload.get("simulation_id") or "").strip()
            agent_id = str(payload.get("agent_id") or "").strip()
            message = str(payload.get("message") or "").strip()
            if not simulation_id or not agent_id or not message:
                self._send_json({"error": "simulation_id, agent_id, and message are required"}, status=HTTPStatus.BAD_REQUEST)
                return
            try:
                response_payload = _run_simulation_agent_interview(
                    runner=runner,
                    simulation_id=simulation_id,
                    agent_id=agent_id,
                    message=message,
                    chat_history=payload.get("chat_history") or [],
                )
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            self._send_json(response_payload)

        def _handle_simulation_interview_batch(self, runner: SimulationRunner) -> None:
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            simulation_id = str(payload.get("simulation_id") or "").strip()
            agent_ids = [str(agent_id).strip() for agent_id in payload.get("agent_ids", []) if str(agent_id).strip()]
            message = str(payload.get("message") or "").strip()
            if not simulation_id or not agent_ids or not message:
                self._send_json({"error": "simulation_id, agent_ids, and message are required"}, status=HTTPStatus.BAD_REQUEST)
                return
            items = []
            for agent_id in agent_ids:
                try:
                    items.append(
                        _run_simulation_agent_interview(
                            runner=runner,
                            simulation_id=simulation_id,
                            agent_id=agent_id,
                            message=message,
                            chat_history=payload.get("chat_history") or [],
                        )
                    )
                except FileNotFoundError as exc:
                    self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                    return
            self._send_json({"simulation_id": simulation_id, "items": items})

        def _handle_simulation_interview_all(self, runner: SimulationRunner) -> None:
            payload = self._read_json_body()
            if payload is None:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return
            simulation_id = str(payload.get("simulation_id") or "").strip()
            message = str(payload.get("message") or "").strip()
            if not simulation_id or not message:
                self._send_json({"error": "simulation_id and message are required"}, status=HTTPStatus.BAD_REQUEST)
                return
            bundle = runner.load_interaction_bundle(simulation_id)
            if bundle is None or not bundle.get("environment"):
                self._send_json({"error": "Simulation context not found"}, status=HTTPStatus.NOT_FOUND)
                return
            agent_ids = [str(profile.get("agent_id") or "").strip() for profile in (bundle["environment"] or {}).get("profiles", []) if profile.get("agent_id")]
            items = []
            for agent_id in agent_ids:
                items.append(
                    _run_simulation_agent_interview(
                        runner=runner,
                        simulation_id=simulation_id,
                        agent_id=agent_id,
                        message=message,
                        chat_history=payload.get("chat_history") or [],
                    )
                )
            self._send_json({"simulation_id": simulation_id, "items": items})

        def _handle_graph_build(self, results_root: Path) -> None:
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length) if content_length else b"{}"
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except Exception:
                self._send_json({"error": "Invalid JSON"}, status=HTTPStatus.BAD_REQUEST)
                return

            latest_json = _find_latest_result(results_root)
            latest_payload = _build_frontend_payload(latest_json) if latest_json else {}
            run_id = str(payload.get("run_id") or latest_payload.get("run_id") or "").strip()
            if not run_id:
                self._send_json({"error": "run_id is required"}, status=HTTPStatus.BAD_REQUEST)
                return

            from .config import load_config

            config = load_config({"results_dir": str(results_root)})
            try:
                context = build_graph_for_run(run_id=run_id, config=config, results_dir=results_root)
            except FileNotFoundError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return

            self._send_json(
                {
                    "project_id": context.project_id,
                    "task_id": context.task_id,
                    "run_id": context.run_id,
                    "artifact_path": context.artifact_path,
                    "status": "queued",
                },
                status=HTTPStatus.ACCEPTED,
            )

        def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict | None:
            content_length = int(self.headers.get("Content-Length", 0))
            if content_length == 0:
                return {}
            try:
                raw_body = self.rfile.read(content_length)
                return json.loads(raw_body.decode("utf-8"))
            except Exception:
                return None

    return AppHandler


def create_app_server(
    frontend_dir: str | Path,
    results_dir: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), create_app_handler_class(frontend_dir, results_dir))


def serve_app(frontend_dir: str | Path, results_dir: str | Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    resume_pending_graph_builds(config={"results_dir": str(results_dir)}, results_dir=results_dir)
    server = create_app_server(frontend_dir=frontend_dir, results_dir=results_dir, host=host, port=port)
    print(f"Serving frontend from {Path(frontend_dir).resolve()}")
    print(f"Serving results from {Path(results_dir).resolve()}")
    print(f"Open http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the React frontend and result artifacts from one HTTP server.")
    parser.add_argument("--frontend-dir", default="frontend/dist", help="Built frontend directory to serve.")
    parser.add_argument("--results-dir", default="results", help="Directory containing generated JSON and HTML results.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    serve_app(frontend_dir=args.frontend_dir, results_dir=args.results_dir, host=args.host, port=args.port)


def _find_latest_result(results_root: Path) -> Path | None:
    candidates = sorted(results_root.glob("*.json"))
    if not candidates:
        return None
    return candidates[-1]


def _load_all_results(results_root: Path) -> list[dict]:
    payloads = []
    for path in sorted(results_root.glob("*.json")):
        try:
            payloads.append(_build_frontend_payload(path))
        except json.JSONDecodeError:
            continue
    return payloads


def _build_frontend_payload(json_path: Path) -> dict:
    raw = json_path.read_text(encoding="utf-8")
    sanitized = re.sub(r"\bNaN\b|\bInfinity\b|-Infinity\b", "null", raw)
    parsed = json.loads(sanitized)
    report_name = json_path.name.replace(".json", ".html")
    report_path = json_path.with_suffix(".html")
    parsed["sourceFile"] = json_path.name
    parsed["source_file"] = json_path.name
    parsed["reportPath"] = f"/reports/{report_name}" if report_path.exists() else None
    parsed["analysis_report"] = None
    return parsed


def _load_result_by_run_id(results_root: Path, run_id: str | None) -> tuple[Path, dict]:
    if not run_id:
        latest_json = _find_latest_result(results_root)
        if latest_json is None:
            raise FileNotFoundError("No result found to provide context")
        return latest_json, _build_frontend_payload(latest_json)
    for path in sorted(results_root.glob("*.json")):
        try:
            payload = _build_frontend_payload(path)
        except json.JSONDecodeError:
            continue
        resolved_run_id = str(payload.get("run_id") or path.stem)
        if resolved_run_id == run_id:
            return path, payload
    raise FileNotFoundError(f"No result found for run_id: {run_id}")


def _format_chat_history(chat_history: list[dict], *, limit: int = 6) -> str:
    turns = []
    for item in chat_history[-limit:]:
        role = str(item.get("role") or "user").strip().lower()
        content = str(item.get("content") or "").strip()
        if not content:
            continue
        turns.append(f"{role.upper()}: {content}")
    return "\n".join(turns)


def _bounded_lines(values: list[str], *, limit: int = 8) -> str:
    cleaned = [value.strip() for value in values if str(value).strip()]
    return "\n".join(f"- {value}" for value in cleaned[:limit])


def _extract_report_sections(payload: dict, focus_category: str | None = None) -> list[str]:
    category_keys = {
        "economic": "economic_report",
        "political": "political_report",
        "social": "social_report",
        "market": "market_context_report",
    }
    if focus_category:
        key = category_keys.get(focus_category, "")
        return [str(payload.get(key) or "")]
    ordered_keys = [
        "economic_report",
        "political_report",
        "social_report",
        "market_context_report",
        "summary",
    ]
    return [str(payload.get(key) or "") for key in ordered_keys]


def _build_report_chat_citations(payload: dict, json_path: Path, focus_category: str | None = None) -> list[dict]:
    citations = []
    if payload.get("reportPath"):
        citations.append({"type": "report", "label": "Latest report", "ref": payload["reportPath"]})
    if payload.get("run_id"):
        citations.append({"type": "run", "label": "Run ID", "ref": str(payload["run_id"])})
    for item in (payload.get("top_drivers") or [])[:3]:
        name = str(item.get("name") or "").strip()
        if name:
            citations.append({"type": "driver", "label": name, "ref": name})
    for item in (payload.get("sources") or [])[:3]:
        title = str(item.get("title") or item.get("id") or "").strip()
        if title and (not focus_category or item.get("category") == focus_category):
            citations.append({"type": "source", "label": title, "ref": str(item.get("id") or title)})
    if not citations:
        citations.append({"type": "artifact", "label": json_path.name, "ref": json_path.name})
    return citations[:6]


def _run_report_chat(
    *,
    results_root: Path,
    message: str,
    run_id: str | None,
    chat_history: list[dict],
    focus_category: str | None,
) -> dict:
    json_path, payload = _load_result_by_run_id(results_root, run_id)
    config = _load_runtime_config()
    llm_client = create_llm_client(config)
    graph_summary = None
    if payload.get("run_id"):
        try:
            graph_summary = get_graph_for_run(str(payload["run_id"]), config=config)
        except Exception:
            graph_summary = None
    report_sections = _extract_report_sections(payload, focus_category)
    source_lines = []
    for item in payload.get("sources", [])[:8]:
        if focus_category and item.get("category") != focus_category:
            continue
        source_lines.append(f"{item.get('title', item.get('id', 'source'))}: {item.get('summary', '')}")
    driver_lines = [f"{item.get('name', 'driver')}: {item.get('summary', '')}" for item in (payload.get("top_drivers") or [])[:5]]
    notes = [str(note) for note in (payload.get("confidence_notes") or [])[:5]]
    system_prompt = (
        "You are a market forecast follow-up analyst. Answer only from the completed artifact context, "
        "call out uncertainty when the artifact is incomplete, and do not roleplay as a persona."
    )
    if focus_category:
        system_prompt += f" Focus on the {focus_category} desk perspective unless the user asks for cross-desk comparison."
    user_prompt = "\n\n".join(
        part
        for part in [
            f"Run ID: {payload.get('run_id') or json_path.stem}",
            f"Target: {payload.get('target', 'Unknown')}",
            f"Prediction Label: {payload.get('prediction_label', 'Unknown')}",
            f"Summary:\n{_bounded_lines(report_sections, limit=5)}",
            f"Top Drivers:\n{_bounded_lines(driver_lines, limit=5)}",
            f"Confidence Notes:\n{_bounded_lines(notes, limit=5)}",
            f"Source Evidence:\n{_bounded_lines(source_lines, limit=8)}",
            (
                f"Graph Summary:\n{str((graph_summary or {}).get('ontology', {}).get('analysis_summary') or '').strip()}"
                if graph_summary
                else ""
            ),
            f"Recent Chat:\n{_format_chat_history(chat_history)}",
            f"User Question: {message}",
        ]
        if part.strip()
    )
    fallback = str(payload.get("summary") or "The current artifact does not contain enough context to answer that follow-up confidently.")
    response = llm_client.summarize(system_prompt=system_prompt, user_prompt=user_prompt, fallback=fallback)
    return {
        "response": response,
        "citations": _build_report_chat_citations(payload, json_path, focus_category),
        "diagnostics": llm_client.diagnostics(),
        "resolved_run_id": str(payload.get("run_id") or json_path.stem),
    }


def _build_environment_summary(environment: dict) -> dict:
    return {
        "environment_id": environment.get("environment_id"),
        "mode": environment.get("mode"),
        "target": environment.get("target"),
        "prediction_date": environment.get("prediction_date"),
        "profile_count": len(environment.get("profiles", [])),
        "seed_post_count": len(environment.get("seed_posts", [])),
        "round_count": int(((environment.get("time_config") or {}).get("total_rounds") or 0)),
    }


def _build_simulation_timeline(action_rounds: list[dict]) -> list[dict]:
    timeline = []
    for payload in action_rounds:
        timeline.append(
            {
                "round_index": int(payload.get("round_index") or 0),
                "summary": payload.get("summary", ""),
                "active_agent_ids": payload.get("active_agent_ids", []),
                "consensus_score": payload.get("consensus_score"),
                "conflict_score": payload.get("conflict_score"),
                "actions": payload.get("actions", []),
            }
        )
    return timeline


def _build_agent_stats(environment: dict, timeline: list[dict]) -> list[dict]:
    profiles = {profile.get("agent_id"): profile for profile in environment.get("profiles", []) if profile.get("agent_id")}
    stats = {
        agent_id: {
            "agent_id": agent_id,
            "name": profile.get("name"),
            "stance_bias": profile.get("stance_bias"),
            "action_count": 0,
            "post_count": 0,
            "comment_count": 0,
            "rounds_active": set(),
            "evidence_ids": set(),
        }
        for agent_id, profile in profiles.items()
    }
    for round_payload in timeline:
        round_index = int(round_payload.get("round_index") or 0)
        for action in round_payload.get("actions", []):
            agent_id = action.get("agent_id")
            if not agent_id:
                continue
            payload = stats.setdefault(
                agent_id,
                {
                    "agent_id": agent_id,
                    "name": agent_id,
                    "stance_bias": "neutral",
                    "action_count": 0,
                    "post_count": 0,
                    "comment_count": 0,
                    "rounds_active": set(),
                    "evidence_ids": set(),
                },
            )
            payload["action_count"] += 1
            if action.get("action_type") == "create_post":
                payload["post_count"] += 1
            else:
                payload["comment_count"] += 1
            payload["rounds_active"].add(round_index)
            payload["evidence_ids"].update(action.get("referenced_evidence_ids", []))
    items = []
    for payload in stats.values():
        items.append(
            {
                **payload,
                "rounds_active": sorted(payload["rounds_active"]),
                "evidence_ids": sorted(payload["evidence_ids"]),
            }
        )
    items.sort(key=lambda item: (-item["action_count"], item["agent_id"]))
    return items


def _build_runtime_inspection_payload(bundle: dict) -> dict:
    environment = bundle.get("environment") or {}
    timeline = _build_simulation_timeline(bundle.get("actions", []))
    agent_stats = _build_agent_stats(environment, timeline)
    result = bundle.get("result") or {}
    simulation_state = bundle.get("simulation_state") or {}
    analytics = build_simulation_history_report(bundle)
    return {
        "simulation_id": bundle["simulation_id"],
        "run_state": bundle.get("run_state"),
        "queue_record": bundle.get("queue_record"),
        "analytics": analytics,
        "swarm_reporting": (result.get("swarm_reporting") or analytics.get("swarm_reporting") or {}),
        "environment_summary": _build_environment_summary(environment),
        "profiles": environment.get("profiles", []),
        "timeline": timeline,
        "agent_stats": agent_stats,
        "seed_posts": environment.get("seed_posts", []),
        "memory_snapshot": bundle.get("memory_snapshot") or {},
        "simulation_state": simulation_state,
        "graph_deltas": simulation_state.get("graph_deltas", []),
        "event_history": simulation_state.get("event_history", []),
        "decision_traces": ((simulation_state.get("agent_state") or {}).get("decision_traces") or {}),
        "social_dynamics": ((simulation_state.get("agent_state") or {}).get("social_dynamics") or {}),
        "ready_for_interview": bool(environment.get("profiles")),
        "dominant_stance": (result.get("swarm_summary") or {}).get("dominant_stance"),
    }


def _find_profile(environment: dict, agent_id: str) -> dict | None:
    for profile in environment.get("profiles", []):
        if profile.get("agent_id") == agent_id:
            return profile
    return None


def _run_simulation_agent_interview(
    *,
    runner: SimulationRunner,
    simulation_id: str,
    agent_id: str,
    message: str,
    chat_history: list[dict],
) -> dict:
    bundle = runner.load_interaction_bundle(simulation_id)
    if bundle is None or not bundle.get("environment"):
        raise FileNotFoundError("Simulation context not found")
    environment = bundle["environment"] or {}
    profile = _find_profile(environment, agent_id)
    if profile is None:
        raise FileNotFoundError(f"No agent found for agent_id: {agent_id}")
    timeline = _build_simulation_timeline(bundle.get("actions", []))
    agent_rounds = []
    agent_actions = []
    for round_payload in timeline:
        relevant_actions = [action for action in round_payload.get("actions", []) if action.get("agent_id") == agent_id]
        if relevant_actions:
            agent_rounds.append(int(round_payload.get("round_index") or 0))
            agent_actions.extend(relevant_actions)
    config = _load_runtime_config()
    llm_client = create_llm_client(config)
    system_prompt = (
        f"You are {profile.get('name', agent_id)}, a persisted market simulation persona. "
        "Answer in first person using only the saved simulation profile, actions, and final artifact context. "
        "Do not claim to know anything beyond the persisted runtime."
    )
    recent_actions = [
        f"round {action.get('round_index', '?') + 1}: {action.get('action_type', 'action')} | {action.get('content', '')}"
        for action in agent_actions[-6:]
    ]
    result = bundle.get("result") or {}
    simulation_state = bundle.get("simulation_state") or {}
    memory_snapshot = bundle.get("memory_snapshot") or {}
    decision_trace = (((simulation_state.get("agent_state") or {}).get("decision_traces") or {}).get(agent_id) or [])
    memory_entry = next((entry for entry in (memory_snapshot.get("individual") or []) if entry.get("agent_id") == agent_id), {})
    user_prompt = "\n\n".join(
        part
        for part in [
            f"Simulation ID: {simulation_id}",
            f"Target: {environment.get('target', 'Unknown')}",
            f"Persona: {profile.get('bio') or profile.get('persona') or profile.get('archetype') or profile.get('name', agent_id)}",
            f"Stance Bias: {profile.get('stance_bias', 'neutral')}",
            f"Focus Categories: {', '.join(profile.get('focus_categories', []))}",
            f"Runtime Summary: {result.get('summary', '')}",
            f"Recent Memory Actions: {len(memory_entry.get('recent_actions', []))}",
            f"Decision Trace Count: {len(decision_trace)}",
            f"Recent Agent Actions:\n{_bounded_lines(recent_actions, limit=6)}",
            f"Recent Chat:\n{_format_chat_history(chat_history)}",
            f"User Question: {message}",
        ]
        if part.strip()
    )
    fallback = profile.get("bio") or profile.get("persona") or f"{profile.get('name', agent_id)} has no saved response context beyond the persisted simulation profile."
    response = llm_client.summarize(system_prompt=system_prompt, user_prompt=user_prompt, fallback=fallback)
    return {
        "agent_id": agent_id,
        "response": response,
        "used_rounds": agent_rounds,
        "used_action_count": len(agent_actions),
        "decision_trace_count": len(decision_trace),
        "memory_action_count": len(memory_entry.get("recent_actions", [])),
        "diagnostics": llm_client.diagnostics(),
    }


class _BytesIOWithClose(BytesIO):
    pass

if __name__ == "__main__":
    main()
