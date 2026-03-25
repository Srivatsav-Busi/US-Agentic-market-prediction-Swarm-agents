from __future__ import annotations

import argparse

from .config import load_config
from .graph import backfill_graphs, build_graph_for_run
from .pipeline import run_daily_prediction, run_prediction_scheduler
from .pipelines.bootstrap_history import bootstrap_history


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a US market prediction pipeline driven by daily agent research.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    daily = subparsers.add_parser("run-daily", help="Run one daily prediction and write JSON + HTML outputs.")
    daily.add_argument("--output-dir", required=True, help="Directory for timestamped JSON and HTML outputs.")
    daily.add_argument("--target", default="S&P 500", help="Primary US market target to predict.")
    daily.add_argument("--date", help="Prediction date in YYYY-MM-DD format.")
    daily.add_argument("--provider", choices=["rule_based", "openrouter"], default="openrouter", help="Narrative backend for optional agent summaries.")
    daily.add_argument("--model", help="Optional model name for the selected provider.")
    daily.add_argument("--database-url", help="Database URL for optional persistence, e.g. sqlite:///./results/market_intelligence.db")
    daily.add_argument("--persist-db", action="store_true", help="Persist latest snapshot, evidence, and feature rows into the local database.")

    bootstrap = subparsers.add_parser("bootstrap-history", help="Create schema and backfill historical market intelligence data.")
    bootstrap.add_argument("--years", type=int, default=2, help="Years of history to backfill.")
    bootstrap.add_argument("--target", default="S&P 500", help="Primary target used for feature-scope defaults.")
    bootstrap.add_argument("--database-url", help="Database URL for persistence, e.g. sqlite:///./results/market_intelligence.db")
    bootstrap.add_argument("--diagnostics-path", help="Optional JSON path for bootstrap diagnostics.")

    graph = subparsers.add_parser("build-graph", help="Build a knowledge graph for a completed run.")
    graph.add_argument("--run-id", required=True, help="Run id to graph.")
    graph.add_argument("--results-dir", default="results", help="Directory containing generated JSON outputs.")
    graph.add_argument("--database-url", help="Database URL for graph lifecycle state.")

    backfill = subparsers.add_parser("backfill-graphs", help="Queue graph builds for historical runs.")
    backfill.add_argument("--results-dir", default="results", help="Directory containing generated JSON outputs.")
    backfill.add_argument("--database-url", help="Database URL for graph lifecycle state.")
    backfill.add_argument("--limit", type=int, help="Optional limit on queued historical graphs.")

    scheduler = subparsers.add_parser("scheduler", help="Run the built-in weekday morning scheduler.")
    scheduler.add_argument("--output-dir", required=True, help="Directory for timestamped JSON and HTML outputs.")
    scheduler.add_argument("--target", default="S&P 500", help="Primary US market target to predict.")
    scheduler.add_argument("--time", default="08:30", help="Weekday run time in HH:MM.")
    scheduler.add_argument("--timezone", default="America/New_York", help="Scheduler timezone.")
    scheduler.add_argument("--max-runs", type=int, help="Optional cap for completed runs.")
    scheduler.add_argument("--provider", choices=["rule_based", "openrouter"], default="openrouter", help="Narrative backend for optional agent summaries.")
    scheduler.add_argument("--model", help="Optional model name for the selected provider.")

    ui = subparsers.add_parser("serve-ui", help="Serve the built React frontend with live result endpoints.")
    ui.add_argument("--frontend-dir", default="frontend/dist", help="Built frontend directory to serve.")
    ui.add_argument("--results-dir", default="results", help="Directory containing generated JSON and HTML outputs.")
    ui.add_argument("--host", default="127.0.0.1", help="Bind host.")
    ui.add_argument("--port", type=int, default=8000, help="Bind port.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-daily":
        config_overrides = {
            "llm_provider": args.provider,
            "llm_model": args.model,
            "database_url": args.database_url,
            "persist_to_db": args.persist_db,
        }
        run_daily_prediction(
            output_dir=args.output_dir,
            target=args.target,
            prediction_date=args.date,
            config_overrides=config_overrides,
        )
        return

    if args.command == "bootstrap-history":
        config = load_config({"database_url": args.database_url} if args.database_url else None)
        diagnostics = bootstrap_history(
            config=config,
            years=args.years,
            target=args.target,
            diagnostics_path=args.diagnostics_path,
        )
        print(f"Bootstrapped history into {config['database_url']}")
        print(f"Instruments: {diagnostics['instrument_count']}")
        print(f"Daily prices: {diagnostics['daily_price_count']}")
        print(f"Macro rows: {diagnostics['macro_series_count']}")
        print(f"Feature snapshots: {diagnostics['feature_snapshot_count']}")
        return

    if args.command == "serve-ui":
        from .webapp import serve_app

        serve_app(
            frontend_dir=args.frontend_dir,
            results_dir=args.results_dir,
            host=args.host,
            port=args.port,
        )
        return

    if args.command == "build-graph":
        config = load_config({"database_url": args.database_url, "results_dir": args.results_dir})
        context = build_graph_for_run(run_id=args.run_id, config=config, results_dir=args.results_dir)
        print(f"Queued graph build for {context.run_id}")
        print(f"Project: {context.project_id}")
        print(f"Task: {context.task_id}")
        return

    if args.command == "backfill-graphs":
        config = load_config({"database_url": args.database_url, "results_dir": args.results_dir})
        contexts = backfill_graphs(config=config, results_dir=args.results_dir, limit=args.limit)
        print(f"Queued {len(contexts)} graph builds")
        for context in contexts[:10]:
            print(f"{context.run_id} -> {context.task_id}")
        return

    config_overrides = {"llm_provider": args.provider, "llm_model": args.model}
    run_prediction_scheduler(
        output_dir=args.output_dir,
        target=args.target,
        run_time=args.time,
        timezone_name=args.timezone,
        max_runs=args.max_runs,
        config_overrides=config_overrides,
    )


if __name__ == "__main__":
    main()
