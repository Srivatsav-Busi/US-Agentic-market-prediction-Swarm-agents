# US Agentic Market Prediction

This repository contains the production source for a market-intelligence pipeline and local dashboard focused on next-session `S&P 500` direction. It generates a daily JSON artifact, a static HTML report, and a local web UI for inspection.

## Repo Layout

- `src/market_direction_dashboard/`: Python application code, CLI, pipeline, web server, graph integration, and reporting.
- `frontend/`: React/Vite source for the local UI.
- `run_app.sh`: Main local entrypoint. Optionally runs the pipeline, builds the frontend, and serves the UI.
- `pyproject.toml`: Python packaging and CLI entrypoint.
- `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh`: Container support.

## Requirements

- Python `3.11+`
- Node.js with `npm`
- A populated `.venv` for Python dependencies
- Optional local `.env` for graph or provider-specific configuration

## Main Commands

Run one daily prediction:

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli run-daily \
  --output-dir results \
  --target "S&P 500" \
  --date 2026-03-18 \
  --provider openrouter \
  --persist-db \
  --database-url "sqlite:///results/market_intelligence.db"
```

Serve the built UI:

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli serve-ui \
  --frontend-dir frontend/dist \
  --results-dir results \
  --host 127.0.0.1 \
  --port 8000
```

Run the combined local flow:

```bash
./run_app.sh
```

Run the combined flow and generate a fresh daily artifact first:

```bash
RUN_PIPELINE=true ./run_app.sh
```

Trigger a graph build manually for a completed run:

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli build-graph \
  --run-id ingestion-a9fcad2d23af \
  --results-dir results \
  --database-url "sqlite:///results/market_intelligence.db"
```

## Typical Local Rerun

Rerun today's prediction, then serve the UI:

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli run-daily \
  --output-dir results \
  --target "S&P 500" \
  --date 2026-03-18 \
  --provider openrouter \
  --persist-db \
  --database-url "sqlite:///results/market_intelligence.db"

PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli serve-ui \
  --frontend-dir frontend/dist \
  --results-dir results \
  --host 127.0.0.1 \
  --port 8000
```

Local UI: `http://127.0.0.1:8000`

If the daily run finishes but `results/_state/runtime_status.json` or `results/logs/pipeline-events.jsonl` shows `graph_queue_skipped`, queue the graph manually with `build-graph`. The graph task state is persisted in the local SQLite database under `graph_projects`, `graph_build_tasks`, and `graph_snapshots`.

## Runtime Outputs

Runtime artifacts are written under `results/` and are intentionally not tracked:

- `YYYY-MM-DD_s_p_500.json`
- `YYYY-MM-DD_s_p_500.html`
- `results/runs/`
- `results/_state/`
- `results/logs/`
- local SQLite files used by persistence features

## Notes

- The app creates the runtime directory structure on demand.
- The frontend is source-controlled; `frontend/dist` is generated at build time.
- Graph features depend on local Neo4j Aura settings. Configure those with untracked environment variables or a local credentials file such as `neo4j-credentials.local.txt`, and keep `.env` files out of version control.
- A completed graph build writes both to the configured Neo4j backend and to the local SQLite graph lifecycle tables for inspection.
- `build-graph` queues work on a background thread. If you run it from a short-lived process, keep that process alive long enough for the task to complete, or resume pending graph builds from a longer-lived process such as the local web app.
