# US Agentic Market Prediction

<p align="center">
  <img src="assets/banner.png" alt="US Agentic Market Prediction" width="100%">
</p>

An agentic market-intelligence system for generating next-session US equity market forecasts, structured research artifacts, and a local dashboard for review. The project combines market data ingestion, feature generation, forecasting, reporting, and optional graph-backed research workflows in a single local-first stack.

## Overview

This repository is designed around a daily research and prediction loop for the US market, with `S&P 500` as the default target. A typical run produces:

- a daily JSON artifact with forecast and evidence
- a self-contained HTML report
- local SQLite persistence for run state and artifacts
- a web UI for browsing the latest outputs
- optional graph build workflows for knowledge extraction and retrieval

The codebase is primarily Python for the prediction pipeline and backend services, with a React/Vite frontend for the dashboard.

## Core Capabilities

- Daily market prediction pipeline driven by market, macro, and narrative inputs
- Local artifact generation in JSON and HTML formats
- Forecasting utilities, simulation components, and reporting helpers
- Optional persistence to SQLite for run history and graph lifecycle state
- Knowledge graph build and backfill commands for completed runs
- Local web UI for reviewing latest outputs and runtime state
- Docker support for containerized local execution

## Repository Structure

- `src/market_direction_dashboard/` Python application code, CLI, pipeline orchestration, graph workflows, storage, and web serving
- `frontend/` React/Vite frontend source
- `tests/` automated test coverage for swarm and graph-related flows
- `run_app.sh` local entrypoint that can run the pipeline, build the frontend, and serve the UI
- `Dockerfile`, `docker-compose.yml`, `docker-entrypoint.sh` container workflow assets
- `pyproject.toml` package metadata and CLI entrypoint

## Technology Stack

- Python `3.11+`
- React + Vite
- SQLite
- `pandas`, `numpy`, `scikit-learn`, `statsmodels`, `yfinance`
- Optional Neo4j integration for graph workflows

## Getting Started

### Prerequisites

- Python `3.11` or later
- Node.js and `npm`
- A local virtual environment at `.venv`

### Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cd frontend
npm install
cd ..
```

## Running the Project

### 1. Run a Daily Prediction

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli run-daily \
  --output-dir results \
  --target "S&P 500" \
  --date 2026-03-25 \
  --provider openrouter \
  --persist-db \
  --database-url "sqlite:///results/market_intelligence.db"
```

### 2. Serve the Dashboard

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli serve-ui \
  --frontend-dir frontend/dist \
  --results-dir results \
  --host 127.0.0.1 \
  --port 8000
```

### 3. Use the Combined Local Runner

Serve the dashboard using the latest available artifacts:

```bash
./run_app.sh
```

Generate a fresh prediction first, then build and serve the UI:

```bash
RUN_PIPELINE=true ./run_app.sh
```

Default local URL:

```text
http://127.0.0.1:8000
```

## CLI Commands

The main CLI entrypoint is exposed through:

```bash
market-prediction
```

Available workflows include:

- `run-daily` run one prediction cycle and write JSON and HTML outputs
- `bootstrap-history` backfill historical market intelligence data into storage
- `build-graph` build a graph workflow for a completed run
- `backfill-graphs` queue graph builds for historical runs
- `scheduler` run the built-in weekday scheduler
- `serve-ui` serve the built frontend and live result endpoints

Example:

```bash
market-prediction run-daily --output-dir results --target "S&P 500"
```

## Graph Workflow

If graph mode is enabled in local configuration, completed runs can be sent through the graph pipeline for extraction, linking, and storage.

Build a graph for a specific run:

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli build-graph \
  --run-id ingestion-a9fcad2d23af \
  --results-dir results \
  --database-url "sqlite:///results/market_intelligence.db"
```

Queue graph builds for historical runs:

```bash
PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli backfill-graphs \
  --results-dir results \
  --database-url "sqlite:///results/market_intelligence.db"
```

## Configuration

Configuration is environment-driven. Common values include:

- `OUTPUT_DIR` output location for generated artifacts when using `run_app.sh`
- `TARGET` market target, defaulting to `S&P 500`
- `PROVIDER` narrative provider, defaulting to `openrouter`
- `PORT` UI server port, defaulting to `8000`
- `APP_HOST` bind host, defaulting to `127.0.0.1`
- `RUN_PIPELINE` whether `run_app.sh` should generate a fresh run before serving
- `DATABASE_URL` SQLite or other supported SQLAlchemy database URL

The Python config layer also supports graph- and provider-specific environment variables. Keep all secrets in local untracked environment files or shell environment variables.

## Outputs

Runtime outputs are written under `results/` by default and are intentionally not tracked in git.

Typical artifacts include:

- `results/YYYY-MM-DD_s_p_500.json`
- `results/YYYY-MM-DD_s_p_500.html`
- `results/runs/`
- `results/_state/`
- `results/logs/`
- local SQLite database files

## Security and Repository Hygiene

- Do not commit `.env` files or credential files
- Keep provider API keys in local environment variables only
- Keep Neo4j credentials in untracked local files such as `neo4j-credentials.local.txt`
- Generated runtime outputs under `results/` and `results_*` are ignored intentionally

## Testing

The repository includes Python tests under `tests/`.

Run tests with:

```bash
python -m pytest
```

If `pytest` is not installed in your environment, install project dependencies in `.venv` first.

## Notes

- The frontend build output in `frontend/dist` is generated locally and should not be hand-edited
- `run_app.sh` automatically loads a local `.env` file if present
- Graph build tasks may continue asynchronously; for short-lived processes, keep the process alive long enough for queued work to complete

## License

Add a project license file if you intend to distribute or open source the repository publicly.
