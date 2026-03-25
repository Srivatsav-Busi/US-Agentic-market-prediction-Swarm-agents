#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/results}"
TARGET="${TARGET:-S&P 500}"
PROVIDER="${PROVIDER:-openrouter}"
PORT="${PORT:-8000}"
APP_HOST="${APP_HOST:-127.0.0.1}"
RUN_PIPELINE="${RUN_PIPELINE:-false}"
DATABASE_URL="${DATABASE_URL:-sqlite:///$OUTPUT_DIR/market_intelligence.db}"

cd "$ROOT_DIR"

# Load local environment defaults for daily runs.
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/runs" "$OUTPUT_DIR/_state" "$OUTPUT_DIR/logs"

if [[ "$RUN_PIPELINE" == "true" ]]; then
  PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli run-daily \
    --output-dir "$OUTPUT_DIR" \
    --target "$TARGET" \
    --provider "$PROVIDER" \
    --persist-db \
    --database-url "$DATABASE_URL"
fi

(cd frontend && npm run build >/dev/null)

PYTHONPATH=src .venv/bin/python -m market_direction_dashboard.cli serve-ui \
  --frontend-dir "$ROOT_DIR/frontend/dist" \
  --results-dir "$OUTPUT_DIR" \
  --host "$APP_HOST" \
  --port "$PORT"
