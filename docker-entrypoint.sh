#!/bin/bash
set -eo pipefail

echo "Starting Market Prediction Container..."

if [[ -f ".env" ]]; then
  set -a
  source ".env"
  set +a
fi

# Ensure results directory and stateful subdirectories exist
mkdir -p "$RESULTS_DIR" "$RESULTS_DIR/runs" "$RESULTS_DIR/_state" "$RESULTS_DIR/logs"

# 1. Start the scheduled background agent
# This agent will wake up daily at $SCHEDULE_TIME in $TIMEZONE to scrape, analyze, and write a new json output to $RESULTS_DIR
echo "Starting background scheduler (Target: '${TARGET:-S&P 500}', Time: ${SCHEDULE_TIME:-08:30} ${TIMEZONE:-America/New_York})..."
python3 -m market_direction_dashboard.cli scheduler \
  --output-dir "$RESULTS_DIR" \
  --target "${TARGET:-S&P 500}" \
  --time "${SCHEDULE_TIME:-08:30}" \
  --timezone "${TIMEZONE:-America/New_York}" &

# Wait for the scheduler process to initialize
sleep 3

# 2. Start the foreground web API / UI server
echo "Starting frontend UI and JSON web server on port ${PORT:-8000}..."
exec python3 -m market_direction_dashboard.cli serve-ui \
  --frontend-dir "$FRONTEND_DIR" \
  --results-dir "$RESULTS_DIR" \
  --host "${APP_HOST:-0.0.0.0}" \
  --port "${PORT:-8000}"
