from __future__ import annotations

import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


def compute_next_run(now: datetime, run_time: str, timezone_name: str) -> datetime:
    zone = ZoneInfo(timezone_name)
    local_now = now.astimezone(zone)
    hour, minute = [int(part) for part in run_time.split(":", 1)]
    candidate = local_now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate <= local_now:
        candidate += timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def run_scheduler(
    job,
    run_time: str,
    timezone_name: str,
    max_runs: int | None = None,
    poll_seconds: int = 30,
    run_now: bool = False,
) -> list[dict]:
    completed: list[dict] = []
    if run_now and (max_runs is None or len(completed) < max_runs):
        completed.append(job())
    while max_runs is None or len(completed) < max_runs:
        now = datetime.now(ZoneInfo(timezone_name))
        next_run = compute_next_run(now, run_time, timezone_name)
        remaining = max((next_run - now).total_seconds(), 0)
        while remaining > 0:
            sleep_chunk = min(poll_seconds, remaining)
            time.sleep(sleep_chunk)
            remaining -= sleep_chunk
        completed.append(job())
    return completed
