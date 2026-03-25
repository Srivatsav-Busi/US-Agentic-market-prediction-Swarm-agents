from __future__ import annotations

from datetime import UTC, datetime

from ..config import list_tracked_instruments
from ..sources import _fetch_fred_history, _fetch_market_history_series
from ..storage.models import InstrumentRecord


def tracked_instruments(config: dict) -> list[InstrumentRecord]:
    records = []
    for instrument in list_tracked_instruments(config):
        records.append(
            InstrumentRecord(
                symbol=instrument["symbol"],
                display_name=instrument["display_name"],
                asset_class=instrument["asset_class"],
                category=instrument["category"],
            )
        )
    return records


def fetch_historical_market_data(config: dict, years: int = 2) -> tuple[dict[str, list[dict]], list[dict]]:
    history_days = max(int(years * 365), 90)
    source_health = {"unused_sources": [], "network_error": False}
    diagnostics: list[dict] = []
    history: dict[str, list[dict]] = {}
    all_symbols = {}
    all_symbols.update(config.get("market_symbols", {}))
    all_symbols.update(config.get("sector_symbols", {}))
    for label, symbol in all_symbols.items():
        rows, attempts = _fetch_market_history_series(label, symbol, history_days, source_health)
        diagnostics.append(
            {
                "label": label,
                "symbol": symbol,
                "row_count": len(rows),
                "providers": [attempt.provider_name for attempt in attempts],
                "statuses": [attempt.status for attempt in attempts],
            }
        )
        if rows:
            history[label] = rows
    return history, diagnostics


def fetch_macro_series_history(config: dict, years: int = 2) -> tuple[dict[str, list[dict]], list[dict]]:
    history_days = max(int(years * 365), 180)
    source_health = {"unused_sources": [], "network_error": False}
    diagnostics: list[dict] = []
    macro_history: dict[str, list[dict]] = {}
    for series_name, meta in config.get("macro_series", {}).items():
        rows, fetch_result = _fetch_fred_history(
            series_name,
            f"fred-{series_name.lower()}-history",
            "macro_history",
            history_days,
            source_health,
        )
        diagnostics.append(
            {
                "series_name": series_name,
                "display_name": meta.get("display_name", series_name),
                "row_count": len(rows),
                "provider": fetch_result.provider_name,
                "status": fetch_result.status,
            }
        )
        if rows:
            macro_history[series_name] = rows
    return macro_history, diagnostics


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()
