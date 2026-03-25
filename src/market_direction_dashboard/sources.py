from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import re
import socket
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib import error, parse, request

from .graph_features import GraphPredictionContext
from .models import DataQualitySummary, FetchResult, SourceItem
from .sector_scraper import fetch_sector_data


TRADING_AGENTS_ROOT = Path("/Users/srivatsavbusi/TradingAgents")
GOOGLE_NEWS_RSS = "https://news.google.com/rss/search"
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


def collect_sources(
    prediction_date: str,
    target: str,
    config: dict,
) -> tuple[list[SourceItem], dict, list[str], dict]:
    run_timestamp = _reference_fetch_timestamp(prediction_date)
    fetch_results: list[FetchResult] = []
    source_health = {
        "network_error": False,
        "quote_failures": [],
        "rss_ok": False,
        "tradingagents_ok": False,
        "proxy_quotes": [],
        "quote_details": [],
        "unused_sources": [],
    }

    snapshot = build_market_snapshot(target, config, prediction_date, source_health, fetch_results)
    snapshot["history"] = build_market_history(config, source_health, fetch_results)
    snapshot["sector_outlook"] = fetch_sector_data() if config.get("enable_sector_scrape", False) else {}
    raw_items = build_market_snapshot_items(snapshot, prediction_date, run_timestamp)

    ta_items, ta_result = _fetch_tradingagents_global_news(prediction_date, config["source_limit"], source_health)
    fetch_results.append(ta_result)
    raw_items.extend(ta_items)

    for category, queries in config["rss_queries"].items():
        fetched, result = _fetch_google_news_items(
            queries=queries,
            category=category,
            prediction_date=prediction_date,
            limit=config["source_limit"],
            source_health=source_health,
        )
        fetch_results.append(result)
        raw_items.extend(fetched)

    raw_items.extend(_ensure_minimum_categories(prediction_date, raw_items, run_timestamp))
    validated_items, quality_summary = validate_and_normalize_items(raw_items, prediction_date, config)
    warnings = _build_source_warnings(source_health, snapshot, validated_items, quality_summary)
    diagnostics = _build_source_diagnostics(source_health, snapshot, validated_items, fetch_results, quality_summary)
    return validated_items, snapshot, warnings, diagnostics


def _reference_fetch_timestamp(prediction_date: str) -> str:
    today = datetime.now(UTC).date().isoformat()
    if prediction_date and prediction_date != today:
        return f"{prediction_date}T12:00:00+00:00"
    return _iso_now()


def build_market_snapshot(
    target: str,
    config: dict,
    prediction_date: str,
    source_health: dict,
    fetch_results: list[FetchResult],
) -> dict:
    snapshot = {
        "target": target,
        "target_column": target,
        "latest_price": None,
        "latest_date": prediction_date,
        "series": {},
    }
    for column, symbol in config.get("market_symbols", {}).items():
        quote, attempts = _fetch_live_market_quote(column, symbol, source_health)
        fetch_results.extend(attempts)
        if not quote:
            source_health["quote_failures"].append((column, symbol))
            continue
        latest_value = quote["latest"]
        previous_value = quote["previous"]
        pct_change = 0.0 if previous_value == 0 else (latest_value / previous_value - 1.0) * 100.0
        snapshot["series"][column] = {
            "latest": latest_value,
            "previous": previous_value,
            "pct_change": pct_change,
            "provider": quote.get("provider", "unknown"),
            "proxy_for": quote.get("proxy_for"),
            "fetched_at": quote.get("fetched_at", _iso_now()),
            "freshness_seconds": quote.get("freshness_seconds"),
            "status": quote.get("status", "ok"),
        }
        source_health["quote_details"].append(
            {
                "label": column,
                "provider": quote.get("provider", "unknown"),
                "proxy_for": quote.get("proxy_for"),
                "freshness_seconds": quote.get("freshness_seconds"),
                "fetched_at": quote.get("fetched_at"),
            }
        )
        if quote.get("proxy_for"):
            source_health["proxy_quotes"].append({"label": column, "proxy_for": quote["proxy_for"]})
        if column == target:
            snapshot["latest_price"] = latest_value
            snapshot["latest_date"] = quote.get("date", prediction_date)
    return snapshot


def build_market_history(
    config: dict,
    source_health: dict,
    fetch_results: list[FetchResult],
) -> dict[str, list[dict[str, str | float | None]]]:
    history_days = int(config.get("history_lookback_days", 180))
    history: dict[str, list[dict[str, str | float | None]]] = {}
    for column, symbol in config.get("market_symbols", {}).items():
        rows, attempts = _fetch_market_history_series(column, symbol, history_days, source_health)
        fetch_results.extend(attempts)
        if rows:
            history[column] = rows
    return history


def build_market_snapshot_items(snapshot: dict, prediction_date: str, fetched_at: str) -> list[SourceItem]:
    items: list[SourceItem] = []
    for column, values in snapshot.get("series", {}).items():
        pct_change = float(values["pct_change"])
        score = _market_impact_score(column, pct_change)
        provider_text = values.get("provider", "live-provider")
        proxy_suffix = f" using proxy {values['proxy_for']}" if values.get("proxy_for") else ""
        title = f"{column} latest move"
        summary = f"{column} changed {pct_change:+.2f}% into the latest observation{proxy_suffix}."
        items.append(
            SourceItem(
                id=_stable_item_id("market", title, provider_text, prediction_date),
                title=title,
                source=f"market-data/{provider_text}",
                source_type="market",
                category="market",
                published_at=f"{prediction_date}T08:30:00-05:00",
                fetched_at=fetched_at,
                url="",
                summary=summary,
                raw_text=summary,
                impact=_impact_label(score),
                impact_score=score,
                instrument=column,
                direction=_impact_label(score),
                confidence_hint=min(1.0, 0.45 + abs(score) * 0.45),
                freshness_score=_freshness_score(values.get("freshness_seconds"), 48),
                credibility_score=_credibility_score(provider_text, "market"),
                proxy_used=bool(values.get("proxy_for")),
                quality_score=0.0,
                evidence_kind="proxy" if values.get("proxy_for") else "direct",
            )
        )
    return items


def validate_and_normalize_items(items: list[SourceItem], prediction_date: str, config: dict) -> tuple[list[SourceItem], DataQualitySummary]:
    freshness_hours = config.get(
        "freshness_threshold_hours",
        {"market": 48, "macro": 72, "news": 18, "events": 72, "sentiment": 18},
    )
    valid_items: list[SourceItem] = []
    rejected = 0
    duplicate_count = 0
    stale_count = 0
    malformed_count = 0
    duplicate_clusters: dict[str, str] = {}
    seen_clusters: set[str] = set()

    for index, item in enumerate(sorted(items, key=lambda value: value.published_at, reverse=True)):
        item.id = item.id or _stable_item_id(item.category, item.title, item.source, str(index))
        item.fetched_at = item.fetched_at or f"{prediction_date}T08:30:00-05:00"
        item.raw_text = item.raw_text or f"{item.title}. {item.summary}".strip()
        item.source_type = item.source_type or ("market" if item.category == "market" else "news")
        item.direction = item.direction or _impact_label(item.impact_score)
        item.credibility_score = item.credibility_score or _credibility_score(item.source, item.source_type)

        published_at = _safe_parse_dt(item.published_at)
        fetched_at = _safe_parse_dt(item.fetched_at)
        if not published_at or not fetched_at:
            malformed_count += 1
            rejected += 1
            continue

        age_hours = max(0.0, (fetched_at - published_at).total_seconds() / 3600.0)
        threshold = freshness_hours.get(item.source_type, 24)
        if age_hours > threshold and item.source != "local-fallback":
            stale_count += 1
            rejected += 1
            continue

        cluster = _duplicate_cluster(item.title, item.summary)
        item.duplicate_cluster = cluster
        if cluster in seen_clusters:
            duplicate_count += 1
            rejected += 1
            duplicate_clusters[item.id] = cluster
            continue
        seen_clusters.add(cluster)

        item.freshness_score = item.freshness_score or _freshness_score(int(age_hours * 3600), threshold)
        item.base_quality_score = round(
            max(
                0.0,
                min(
                    1.0,
                    item.freshness_score * 0.35
                    + item.credibility_score * 0.35
                    + (1.0 - (0.2 if item.proxy_used else 0.0)) * 0.2
                    + item.confidence_hint * 0.1,
                ),
            ),
            3,
        )
        item.quality_score = item.base_quality_score
        item.data_quality_flags = _quality_flags(item)
        valid_items.append(item)

    providers = {item.source for item in valid_items if item.source != "local-fallback"}
    average_quality = sum(item.quality_score for item in valid_items) / len(valid_items) if valid_items else 0.0
    gate_failures = _evaluate_quality_gates(valid_items)
    quality_summary = DataQualitySummary(
        valid_item_count=len(valid_items),
        rejected_item_count=rejected,
        duplicate_item_count=duplicate_count,
        stale_item_count=stale_count,
        malformed_item_count=malformed_count,
        proxy_item_count=sum(1 for item in valid_items if item.proxy_used),
        distinct_provider_count=len(providers),
        average_quality_score=round(average_quality, 3),
        gate_failures=gate_failures,
    )
    return valid_items, quality_summary


def apply_graph_quality_layer(
    items: list[SourceItem],
    quality_summary: DataQualitySummary,
    source_diagnostics: dict,
    graph_prediction_context: GraphPredictionContext | None,
) -> tuple[list[SourceItem], DataQualitySummary, dict]:
    if graph_prediction_context is None:
        return items, quality_summary, source_diagnostics
    adjustments = graph_prediction_context.quality_adjustments
    if not items:
        updated = DataQualitySummary(
            **{
                **quality_summary.to_dict(),
                "graph_quality_summary": adjustments.to_dict(),
                "cluster_quality_summary": list(adjustments.cluster_summaries),
            }
        )
        diagnostics = dict(source_diagnostics)
        diagnostics["data_quality_summary"] = updated.to_dict()
        return items, updated, diagnostics

    adjusted_item_count = 0
    for item in items:
        item.base_quality_score = float(item.base_quality_score or item.quality_score or 0.0)
        quality_payload = adjustments.item_adjustments.get(item.id, {})
        quality_multiplier = float(quality_payload.get("quality_multiplier", 1.0))
        quality_delta = float(quality_payload.get("adjustment", 0.0))
        updated_quality = max(0.0, min(1.0, item.base_quality_score * quality_multiplier))
        if abs(updated_quality - item.quality_score) >= 0.01:
            adjusted_item_count += 1
        item.graph_quality_adjustment = round(updated_quality - item.base_quality_score, 4)
        item.graph_quality_reasons = list(quality_payload.get("reasons", []))
        item.quality_score = round(updated_quality, 3)
        if quality_delta < -0.03:
            item.data_quality_flags = sorted(set(item.data_quality_flags + list(item.graph_quality_reasons)))
        elif quality_payload.get("reasons"):
            item.data_quality_flags = sorted(set(item.data_quality_flags + [reason for reason in item.graph_quality_reasons if reason != "independent_corroboration"]))

    average_quality = sum(item.quality_score for item in items) / len(items)
    gate_failures = list(quality_summary.gate_failures)
    if adjustments.severe_graph_risk and "graph_quality_risk" not in gate_failures:
        gate_failures.append("graph_quality_risk")
    if adjustments.source_monoculture_penalty >= 0.14 and "graph_source_monoculture" not in gate_failures:
        gate_failures.append("graph_source_monoculture")
    if adjustments.contradiction_penalty >= 0.12 and "graph_contradiction_cluster" not in gate_failures:
        gate_failures.append("graph_contradiction_cluster")

    updated_summary = DataQualitySummary(
        valid_item_count=quality_summary.valid_item_count,
        rejected_item_count=quality_summary.rejected_item_count,
        duplicate_item_count=quality_summary.duplicate_item_count,
        stale_item_count=quality_summary.stale_item_count,
        malformed_item_count=quality_summary.malformed_item_count,
        proxy_item_count=quality_summary.proxy_item_count,
        distinct_provider_count=quality_summary.distinct_provider_count,
        average_quality_score=round(average_quality, 3),
        graph_adjusted_item_count=adjusted_item_count,
        graph_quality_summary=adjustments.to_dict(),
        cluster_quality_summary=list(adjustments.cluster_summaries),
        gate_failures=gate_failures,
    )
    diagnostics = dict(source_diagnostics)
    diagnostics["data_quality_summary"] = updated_summary.to_dict()
    return items, updated_summary, diagnostics


def _fetch_live_market_quote(label: str, symbol: str, source_health: dict) -> tuple[dict | None, list[FetchResult]]:
    if label == "VIX":
        return _first_quote(
            source_health,
            "market",
            [
                lambda: _fetch_yahoo_chart_quote(symbol, source_health),
                lambda: _fetch_fred_series("VIXCLS", "fred-vix", "market", source_health),
            ],
        )
    if label == "US 10 YR TREASURY":
        return _first_quote(
            source_health,
            "macro",
            [
                lambda: _fetch_yahoo_chart_quote(symbol, source_health),
                lambda: _fetch_fred_series("DGS10", "fred-dgs10", "macro", source_health),
            ],
        )
    if label == "DXY":
        return _first_quote(
            source_health,
            "market",
            [
                lambda: _fetch_yahoo_chart_quote(symbol, source_health),
                lambda: _fetch_stooq_quote("usdidx", "stooq-usdidx", "market", source_health),
                lambda: _fetch_yahoo_chart_quote("UUP", source_health, proxy_for="UUP ETF"),
            ],
        )
    if label == "WTI CRUDE OIL":
        return _first_quote(
            source_health,
            "market",
            [
                lambda: _fetch_yahoo_chart_quote(symbol, source_health),
                lambda: _fetch_stooq_quote("cl.f", "stooq-cl.f", "market", source_health),
                lambda: _fetch_yahoo_chart_quote("USO", source_health, proxy_for="USO ETF"),
            ],
        )
    if label == "GOLD":
        return _first_quote(
            source_health,
            "market",
            [
                lambda: _fetch_yahoo_chart_quote(symbol, source_health),
                lambda: _fetch_stooq_quote("gold", "stooq-gold", "market", source_health),
                lambda: _fetch_yahoo_chart_quote("GLD", source_health, proxy_for="GLD ETF"),
            ],
        )
    if label == "RUSSELL 2000":
        return _first_quote(
            source_health,
            "market",
            [
                lambda: _fetch_yahoo_chart_quote(symbol, source_health),
                lambda: _fetch_stooq_quote("^rut", "stooq-rut", "market", source_health),
                lambda: _fetch_yahoo_chart_quote("IWM", source_health, proxy_for="IWM ETF"),
            ],
        )
    return _first_quote(
        source_health,
        "market",
        [
            lambda: _fetch_yahoo_chart_quote(symbol, source_health),
            lambda: _fetch_stooq_quote(_default_stooq_symbol(symbol), f"stooq-{_default_stooq_symbol(symbol)}", "market", source_health)
            if _default_stooq_symbol(symbol)
            else None,
        ],
    )


def _fetch_market_history_series(
    label: str,
    symbol: str,
    history_days: int,
    source_health: dict,
) -> tuple[list[dict[str, str | float | None]], list[FetchResult]]:
    if label == "VIX":
        return _first_history(
            source_health,
            "market_history",
            [
                lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
                lambda: _fetch_fred_history("VIXCLS", "fred-vix-history", "market_history", history_days, source_health),
            ],
        )
    if label == "US 10 YR TREASURY":
        return _first_history(
            source_health,
            "macro_history",
            [
                lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
                lambda: _fetch_fred_history("DGS10", "fred-dgs10-history", "macro_history", history_days, source_health),
            ],
        )
    if label == "DXY":
        return _first_history(
            source_health,
            "market_history",
            [
                lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
                lambda: _fetch_stooq_history("usdidx", "stooq-usdidx-history", "market_history", history_days, source_health),
                lambda: _fetch_yahoo_chart_history("UUP", history_days, source_health, proxy_for="UUP ETF"),
            ],
        )
    if label == "WTI CRUDE OIL":
        return _first_history(
            source_health,
            "market_history",
            [
                lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
                lambda: _fetch_stooq_history("cl.f", "stooq-cl.f-history", "market_history", history_days, source_health),
                lambda: _fetch_yahoo_chart_history("USO", history_days, source_health, proxy_for="USO ETF"),
            ],
        )
    if label == "GOLD":
        return _first_history(
            source_health,
            "market_history",
            [
                lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
                lambda: _fetch_stooq_history("gold", "stooq-gold-history", "market_history", history_days, source_health),
                lambda: _fetch_yahoo_chart_history("GLD", history_days, source_health, proxy_for="GLD ETF"),
            ],
        )
    if label == "RUSSELL 2000":
        return _first_history(
            source_health,
            "market_history",
            [
                lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
                lambda: _fetch_stooq_history("^rut", "stooq-rut-history", "market_history", history_days, source_health),
                lambda: _fetch_yahoo_chart_history("IWM", history_days, source_health, proxy_for="IWM ETF"),
            ],
        )
    return _first_history(
        source_health,
        "market_history",
        [
            lambda: _fetch_yahoo_chart_history(symbol, history_days, source_health),
            lambda: _fetch_stooq_history(_default_stooq_symbol(symbol), f"stooq-{_default_stooq_symbol(symbol)}-history", "market_history", history_days, source_health)
            if _default_stooq_symbol(symbol)
            else None,
        ],
    )


def _first_history(source_health: dict, fetch_group: str, fetchers: list) -> tuple[list[dict[str, str | float | None]], list[FetchResult]]:
    attempts: list[FetchResult] = []
    for fetcher in fetchers:
        response = fetcher()
        if not response:
            continue
        rows, fetch_result = response
        attempts.append(fetch_result)
        if rows:
            return rows, attempts
    source_health["unused_sources"].append(f"{fetch_group}:no-successful-provider")
    return [], attempts


def _first_quote(source_health: dict, fetch_group: str, fetchers: list) -> tuple[dict | None, list[FetchResult]]:
    attempts: list[FetchResult] = []
    for fetcher in fetchers:
        response = fetcher()
        if not response:
            continue
        quote, fetch_result = response
        attempts.append(fetch_result)
        if quote:
            return quote, attempts
    source_health["unused_sources"].append(f"{fetch_group}:no-successful-provider")
    return None, attempts


def _fetch_yahoo_chart_quote(symbol: str, source_health: dict, proxy_for: str | None = None) -> tuple[dict | None, FetchResult]:
    encoded_symbol = parse.quote(symbol, safe="")
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range=5d&interval=1d",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range=5d&interval=1d",
    ]
    for url in urls:
        payload, latency_ms = _get_json(url, source_health)
        fetch_result = FetchResult(
            fetch_group="market",
            provider_name="yahoo-chart",
            payload=None,
            fetch_timestamp=_iso_now(),
            latency_ms=latency_ms,
            status="ok" if payload else "unavailable",
            fallback_used=bool(proxy_for),
            proxy_for=proxy_for,
        )
        if not payload:
            continue
        try:
            result = payload["chart"]["result"][0]
            closes = [value for value in result["indicators"]["quote"][0]["close"] if value is not None]
            timestamps = result.get("timestamp", [])
            if len(closes) < 2:
                continue
            latest_date = prediction_date_from_timestamp(timestamps[-1]) if timestamps else None
            freshness_seconds = _age_seconds(latest_date)
            quote = {
                "latest": float(closes[-1]),
                "previous": float(closes[-2]),
                "date": latest_date,
                "provider": "yahoo-chart",
                "proxy_for": proxy_for,
                "status": "ok",
                "fetched_at": fetch_result.fetch_timestamp,
                "freshness_seconds": freshness_seconds,
            }
            fetch_result.payload = {"symbol": symbol, "points": len(closes)}
            fetch_result.item_count = len(closes)
            fetch_result.freshness_seconds = freshness_seconds
            return quote, fetch_result
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return None, FetchResult(
        fetch_group="market",
        provider_name="yahoo-chart",
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=0,
        status="error",
        fallback_used=bool(proxy_for),
        proxy_for=proxy_for,
        warning=f"Unable to parse Yahoo data for {symbol}",
    )


def _fetch_yahoo_chart_history(
    symbol: str,
    history_days: int,
    source_health: dict,
    proxy_for: str | None = None,
) -> tuple[list[dict[str, str | float | None]], FetchResult]:
    encoded_symbol = parse.quote(symbol, safe="")
    range_key = "5y" if history_days > 730 else "2y" if history_days > 365 else "1y" if history_days > 180 else "6mo" if history_days > 90 else "3mo"
    urls = [
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range={range_key}&interval=1d",
        f"https://query2.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range={range_key}&interval=1d",
    ]
    for url in urls:
        payload, latency_ms = _get_json(url, source_health)
        fetch_result = FetchResult(
            fetch_group="market_history",
            provider_name="yahoo-chart-history",
            payload=None,
            fetch_timestamp=_iso_now(),
            latency_ms=latency_ms,
            status="ok" if payload else "unavailable",
            fallback_used=bool(proxy_for),
            proxy_for=proxy_for,
        )
        if not payload:
            continue
        try:
            result = payload["chart"]["result"][0]
            timestamps = result.get("timestamp", [])
            closes = result["indicators"]["quote"][0]["close"]
            rows = []
            for timestamp, close in zip(timestamps, closes, strict=False):
                if close is None:
                    continue
                rows.append(
                    {
                        "date": prediction_date_from_timestamp(timestamp),
                        "value": float(close),
                        "provider": "yahoo-chart-history",
                        "proxy_for": proxy_for,
                    }
                )
            rows = rows[-history_days:]
            fetch_result.payload = {"symbol": symbol, "points": len(rows)}
            fetch_result.item_count = len(rows)
            fetch_result.freshness_seconds = _age_seconds(rows[-1]["date"]) if rows else None
            if rows:
                return rows, fetch_result
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return [], FetchResult(
        fetch_group="market_history",
        provider_name="yahoo-chart-history",
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=0,
        status="error",
        fallback_used=bool(proxy_for),
        proxy_for=proxy_for,
        warning=f"Unable to parse Yahoo history for {symbol}",
    )


def _fetch_stooq_quote(symbol: str | None, provider: str, fetch_group: str, source_health: dict) -> tuple[dict | None, FetchResult] | None:
    if not symbol:
        return None
    url = f"https://stooq.com/q/d/l/?s={parse.quote(symbol, safe='')}&i=d"
    text, latency_ms = _get_text(url, source_health)
    fetch_result = FetchResult(
        fetch_group=fetch_group,
        provider_name=provider,
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=latency_ms,
        status="ok" if text else "unavailable",
    )
    if not text or "Date,Open,High,Low,Close,Volume" not in text:
        return None, fetch_result
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) < 3:
        return None, fetch_result
    try:
        previous = rows[-2].split(",")
        latest = rows[-1].split(",")
        freshness_seconds = _age_seconds(latest[0])
        fetch_result.item_count = len(rows) - 1
        fetch_result.freshness_seconds = freshness_seconds
        fetch_result.payload = {"symbol": symbol, "rows": len(rows) - 1}
        return {
            "latest": float(latest[4]),
            "previous": float(previous[4]),
            "date": latest[0],
            "provider": provider,
            "proxy_for": None,
            "status": "ok",
            "fetched_at": fetch_result.fetch_timestamp,
            "freshness_seconds": freshness_seconds,
        }, fetch_result
    except (IndexError, ValueError):
        fetch_result.status = "error"
        return None, fetch_result


def _fetch_stooq_history(
    symbol: str | None,
    provider: str,
    fetch_group: str,
    history_days: int,
    source_health: dict,
) -> tuple[list[dict[str, str | float | None]], FetchResult] | None:
    if not symbol:
        return None
    url = f"https://stooq.com/q/d/l/?s={parse.quote(symbol, safe='')}&i=d"
    text, latency_ms = _get_text(url, source_health)
    fetch_result = FetchResult(
        fetch_group=fetch_group,
        provider_name=provider,
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=latency_ms,
        status="ok" if text else "unavailable",
    )
    if not text or "Date,Open,High,Low,Close,Volume" not in text:
        return [], fetch_result
    rows = [line.strip() for line in text.splitlines() if line.strip()]
    if len(rows) < 3:
        return [], fetch_result
    history = []
    try:
        for row in rows[1:]:
            parts = row.split(",")
            history.append(
                {
                    "date": parts[0],
                    "value": float(parts[4]),
                    "provider": provider,
                    "proxy_for": None,
                }
            )
        history = history[-history_days:]
        fetch_result.payload = {"symbol": symbol, "rows": len(history)}
        fetch_result.item_count = len(history)
        fetch_result.freshness_seconds = _age_seconds(history[-1]["date"]) if history else None
        return history, fetch_result
    except (IndexError, ValueError):
        fetch_result.status = "error"
        return [], fetch_result


def _fetch_fred_series(series_id: str, provider: str, fetch_group: str, source_health: dict) -> tuple[dict | None, FetchResult]:
    text, latency_ms = _get_text(f"{FRED_CSV}{series_id}", source_health)
    fetch_result = FetchResult(
        fetch_group=fetch_group,
        provider_name=provider,
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=latency_ms,
        status="ok" if text else "unavailable",
    )
    if not text:
        return None, fetch_result
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = [row for row in reader if row.get(series_id) not in ("", ".", None)]
        if len(rows) < 2:
            return None, fetch_result
        previous = rows[-2]
        latest = rows[-1]
        freshness_seconds = _age_seconds(latest["DATE"])
        fetch_result.item_count = len(rows)
        fetch_result.freshness_seconds = freshness_seconds
        fetch_result.payload = {"series_id": series_id, "rows": len(rows)}
        return {
            "latest": float(latest[series_id]),
            "previous": float(previous[series_id]),
            "date": latest["DATE"],
            "provider": provider,
            "proxy_for": None,
            "status": "ok",
            "fetched_at": fetch_result.fetch_timestamp,
            "freshness_seconds": freshness_seconds,
        }, fetch_result
    except (KeyError, ValueError):
        fetch_result.status = "error"
        return None, fetch_result


def _fetch_fred_history(
    series_id: str,
    provider: str,
    fetch_group: str,
    history_days: int,
    source_health: dict,
) -> tuple[list[dict[str, str | float | None]], FetchResult]:
    text, latency_ms = _get_text(f"{FRED_CSV}{series_id}", source_health)
    fetch_result = FetchResult(
        fetch_group=fetch_group,
        provider_name=provider,
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=latency_ms,
        status="ok" if text else "unavailable",
    )
    if not text:
        return [], fetch_result
    try:
        reader = csv.DictReader(io.StringIO(text))
        rows = [row for row in reader if row.get(series_id) not in ("", ".", None)]
        history = [
            {
                "date": row["DATE"],
                "value": float(row[series_id]),
                "provider": provider,
                "proxy_for": None,
            }
            for row in rows[-history_days:]
        ]
        fetch_result.payload = {"series_id": series_id, "rows": len(history)}
        fetch_result.item_count = len(history)
        fetch_result.freshness_seconds = _age_seconds(history[-1]["date"]) if history else None
        return history, fetch_result
    except (KeyError, ValueError):
        fetch_result.status = "error"
        return [], fetch_result


def _default_stooq_symbol(symbol: str) -> str | None:
    return {
        "^GSPC": "^spx",
        "^NDX": "^ndx",
        "^DJI": "^dji",
        "^RUT": "^rut",
    }.get(symbol)


def prediction_date_from_timestamp(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, tz=UTC).date().isoformat()


def _fetch_tradingagents_global_news(prediction_date: str, limit: int, source_health: dict) -> tuple[list[SourceItem], FetchResult]:
    module_path = TRADING_AGENTS_ROOT / "tradingagents" / "dataflows" / "yfinance_news.py"
    fetch_result = FetchResult(
        fetch_group="news",
        provider_name="TradingAgents/yfinance",
        payload=None,
        fetch_timestamp=_iso_now(),
        latency_ms=0,
        status="unavailable",
    )
    if not module_path.exists():
        fetch_result.warning = "TradingAgents adapter not installed"
        return [], fetch_result

    spec = importlib.util.spec_from_file_location("ta_yfinance_news", module_path)
    if not spec or not spec.loader:
        fetch_result.warning = "TradingAgents adapter could not be loaded"
        return [], fetch_result

    module = importlib.util.module_from_spec(spec)
    started = time.perf_counter()
    try:
        spec.loader.exec_module(module)
        raw = module.get_global_news_yfinance(prediction_date, look_back_days=2, limit=limit)
        source_health["tradingagents_ok"] = True
    except ModuleNotFoundError:
        fetch_result.warning = "TradingAgents dependencies unavailable"
        return [], fetch_result
    except Exception as exc:
        if _is_network_error(exc):
            source_health["network_error"] = True
        fetch_result.warning = str(exc)
        return [], fetch_result

    fetch_result.latency_ms = int((time.perf_counter() - started) * 1000)
    items: list[SourceItem] = []
    blocks = [block.strip() for block in raw.split("### ") if block.strip()]
    for block in blocks[:limit]:
        lines = block.splitlines()
        header = lines[0]
        match = re.match(r"(.+?) \(source: (.+)\)", header)
        title = match.group(1) if match else header
        source = match.group(2) if match else "TradingAgents/yfinance"
        url = ""
        summary_parts: list[str] = []
        for line in lines[1:]:
            if line.startswith("Link: "):
                url = line.replace("Link: ", "", 1).strip()
            elif line.strip():
                summary_parts.append(line.strip())
        summary = " ".join(summary_parts).strip()
        score = _infer_news_score(title, summary)
        items.append(
            SourceItem(
                id=_stable_item_id("economic", title, source, prediction_date),
                title=title,
                source=source,
                source_type="news",
                category="economic",
                published_at=f"{prediction_date}T08:30:00-05:00",
                fetched_at=fetch_result.fetch_timestamp,
                url=url,
                summary=summary,
                raw_text=f"{title}. {summary}".strip(),
                impact=_impact_label(score),
                impact_score=score,
                direction=_impact_label(score),
                confidence_hint=min(1.0, 0.4 + abs(score) * 0.4),
                freshness_score=0.8,
                credibility_score=0.72,
                evidence_kind="inferred",
            )
        )
    fetch_result.status = "ok" if items else "empty"
    fetch_result.item_count = len(items)
    fetch_result.payload = {"items": len(items)}
    fetch_result.freshness_seconds = _age_seconds(prediction_date)
    return items, fetch_result


def _fetch_google_news_items(
    queries: list[str],
    category: str,
    prediction_date: str,
    limit: int,
    source_health: dict,
) -> tuple[list[SourceItem], FetchResult]:
    collected: list[SourceItem] = []
    seen: set[str] = set()
    started = time.perf_counter()
    for query in queries:
        url = f"{GOOGLE_NEWS_RSS}?{parse.urlencode({'q': query, 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'})}"
        payload, _latency_ms = _get_bytes(url, source_health)
        if not payload:
            continue
        try:
            root = ET.fromstring(payload)
        except ET.ParseError:
            continue

        for item in root.findall(".//item"):
            title = _xml_text(item, "title")
            if not title or title in seen:
                continue
            seen.add(title)
            description = re.sub(r"<[^>]+>", " ", _xml_text(item, "description"))
            published_at = _parse_rss_date(_xml_text(item, "pubDate")) or f"{prediction_date}T08:30:00-05:00"
            score = _infer_news_score(title, description)
            provider = "Google News RSS"
            collected.append(
                SourceItem(
                    id=_stable_item_id(category, title, provider, published_at),
                    title=title,
                    source=provider,
                    source_type="news",
                    category=category,
                    published_at=published_at,
                    fetched_at=_iso_now(),
                    url=_xml_text(item, "link"),
                    summary=" ".join(description.split()),
                    raw_text=f"{title}. {' '.join(description.split())}".strip(),
                    impact=_impact_label(score),
                    impact_score=score,
                    direction=_impact_label(score),
                    confidence_hint=min(1.0, 0.35 + abs(score) * 0.5),
                    freshness_score=_freshness_score(_age_seconds(published_at), 18),
                    credibility_score=_credibility_score(provider, "news"),
                    evidence_kind="direct",
                )
            )
            if len(collected) >= limit:
                break
        if len(collected) >= limit:
            break

    if collected:
        source_health["rss_ok"] = True
    return collected, FetchResult(
        fetch_group="news" if category != "social" else "sentiment",
        provider_name=f"rss-{category}",
        payload={"queries": len(queries), "items": len(collected)},
        fetch_timestamp=_iso_now(),
        latency_ms=int((time.perf_counter() - started) * 1000),
        status="ok" if collected else "empty",
        freshness_seconds=min((_age_seconds(item.published_at) for item in collected), default=None),
        item_count=len(collected),
    )


def _ensure_minimum_categories(prediction_date: str, items: list[SourceItem], fetched_at: str) -> list[SourceItem]:
    present = {item.category for item in items}
    fillers: list[SourceItem] = []
    for category, title in (
        ("economic", "No live economic feed available"),
        ("political", "No live political feed available"),
        ("social", "No live sentiment feed available"),
    ):
        if category not in present:
            fillers.append(
                SourceItem(
                    id=_stable_item_id(category, title, "local-fallback", prediction_date),
                    title=title,
                    source="local-fallback",
                    source_type="news",
                    category=category,
                    published_at=f"{prediction_date}T08:30:00-05:00",
                    fetched_at=fetched_at,
                    url="",
                    summary=f"{category.title()} inputs were unavailable for this run, so this category remains neutral.",
                    raw_text=title,
                    impact="neutral",
                    impact_score=0.0,
                    confidence_hint=0.2,
                    freshness_score=1.0,
                    credibility_score=0.3,
                    quality_score=0.4,
                    evidence_kind="proxy",
                )
            )
    return fillers


def _build_source_warnings(source_health: dict, snapshot: dict, items: list[SourceItem], quality_summary: DataQualitySummary) -> list[str]:
    warnings: list[str] = []
    if not snapshot.get("series"):
        if source_health.get("network_error"):
            warnings.append("Live market snapshot is unavailable because the app could not reach external data providers.")
        elif source_health.get("quote_failures"):
            warnings.append("Live market snapshot is incomplete because quote providers returned no usable data.")
    elif source_health.get("proxy_quotes"):
        warnings.append("Some market indicators were filled using live proxy instruments. Check Data Availability before acting.")

    non_fallback_items = [item for item in items if item.source != "local-fallback"]
    if not non_fallback_items:
        if source_health.get("network_error"):
            warnings.append("Live news sources are unavailable because the app could not reach external providers.")
        else:
            warnings.append("Live news sources returned no usable articles for this run.")
    elif not source_health.get("tradingagents_ok") and source_health.get("rss_ok"):
        warnings.append("TradingAgents news adapter was unavailable, so the app used direct RSS sources instead.")

    if quality_summary.duplicate_item_count:
        warnings.append(f"{quality_summary.duplicate_item_count} duplicate narrative items were removed before reasoning.")
    if quality_summary.stale_item_count:
        warnings.append(f"{quality_summary.stale_item_count} stale items were rejected during validation.")
    if quality_summary.gate_failures:
        warnings.append("Trust gates detected insufficient live evidence for strong directional conviction.")

    return warnings


def _build_source_diagnostics(
    source_health: dict,
    snapshot: dict,
    items: list[SourceItem],
    fetch_results: list[FetchResult],
    quality_summary: DataQualitySummary,
) -> dict:
    available_quotes = sorted(snapshot.get("series", {}).keys())
    available_history = sorted(snapshot.get("history", {}).keys())
    failed_quotes = [
        {"label": label, "symbol": symbol}
        for label, symbol in source_health.get("quote_failures", [])
        if label not in available_quotes
    ]
    non_fallback_items = [item for item in items if item.source != "local-fallback"]
    used_proxies = [entry["label"] for entry in source_health.get("proxy_quotes", [])]
    return {
        "network_error": bool(source_health.get("network_error")),
        "quote_provider_status": "ok" if available_quotes else "unavailable",
        "news_provider_status": "ok" if non_fallback_items else "unavailable",
        "tradingagents_adapter": "ok" if source_health.get("tradingagents_ok") else "fallback",
        "rss_provider": "ok" if source_health.get("rss_ok") else "unavailable",
        "available_quote_labels": available_quotes,
        "available_history_labels": available_history,
        "failed_quotes": failed_quotes,
        "live_source_count": len(non_fallback_items),
        "fallback_source_count": len(items) - len(non_fallback_items),
        "proxy_quotes": source_health.get("proxy_quotes", []),
        "quote_details": source_health.get("quote_details", []),
        "fetch_results": [result.to_dict() for result in fetch_results],
        "data_quality_summary": quality_summary.to_dict(),
        "used_proxies": used_proxies,
        "unused_sources": source_health.get("unused_sources", []),
    }


def _xml_text(node: ET.Element, tag: str) -> str:
    child = node.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def _parse_rss_date(value: str) -> str | None:
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except (TypeError, ValueError, OverflowError):
        return None


def _market_impact_score(column: str, pct_change: float) -> float:
    lowered = column.lower()
    if "vix" in lowered:
        return max(min(-pct_change / 2.0, 1.0), -1.0)
    if "10 yr" in lowered or "10y" in lowered or "treasury" in lowered or "yield" in lowered:
        return max(min(-pct_change / 3.0, 1.0), -1.0)
    if "gold" in lowered:
        return max(min((-pct_change / 6.0) if pct_change > 0 else (abs(pct_change) / 8.0), 1.0), -1.0)
    if "oil" in lowered or "wti" in lowered:
        return max(min(-pct_change / 5.0, 1.0), -1.0)
    if "dxy" in lowered:
        return max(min(-pct_change / 2.5, 1.0), -1.0)
    return max(min(pct_change / 2.5, 1.0), -1.0)


def _infer_news_score(title: str, summary: str) -> float:
    text = f"{title} {summary}".lower()
    positive = {
        "cooling inflation": 0.8,
        "rate cut": 0.9,
        "soft landing": 0.8,
        "beat estimates": 0.7,
        "growth": 0.4,
        "rally": 0.6,
        "optimism": 0.4,
        "stimulus": 0.6,
    }
    negative = {
        "tariff": -0.8,
        "war": -0.9,
        "recession": -1.0,
        "inflation hotter": -0.9,
        "rate hike": -0.9,
        "selloff": -0.8,
        "volatility": -0.5,
        "shutdown": -0.7,
        "layoffs": -0.4,
    }
    score = 0.0
    for phrase, value in positive.items():
        if phrase in text:
            score += value
    for phrase, value in negative.items():
        if phrase in text:
            score += value
    if score == 0.0:
        if any(word in text for word in ("upbeat", "gains", "surge", "advance")):
            score = 0.35
        elif any(word in text for word in ("drop", "fall", "slump", "concern")):
            score = -0.35
    return max(min(score, 1.0), -1.0)


def _impact_label(score: float) -> str:
    if score > 0.2:
        return "bullish"
    if score < -0.2:
        return "bearish"
    return "neutral"


def _is_network_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.gaierror)):
        return True
    if isinstance(exc, error.URLError):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (TimeoutError, socket.gaierror))


def _default_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
    }


def _get_bytes(url: str, source_health: dict) -> tuple[bytes | None, int]:
    req = request.Request(url, headers=_default_headers())
    started = time.perf_counter()
    try:
        with request.urlopen(req, timeout=12) as response:
            return response.read(), int((time.perf_counter() - started) * 1000)
    except Exception as exc:
        if _is_network_error(exc):
            source_health["network_error"] = True
        return None, int((time.perf_counter() - started) * 1000)


def _get_text(url: str, source_health: dict) -> tuple[str | None, int]:
    payload, latency_ms = _get_bytes(url, source_health)
    if not payload:
        return None, latency_ms
    try:
        return payload.decode("utf-8"), latency_ms
    except UnicodeDecodeError:
        return payload.decode("latin-1", errors="ignore"), latency_ms


def _get_json(url: str, source_health: dict) -> tuple[dict | None, int]:
    text, latency_ms = _get_text(url, source_health)
    if not text:
        return None, latency_ms
    try:
        return json.loads(text), latency_ms
    except json.JSONDecodeError:
        return None, latency_ms


def _freshness_score(age_seconds: int | None, threshold_hours: int) -> float:
    if age_seconds is None:
        return 0.4
    threshold_seconds = max(1, threshold_hours * 3600)
    ratio = min(1.0, max(0.0, age_seconds / threshold_seconds))
    return round(max(0.0, 1.0 - ratio), 3)


def _credibility_score(source: str, source_type: str) -> float:
    source_lower = source.lower()
    if "fred" in source_lower:
        return 0.95
    if "yahoo" in source_lower or "stooq" in source_lower:
        return 0.82
    if "google news" in source_lower:
        return 0.65
    if "tradingagents" in source_lower:
        return 0.7
    if source_type == "market":
        return 0.75
    return 0.5


def _stable_item_id(category: str, title: str, source: str, salt: str) -> str:
    digest = hashlib.sha1(f"{category}|{title}|{source}|{salt}".encode("utf-8")).hexdigest()
    return digest[:12]


def _duplicate_cluster(title: str, summary: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", " ", f"{title} {summary}".lower()).strip()
    tokens = [token for token in compact.split() if len(token) > 3]
    canonical = " ".join(tokens[:8]) or compact
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


def _safe_parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _quality_flags(item: SourceItem) -> list[str]:
    flags: list[str] = []
    if item.proxy_used:
        flags.append("proxy_used")
    if item.evidence_kind != "direct":
        flags.append(f"evidence_{item.evidence_kind}")
    if item.freshness_score < 0.35:
        flags.append("aging")
    if item.credibility_score < 0.5:
        flags.append("low_credibility")
    return flags


def _evaluate_quality_gates(items: list[SourceItem]) -> list[str]:
    gate_failures: list[str] = []
    market_items = [item for item in items if item.category == "market"]
    fresh_news = [item for item in items if item.source_type == "news" and item.freshness_score >= 0.35 and item.source != "local-fallback"]
    providers = {item.source for item in items if item.source != "local-fallback"}
    proxy_ratio = (sum(1 for item in items if item.proxy_used) / len(items)) if items else 1.0
    stale_ratio = (sum(1 for item in items if item.freshness_score < 0.35) / len(items)) if items else 1.0
    if len(market_items) < 4:
        gate_failures.append("minimum_market_instruments")
    if len(fresh_news) < 2:
        gate_failures.append("minimum_fresh_news")
    if len(providers) < 2:
        gate_failures.append("minimum_distinct_providers")
    if proxy_ratio > 0.4:
        gate_failures.append("maximum_proxy_ratio")
    if stale_ratio > 0.35:
        gate_failures.append("maximum_stale_ratio")
    return gate_failures


def _age_seconds(value: str | None) -> int | None:
    dt = _safe_parse_dt(value)
    if not dt:
        return None
    return max(0, int((datetime.now(UTC) - dt).total_seconds()))


def _iso_now() -> str:
    return datetime.now(UTC).isoformat()
