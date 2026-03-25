from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import erf, sqrt
from typing import Any

import numpy as np
import pandas as pd

from .graph_features import GraphPredictionContext, GraphPredictionPriors
from .models import DataQualitySummary, DecisionTraceStep, SignalFeature, SourceItem


REGIME_NAMES = ("risk_on", "mixed", "risk_off")


@dataclass
class BayesianEvidence:
    probabilities: dict[str, float]
    category_posteriors: dict[str, dict[str, float]]
    signal_score: float
    uncertainty: float
    graph_evidence_adjustments: dict[str, float] = field(default_factory=dict)
    graph_conflict_summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticalDecision:
    label: str
    confidence: float
    final_score: float
    expected_return: float
    expected_volatility: float
    posterior_probabilities: dict[str, float]
    base_probabilities: dict[str, float]
    evidence_probabilities: dict[str, float]
    regime_probabilities: dict[str, float]
    history_coverage: dict[str, Any]
    neutral_band: dict[str, float]
    engine_status: str
    failures: list[str] = field(default_factory=list)
    confidence_components: dict[str, float] = field(default_factory=dict)
    category_posteriors: dict[str, dict[str, float]] = field(default_factory=dict)
    trace_steps: list[dict[str, Any]] = field(default_factory=list)
    graph_priors: dict[str, Any] = field(default_factory=dict)
    graph_feature_summary: dict[str, Any] = field(default_factory=dict)
    graph_evidence_adjustments: dict[str, Any] = field(default_factory=dict)
    graph_conflict_summary: dict[str, Any] = field(default_factory=dict)
    graph_quality_summary: dict[str, Any] = field(default_factory=dict)
    graph_delta_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_statistical_decision(
    target: str,
    snapshot: dict,
    items: list[SourceItem],
    features: list[SignalFeature],
    quality_summary: DataQualitySummary,
    config: dict,
    graph_prediction_context: GraphPredictionContext | None = None,
    graph_delta_summary: dict[str, Any] | None = None,
) -> StatisticalDecision:
    failures: list[str] = []
    minimum_history_rows = int(config.get("minimum_history_rows", 45))
    history_frame = _build_history_frame(snapshot)
    history_coverage = _history_coverage(history_frame, snapshot, target, minimum_history_rows)
    if target not in history_frame or history_frame[target].dropna().shape[0] < minimum_history_rows:
        failures.append("insufficient_target_history")
        return _neutral_decision(history_coverage, failures, quality_summary, graph_prediction_context, graph_delta_summary)

    history_features = _engineer_history_features(history_frame, target)
    usable_history = history_features.dropna(
        subset=[
            "target_return",
            "future_return",
            "pressure_observed",
            "breadth_score",
            "realized_vol_5",
        ]
    ).copy()
    if len(usable_history) < minimum_history_rows:
        failures.append("insufficient_usable_history")
        return _neutral_decision(history_coverage, failures, quality_summary, graph_prediction_context, graph_delta_summary)

    latent_state = _kalman_filter(usable_history["pressure_observed"].to_numpy())
    usable_history["latent_state"] = latent_state

    regime_frame = _infer_regimes(latent_state, usable_history["realized_vol_5"].to_numpy())
    usable_history = pd.concat([usable_history, regime_frame], axis=1)

    return_fit = _fit_dynamic_regression(usable_history)
    if return_fit["status"] != "ok":
        failures.append(return_fit["status"])
    usable_history["return_fitted"] = return_fit["fitted"]

    volatility_fit = _forecast_volatility(usable_history["target_return"].to_numpy())
    if volatility_fit["status"] != "ok":
        failures.append(volatility_fit["status"])
    usable_history["conditional_volatility"] = volatility_fit["series"]

    current_row = usable_history.iloc[-1]
    expected_return = float(return_fit["forecast"])
    expected_volatility = float(volatility_fit["forecast"])
    regime_probabilities = {
        name: float(current_row[f"regime_{name}"])
        for name in REGIME_NAMES
    }

    neutral_band_width = _neutral_band_width(
        config,
        expected_volatility,
        regime_probabilities,
        quality_summary,
    )
    base_probabilities = _ordered_probability_map(
        expected_return=expected_return,
        expected_volatility=expected_volatility,
        latent_state=float(current_row["latent_state"]),
        regime_probabilities=regime_probabilities,
        neutral_band_width=neutral_band_width,
    )

    recent_future_returns = usable_history["future_return"].dropna().tail(60)
    evidence = _aggregate_bayesian_evidence(
        items=items,
        features=features,
        quality_summary=quality_summary,
        recent_returns=recent_future_returns,
        neutral_band_width=neutral_band_width,
        graph_prediction_context=graph_prediction_context,
    )
    posterior_probabilities = _combine_probabilities(
        base_probabilities=base_probabilities,
        evidence_probabilities=evidence.probabilities,
        quality_summary=quality_summary,
        graph_priors=graph_prediction_context.priors if graph_prediction_context else None,
    )
    posterior_probabilities = _apply_graph_delta_modifiers(
        posterior_probabilities=posterior_probabilities,
        graph_delta_summary=graph_delta_summary,
    )

    final_score = float(posterior_probabilities["UP"] - posterior_probabilities["DOWN"])
    label = _select_label(
        posterior_probabilities=posterior_probabilities,
        expected_return=expected_return,
        neutral_band_width=neutral_band_width,
        failures=failures,
        quality_summary=quality_summary,
    )
    confidence_components = _confidence_components(
        posterior_probabilities=posterior_probabilities,
        quality_summary=quality_summary,
        regime_probabilities=regime_probabilities,
        history_coverage=history_coverage,
        expected_volatility=expected_volatility,
        evidence=evidence,
    )
    confidence = float(
        max(
            28.0,
            min(
                91.0,
                100.0
                * (
                    confidence_components["posterior_separation"] * 0.32
                    + confidence_components["evidence_quality"] * 0.16
                    + confidence_components["regime_certainty"] * 0.14
                    + confidence_components["history_completeness"] * 0.16
                    + confidence_components["volatility_stability"] * 0.12
                    + confidence_components["signal_alignment"] * 0.10
                ),
            ),
        )
    )

    if failures and label != "NEUTRAL":
        confidence = min(confidence, 62.0)
    engine_status = _engine_status(failures, history_coverage)
    trace_steps = [
        DecisionTraceStep(
            stage="signal_smoothing",
            summary="Smoothed cross-asset market pressure into a latent state with Kalman-style filtering.",
            references=["pressure_observed", "latent_state"],
            value=round(float(current_row["latent_state"]), 4),
        ).to_dict(),
        DecisionTraceStep(
            stage="regime_detection",
            summary="Estimated live regime probabilities from latent pressure and realized volatility.",
            references=[f"{name}:{regime_probabilities[name]:.2f}" for name in REGIME_NAMES],
            value=round(max(regime_probabilities.values()), 4),
        ).to_dict(),
        DecisionTraceStep(
            stage="return_modeling",
            summary="Forecast next-session return with dynamic regression on lagged returns and cross-asset exogenous features.",
            references=["lag_target_return_1", "lag_target_return_2", "breadth_score", "yield_pressure"],
            value=round(expected_return, 5),
        ).to_dict(),
        DecisionTraceStep(
            stage="probability_mapping",
            summary="Mapped the latent score into ordered class probabilities and updated them with Bayesian evidence aggregation.",
            references=[f"{label_name}:{posterior_probabilities[label_name]:.2f}" for label_name in ("DOWN", "NEUTRAL", "UP")],
            value=round(final_score, 4),
        ).to_dict(),
    ]
    if graph_prediction_context and graph_prediction_context.priors.influence_weight > 0.0:
        trace_steps.append(
            DecisionTraceStep(
                stage="graph_prior_integration",
                summary="Applied bounded graph priors and graph-aware evidence-quality adjustments before the final posterior normalization.",
                references=[
                    f"graph_influence:{graph_prediction_context.priors.influence_weight:.2f}",
                    f"consensus:{graph_prediction_context.priors.consensus_score:.2f}",
                    f"contradiction:{graph_prediction_context.priors.contradiction_score:.2f}",
                ],
                value=round(graph_prediction_context.priors.credibility_weighted_pressure, 4),
            ).to_dict()
        )
    if graph_delta_summary and graph_delta_summary.get("delta_available"):
        trace_steps.append(
            DecisionTraceStep(
                stage="graph_delta_integration",
                summary="Applied temporal graph-delta modifiers to account for narrative acceleration, reversal risk, and cross-day structure change.",
                references=[
                    f"theme_acceleration:{float(graph_delta_summary.get('theme_acceleration', 0.0)):.2f}",
                    f"delta_strength:{float(graph_delta_summary.get('delta_strength', 0.0)):.2f}",
                    f"reversal:{int(bool(graph_delta_summary.get('narrative_reversal_flag')))}",
                ],
                value=round(float(graph_delta_summary.get("delta_strength", 0.0)), 4),
            ).to_dict()
        )
    return StatisticalDecision(
        label=label,
        confidence=round(confidence, 1),
        final_score=round(final_score, 4),
        expected_return=round(expected_return, 6),
        expected_volatility=round(expected_volatility, 6),
        posterior_probabilities={key: round(value, 4) for key, value in posterior_probabilities.items()},
        base_probabilities={key: round(value, 4) for key, value in base_probabilities.items()},
        evidence_probabilities={key: round(value, 4) for key, value in evidence.probabilities.items()},
        regime_probabilities={key: round(value, 4) for key, value in regime_probabilities.items()},
        history_coverage=history_coverage,
        neutral_band={
            "base_width": round(float(config.get("neutral_band_base", 0.0025)), 6),
            "effective_width": round(neutral_band_width, 6),
        },
        engine_status=engine_status,
        failures=failures,
        confidence_components={key: round(value, 4) for key, value in confidence_components.items()},
        category_posteriors=evidence.category_posteriors,
        trace_steps=trace_steps,
        graph_priors=graph_prediction_context.priors.to_dict() if graph_prediction_context else {},
        graph_feature_summary=(graph_prediction_context.feature_summary if graph_prediction_context else {}),
        graph_evidence_adjustments=evidence.graph_evidence_adjustments,
        graph_conflict_summary=evidence.graph_conflict_summary,
        graph_quality_summary=quality_summary.graph_quality_summary,
        graph_delta_summary=graph_delta_summary or {},
    )


def _neutral_decision(
    history_coverage: dict[str, Any],
    failures: list[str],
    quality_summary: DataQualitySummary,
    graph_prediction_context: GraphPredictionContext | None = None,
    graph_delta_summary: dict[str, Any] | None = None,
) -> StatisticalDecision:
    evidence_quality = float(max(0.2, quality_summary.average_quality_score or 0.0))
    posterior = {"DOWN": 0.22, "NEUTRAL": 0.56, "UP": 0.22}
    return StatisticalDecision(
        label="NEUTRAL",
        confidence=round(32.0 + evidence_quality * 18.0, 1),
        final_score=0.0,
        expected_return=0.0,
        expected_volatility=0.0,
        posterior_probabilities=posterior,
        base_probabilities=posterior,
        evidence_probabilities=posterior,
        regime_probabilities={name: round(1.0 / len(REGIME_NAMES), 4) for name in REGIME_NAMES},
        history_coverage=history_coverage,
        neutral_band={"base_width": 0.0025, "effective_width": 0.004},
        engine_status="DEGRADED",
        failures=failures,
        confidence_components={
            "posterior_separation": 0.1,
            "evidence_quality": evidence_quality,
            "regime_certainty": 0.33,
            "history_completeness": min(1.0, history_coverage.get("target_rows", 0) / max(history_coverage.get("minimum_rows", 1), 1)),
            "volatility_stability": 0.25,
            "signal_alignment": 0.2,
        },
        category_posteriors={},
        trace_steps=[
            DecisionTraceStep(
                stage="statistical_fallback",
                summary="The live run lacked enough history to support a directional statistical read, so the engine collapsed to neutral.",
                references=failures,
                value=0.0,
            ).to_dict()
        ],
        graph_priors=graph_prediction_context.priors.to_dict() if graph_prediction_context else {},
        graph_feature_summary=(graph_prediction_context.feature_summary if graph_prediction_context else {}),
        graph_evidence_adjustments=(graph_prediction_context.evidence_adjustments if graph_prediction_context else {}),
        graph_conflict_summary=(graph_prediction_context.conflict_summary if graph_prediction_context else {}),
        graph_quality_summary=quality_summary.graph_quality_summary,
        graph_delta_summary=graph_delta_summary or {},
    )


def _build_history_frame(snapshot: dict) -> pd.DataFrame:
    history = snapshot.get("history", {})
    columns: dict[str, pd.Series] = {}
    for label, rows in history.items():
        if not rows:
            continue
        series = {
            pd.to_datetime(row["date"], errors="coerce"): float(row["value"])
            for row in rows
            if row.get("date") and row.get("value") is not None
        }
        cleaned = pd.Series(series, dtype=float).sort_index()
        cleaned = cleaned[cleaned.index.notna()]
        if not cleaned.empty:
            columns[label] = cleaned
    if not columns:
        return pd.DataFrame()
    frame = pd.DataFrame(columns).sort_index()
    return frame.ffill().dropna(how="all")


def _history_coverage(history_frame: pd.DataFrame, snapshot: dict, target: str, minimum_rows: int) -> dict[str, Any]:
    available_series = sorted(snapshot.get("history", {}).keys())
    target_rows = int(history_frame[target].dropna().shape[0]) if target in history_frame else 0
    complete_rows = int(history_frame.dropna().shape[0]) if not history_frame.empty else 0
    return {
        "available_series": available_series,
        "target_rows": target_rows,
        "complete_rows": complete_rows,
        "minimum_rows": minimum_rows,
    }


def _engineer_history_features(history_frame: pd.DataFrame, target: str) -> pd.DataFrame:
    dataset = history_frame.copy()
    dataset["target_return"] = np.log(dataset[target] / dataset[target].shift(1))
    dataset["future_return"] = dataset["target_return"].shift(-1)
    dataset["lag_target_return_1"] = dataset["target_return"].shift(1)
    dataset["lag_target_return_2"] = dataset["target_return"].shift(2)

    signed_pressure = {}
    for label, alias, sign in (
        ("VIX", "vix_pressure", -1.0),
        ("US 10 YR TREASURY", "yield_pressure", -1.0),
        ("DXY", "dxy_pressure", -1.0),
        ("WTI CRUDE OIL", "oil_pressure", -1.0),
    ):
        if label in dataset:
            signed_pressure[alias] = sign * np.log(dataset[label] / dataset[label].shift(1))
            dataset[alias] = signed_pressure[alias]
        else:
            dataset[alias] = np.nan

    breadth_sources = [label for label in ("S&P 500", "NASDAQ 100", "DOW JONES", "RUSSELL 2000") if label in dataset]
    if breadth_sources:
        breadth_matrix = []
        for label in breadth_sources:
            breadth_matrix.append(np.sign(np.log(dataset[label] / dataset[label].shift(1))).fillna(0.0))
        dataset["breadth_score"] = pd.concat(breadth_matrix, axis=1).mean(axis=1)
    else:
        dataset["breadth_score"] = 0.0

    dataset["target_momentum_3"] = dataset["target_return"].rolling(3).mean()
    dataset["target_momentum_5"] = dataset["target_return"].rolling(5).mean()
    dataset["realized_vol_5"] = dataset["target_return"].rolling(5).std().clip(lower=1e-6)

    pressure_columns = ["breadth_score", "target_momentum_3", "yield_pressure", "vix_pressure", "dxy_pressure", "oil_pressure"]
    standardized = []
    for column in pressure_columns:
        series = dataset[column].replace([np.inf, -np.inf], np.nan)
        mean = series.mean(skipna=True)
        std = float(series.std(skipna=True)) or 1.0
        standardized.append(((series - mean) / std).clip(-3.0, 3.0).fillna(0.0))
    dataset["pressure_observed"] = pd.concat(standardized, axis=1).mean(axis=1)
    return dataset


def _kalman_filter(values: np.ndarray) -> np.ndarray:
    filtered = np.zeros(len(values), dtype=float)
    if len(values) == 0:
        return filtered
    state = float(values[0])
    covariance = 1.0
    process_variance = 0.05
    observation_variance = max(0.05, float(np.nanvar(values)) or 0.2)
    for index, observation in enumerate(values):
        covariance += process_variance
        gain = covariance / (covariance + observation_variance)
        state = state + gain * (float(observation) - state)
        covariance = (1.0 - gain) * covariance
        filtered[index] = state
    return filtered


def _infer_regimes(latent_state: np.ndarray, realized_volatility: np.ndarray) -> pd.DataFrame:
    if len(latent_state) == 0:
        return pd.DataFrame(columns=[f"regime_{name}" for name in REGIME_NAMES])
    vol = np.nan_to_num(realized_volatility, nan=np.nanmedian(realized_volatility))
    vol_mean = float(np.nanmean(vol)) if np.isfinite(vol).any() else 0.0
    vol_std = float(np.nanstd(vol)) or 1.0
    transition = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.12, 0.76, 0.12],
            [0.05, 0.15, 0.80],
        ],
        dtype=float,
    )
    posterior = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    probabilities = []
    for state, sigma in zip(latent_state, vol, strict=False):
        vol_z = (float(sigma) - vol_mean) / vol_std
        emission = _softmax(
            np.array(
                [
                    1.3 * float(state) - 0.7 * vol_z,
                    -0.35 * abs(float(state)) - 0.15 * abs(vol_z),
                    -1.3 * float(state) + 0.85 * vol_z,
                ]
            )
        )
        posterior = transition.T @ posterior
        posterior = posterior * emission
        posterior = posterior / posterior.sum()
        probabilities.append(posterior.copy())
    frame = pd.DataFrame(probabilities, columns=[f"regime_{name}" for name in REGIME_NAMES])
    return frame


def _fit_dynamic_regression(dataset: pd.DataFrame) -> dict[str, Any]:
    feature_columns = [
        "lag_target_return_1",
        "lag_target_return_2",
        "breadth_score",
        "yield_pressure",
        "vix_pressure",
        "dxy_pressure",
        "oil_pressure",
        "latent_state",
        "regime_risk_on",
        "regime_mixed",
        "regime_risk_off",
    ]
    train = dataset.dropna(subset=["future_return"] + feature_columns).copy()
    if len(train) < 25:
        return {"status": "insufficient_return_model_history", "forecast": 0.0, "fitted": pd.Series(np.zeros(len(dataset)), index=dataset.index)}
    X_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train["future_return"].to_numpy(dtype=float)
    X_design = np.column_stack([np.ones(len(X_train)), X_train])
    coefficients, *_ = np.linalg.lstsq(X_design, y_train, rcond=None)
    fitted = np.full(len(dataset), np.nan, dtype=float)
    fitted_index = train.index
    fitted[dataset.index.get_indexer(fitted_index)] = X_design @ coefficients

    latest = dataset.iloc[-1]
    latest_design = np.array(
        [
            1.0,
            *[float(latest[column]) if pd.notna(latest[column]) else 0.0 for column in feature_columns],
        ]
    )
    forecast = float(latest_design @ coefficients)
    forecast = float(np.clip(forecast, -0.05, 0.05))
    return {"status": "ok", "forecast": forecast, "fitted": pd.Series(fitted, index=dataset.index)}


def _forecast_volatility(returns: np.ndarray) -> dict[str, Any]:
    clean = np.nan_to_num(returns.astype(float), nan=0.0)
    if len(clean) < 10:
        return {"status": "insufficient_volatility_history", "forecast": 0.0, "series": pd.Series(np.zeros(len(clean)))}
    decay = 0.94
    leverage = 1.35
    variances = np.zeros(len(clean), dtype=float)
    variances[0] = max(clean.var(), 1e-6)
    for index in range(1, len(clean)):
        shock = clean[index - 1] ** 2
        if clean[index - 1] < 0:
            shock *= leverage
        variances[index] = decay * variances[index - 1] + (1.0 - decay) * shock
    forecast = decay * variances[-1] + (1.0 - decay) * (clean[-1] ** 2 * (leverage if clean[-1] < 0 else 1.0))
    return {
        "status": "ok",
        "forecast": float(sqrt(max(forecast, 1e-8))),
        "series": pd.Series(np.sqrt(np.maximum(variances, 1e-8))),
    }


def _ordered_probability_map(
    expected_return: float,
    expected_volatility: float,
    latent_state: float,
    regime_probabilities: dict[str, float],
    neutral_band_width: float,
) -> dict[str, float]:
    regime_skew = 0.003 * (regime_probabilities["risk_on"] - regime_probabilities["risk_off"])
    latent_score = expected_return + regime_skew + 0.002 * latent_state
    sigma = max(expected_volatility, neutral_band_width * 0.75, 0.002)
    down = _normal_cdf((-neutral_band_width - latent_score) / sigma)
    up = 1.0 - _normal_cdf((neutral_band_width - latent_score) / sigma)
    neutral = max(0.0, 1.0 - down - up)
    probs = _normalize({"DOWN": down, "NEUTRAL": neutral, "UP": up})
    return probs


def _aggregate_bayesian_evidence(
    items: list[SourceItem],
    features: list[SignalFeature],
    quality_summary: DataQualitySummary,
    recent_returns: pd.Series,
    neutral_band_width: float,
    graph_prediction_context: GraphPredictionContext | None = None,
) -> BayesianEvidence:
    prior_counts = {"DOWN": 1.0, "NEUTRAL": 1.0, "UP": 1.0}
    for value in recent_returns:
        if value > neutral_band_width:
            prior_counts["UP"] += 1.0
        elif value < -neutral_band_width:
            prior_counts["DOWN"] += 1.0
        else:
            prior_counts["NEUTRAL"] += 1.0

    category_counts: dict[str, dict[str, float]] = {}
    for item in items:
        bucket = category_counts.setdefault(item.category, {"DOWN": 1.0, "NEUTRAL": 1.0, "UP": 1.0})
        contribution = abs(item.impact_score) * max(0.15, item.quality_score) * max(0.2, item.credibility_score)
        contribution *= max(0.2, item.freshness_score)
        if item.proxy_used:
            contribution *= 0.8
        if item.impact_score > 0.05:
            bucket["UP"] += contribution
        elif item.impact_score < -0.05:
            bucket["DOWN"] += contribution
        else:
            bucket["NEUTRAL"] += contribution * 0.7
            bucket["UP"] += contribution * 0.15
            bucket["DOWN"] += contribution * 0.15

    for feature in features:
        category = category_counts.setdefault(feature.category, {"DOWN": 1.0, "NEUTRAL": 1.0, "UP": 1.0})
        contribution = feature.strength * max(0.2, feature.time_decay_weight)
        contribution *= max(0.4, 1.0 - feature.conflict_count * 0.15)
        if feature.direction == "bullish":
            category["UP"] += contribution
        elif feature.direction == "bearish":
            category["DOWN"] += contribution
        else:
            category["NEUTRAL"] += contribution

    category_posteriors = {name: _normalize(values) for name, values in category_counts.items()}
    counts = prior_counts.copy()
    for values in category_counts.values():
        for label in counts:
            counts[label] += values[label] - 1.0

    counts["NEUTRAL"] += 0.5 * len(quality_summary.gate_failures)
    counts["NEUTRAL"] += 0.35 * quality_summary.proxy_item_count
    graph_evidence_adjustments = graph_prediction_context.evidence_adjustments if graph_prediction_context else {}
    graph_conflict_summary = graph_prediction_context.conflict_summary if graph_prediction_context else {}
    duplicate_penalty = float(graph_evidence_adjustments.get("duplicate_penalty", 0.0))
    corroboration_boost = float(graph_evidence_adjustments.get("independent_corroboration_boost", 0.0))
    contradiction_penalty = float(graph_evidence_adjustments.get("contradiction_penalty", 0.0))
    graph_quality = quality_summary.graph_quality_summary or {}
    stale_cluster_penalty = float(graph_quality.get("stale_cluster_penalty", 0.0))
    source_monoculture_penalty = float(graph_quality.get("source_monoculture_penalty", 0.0))
    if duplicate_penalty > 0.0:
        counts["UP"] *= max(0.86, 1.0 - duplicate_penalty)
        counts["DOWN"] *= max(0.86, 1.0 - duplicate_penalty)
        counts["NEUTRAL"] += duplicate_penalty * max(len(items), 1) * 0.35
    if corroboration_boost > 0.0:
        dominant_label = "UP" if counts["UP"] >= counts["DOWN"] else "DOWN"
        counts[dominant_label] *= 1.0 + corroboration_boost
    if contradiction_penalty > 0.0:
        counts["UP"] *= max(0.8, 1.0 - contradiction_penalty * 0.7)
        counts["DOWN"] *= max(0.8, 1.0 - contradiction_penalty * 0.7)
        counts["NEUTRAL"] += contradiction_penalty * max(len(features) + len(items), 1) * 0.25
    if stale_cluster_penalty > 0.0:
        counts["UP"] *= max(0.84, 1.0 - stale_cluster_penalty * 0.8)
        counts["DOWN"] *= max(0.84, 1.0 - stale_cluster_penalty * 0.8)
        counts["NEUTRAL"] += stale_cluster_penalty * max(len(items), 1) * 0.3
    if source_monoculture_penalty > 0.0:
        counts["UP"] *= max(0.84, 1.0 - source_monoculture_penalty * 0.7)
        counts["DOWN"] *= max(0.84, 1.0 - source_monoculture_penalty * 0.7)
        counts["NEUTRAL"] += source_monoculture_penalty * max(len(items), 1) * 0.35
    probabilities = _normalize(counts)
    signal_score = float(probabilities["UP"] - probabilities["DOWN"])
    uncertainty = 1.0 - max(probabilities.values())
    rounded_categories = {
        name: {label: round(value, 4) for label, value in values.items()}
        for name, values in category_posteriors.items()
    }
    return BayesianEvidence(
        probabilities={label: round(value, 4) for label, value in probabilities.items()},
        category_posteriors=rounded_categories,
        signal_score=signal_score,
        uncertainty=uncertainty,
        graph_evidence_adjustments={
            **{key: round(float(value), 4) for key, value in graph_evidence_adjustments.items()},
            "stale_cluster_penalty": round(stale_cluster_penalty, 4),
            "source_monoculture_penalty": round(source_monoculture_penalty, 4),
        },
        graph_conflict_summary=graph_conflict_summary,
    )


def _combine_probabilities(
    base_probabilities: dict[str, float],
    evidence_probabilities: dict[str, float],
    quality_summary: DataQualitySummary,
    graph_priors: GraphPredictionPriors | None = None,
) -> dict[str, float]:
    graph_quality = quality_summary.graph_quality_summary or {}
    evidence_weight = 0.25 + 0.20 * max(0.0, min(1.0, quality_summary.average_quality_score))
    evidence_weight += float(graph_quality.get("independent_corroboration_boost", 0.0)) * 0.12
    evidence_weight -= float(graph_quality.get("source_monoculture_penalty", 0.0)) * 0.18
    evidence_weight -= float(graph_quality.get("contradiction_penalty", 0.0)) * 0.12
    evidence_weight = max(0.16, min(0.48, evidence_weight))
    base_weight = 1.0 - evidence_weight
    scores = {}
    for label in ("DOWN", "NEUTRAL", "UP"):
        scores[label] = (base_probabilities[label] ** base_weight) * (evidence_probabilities[label] ** evidence_weight)
    posterior = _normalize(scores)
    if not graph_priors or graph_priors.influence_weight <= 0.0:
        return posterior

    graph_scores = {
        "UP": max(
            1e-8,
            0.33
            + graph_priors.credibility_weighted_pressure * 0.18
            + (graph_priors.bullish_path_strength - graph_priors.bearish_path_strength) * 0.12
            + graph_priors.consensus_score * 0.05,
        ),
        "DOWN": max(
            1e-8,
            0.33
            - graph_priors.credibility_weighted_pressure * 0.18
            + (graph_priors.bearish_path_strength - graph_priors.bullish_path_strength) * 0.12
            + graph_priors.consensus_score * 0.05,
        ),
        "NEUTRAL": max(
            1e-8,
            0.34 + graph_priors.contradiction_score * 0.18 + (0.08 if graph_priors.sparse_graph else 0.0),
        ),
    }
    graph_probabilities = _normalize(graph_scores)
    influence = max(0.0, min(0.18, graph_priors.influence_weight))
    blended = {
        label: posterior[label] * (1.0 - influence) + graph_probabilities[label] * influence
        for label in ("DOWN", "NEUTRAL", "UP")
    }
    return _normalize(blended)


def _apply_graph_delta_modifiers(
    *,
    posterior_probabilities: dict[str, float],
    graph_delta_summary: dict[str, Any] | None,
) -> dict[str, float]:
    if not graph_delta_summary or not graph_delta_summary.get("delta_available"):
        return posterior_probabilities
    adjusted = dict(posterior_probabilities)
    delta_features = graph_delta_summary.get("features") or {}
    bullish_shift = float(delta_features.get("graph_delta__bullish_path_change", 0.0))
    bearish_shift = float(delta_features.get("graph_delta__bearish_path_change", 0.0))
    theme_acceleration = float(graph_delta_summary.get("theme_acceleration", 0.0))
    delta_strength = float(graph_delta_summary.get("delta_strength", 0.0))
    reversal_flag = bool(graph_delta_summary.get("narrative_reversal_flag"))
    directional_nudge = max(-0.05, min(0.05, (bullish_shift - bearish_shift) * 0.12 + theme_acceleration * 0.04))
    if directional_nudge > 0.0:
        adjusted["UP"] += directional_nudge
        adjusted["NEUTRAL"] = max(1e-8, adjusted["NEUTRAL"] - directional_nudge * 0.75)
    elif directional_nudge < 0.0:
        adjusted["DOWN"] += abs(directional_nudge)
        adjusted["NEUTRAL"] = max(1e-8, adjusted["NEUTRAL"] - abs(directional_nudge) * 0.75)
    if reversal_flag:
        neutral_boost = min(0.08, 0.03 + delta_strength * 0.08)
        adjusted["NEUTRAL"] += neutral_boost
        adjusted["UP"] *= max(0.72, 1.0 - neutral_boost * 0.8)
        adjusted["DOWN"] *= max(0.72, 1.0 - neutral_boost * 0.8)
    return _normalize(adjusted)


def _select_label(
    posterior_probabilities: dict[str, float],
    expected_return: float,
    neutral_band_width: float,
    failures: list[str],
    quality_summary: DataQualitySummary,
) -> str:
    graph_quality = quality_summary.graph_quality_summary or {}
    if graph_quality.get("severe_graph_risk"):
        return "NEUTRAL"
    if failures or quality_summary.gate_failures:
        return "NEUTRAL"
    top_label = max(posterior_probabilities, key=posterior_probabilities.get)
    if abs(expected_return) <= neutral_band_width:
        return "NEUTRAL"
    if posterior_probabilities[top_label] < 0.44:
        return "NEUTRAL"
    return top_label


def _neutral_band_width(
    config: dict,
    expected_volatility: float,
    regime_probabilities: dict[str, float],
    quality_summary: DataQualitySummary,
) -> float:
    base = float(config.get("neutral_band_base", 0.0025))
    volatility_component = expected_volatility * float(config.get("neutral_band_volatility_multiplier", 0.45))
    regime_uncertainty = 1.0 - max(regime_probabilities.values())
    quality_component = 0.0006 * len(quality_summary.gate_failures)
    graph_quality = quality_summary.graph_quality_summary or {}
    graph_component = (
        float(graph_quality.get("contradiction_penalty", 0.0)) * 0.004
        + float(graph_quality.get("source_monoculture_penalty", 0.0)) * 0.003
        + float(graph_quality.get("stale_cluster_penalty", 0.0)) * 0.003
    )
    return float(min(0.02, max(base, base + volatility_component + regime_uncertainty * 0.004 + quality_component + graph_component)))


def _confidence_components(
    posterior_probabilities: dict[str, float],
    quality_summary: DataQualitySummary,
    regime_probabilities: dict[str, float],
    history_coverage: dict[str, Any],
    expected_volatility: float,
    evidence: BayesianEvidence,
) -> dict[str, float]:
    separation = max(posterior_probabilities.values()) - sorted(posterior_probabilities.values())[1]
    regime_certainty = max(regime_probabilities.values())
    history_complete = min(1.0, history_coverage.get("target_rows", 0) / max(history_coverage.get("minimum_rows", 1), 1))
    volatility_stability = max(0.0, min(1.0, 1.0 - expected_volatility / 0.03))
    signal_alignment = max(0.0, 1.0 - abs((posterior_probabilities["UP"] - posterior_probabilities["DOWN"]) - evidence.signal_score))
    graph_quality = quality_summary.graph_quality_summary or {}
    graph_component = max(
        0.0,
        min(
            1.0,
            float(graph_quality.get("graph_quality_score", quality_summary.average_quality_score)),
        ),
    )
    return {
        "posterior_separation": max(0.0, min(1.0, separation * 1.8)),
        "evidence_quality": max(0.0, min(1.0, quality_summary.average_quality_score * 0.7 + graph_component * 0.3)),
        "regime_certainty": max(0.0, min(1.0, regime_certainty)),
        "history_completeness": history_complete,
        "volatility_stability": volatility_stability,
        "signal_alignment": signal_alignment,
    }


def _engine_status(failures: list[str], history_coverage: dict[str, Any]) -> str:
    if failures:
        return "DEGRADED"
    if history_coverage.get("complete_rows", 0) < history_coverage.get("minimum_rows", 0):
        return "DEGRADED"
    return "HEALTHY"


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    raw = np.exp(shifted)
    return raw / raw.sum()


def _normalize(values: dict[str, float]) -> dict[str, float]:
    total = float(sum(max(value, 1e-8) for value in values.values()))
    return {label: max(value, 1e-8) / total for label, value in values.items()}


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))
