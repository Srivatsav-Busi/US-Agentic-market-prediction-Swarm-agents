from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .dashboard import render_dashboard
from .data import append_live_row, load_market_data
from .evaluation import build_walk_forward_accuracy, correlation_heatmap_data, top_feature_importance
from .features import engineer_features
from .modeling import compute_ensemble, train_direction_models
from .simulation import combined_signal, run_gbm_simulation


def run_dashboard_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    index_column: str,
    date_column: str | None = None,
    live_as_of: str | None = None,
    overrides: dict[str, float] | None = None,
) -> dict:
    overrides = overrides or {}
    raw_frame, resolved_date = load_market_data(input_path, date_column=date_column)
    augmented_frame = append_live_row(raw_frame, resolved_date, live_as_of, overrides)
    feature_bundle = engineer_features(augmented_frame, resolved_date, index_column)
    dataset = feature_bundle.dataset

    model_artifacts, y = train_direction_models(dataset, feature_bundle.feature_columns, feature_bundle.target_column)
    ensemble = compute_ensemble(dataset, feature_bundle.target_column, feature_bundle.future_return_column, model_artifacts)
    walk_forward = build_walk_forward_accuracy(dataset[resolved_date], y, model_artifacts)
    top_features = top_feature_importance(model_artifacts)
    correlation = correlation_heatmap_data(dataset, top_features)

    vix_column = next((col for col in augmented_frame.columns if "vix" in col.lower()), None)
    vix_level = float(augmented_frame[vix_column].iloc[-1]) if vix_column else None
    simulation = run_gbm_simulation(
        index_prices=augmented_frame[index_column],
        index_returns=dataset[f"ret__{index_column}"],
        date_series=augmented_frame[resolved_date],
        vix_level=vix_level,
    )
    signal_label, signal_confidence = combined_signal(ensemble["probability_up"], simulation.summary["p_up"])

    result = {
        "project_name": "AI Market Direction Probability Dashboard",
        "index_column": index_column,
        "latest_price": float(augmented_frame[index_column].iloc[-1]),
        "summary_text": "Three time-series machine-learning models, walk-forward evaluation, and Monte Carlo simulation combined into a single local-first quant dashboard.",
        "ensemble": ensemble,
        "models": {name: asdict(artifact) for name, artifact in model_artifacts.items()},
        "walk_forward": walk_forward.to_dict(orient="list"),
        "top_features": top_features,
        "correlation": correlation.to_dict(),
        "live_badge": f"Live Data as of {pd.to_datetime(live_as_of).date()}" if live_as_of else None,
        "monte_carlo": {
            "paths": simulation.paths.tolist(),
            "summary": simulation.summary,
            "density_surface": simulation.density_surface.tolist(),
            "price_axis": simulation.price_axis.tolist(),
        },
        "combined_signal": {
            "label": signal_label,
            "confidence": signal_confidence,
        },
    }
    render_dashboard(result, output_path)
    return result
