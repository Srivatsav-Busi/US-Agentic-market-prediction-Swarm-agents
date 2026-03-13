from __future__ import annotations

import numpy as np
import pandas as pd

from .modeling import ModelArtifacts


def build_walk_forward_accuracy(
    dates: pd.Series,
    y_true: pd.Series,
    model_artifacts: dict[str, ModelArtifacts],
    window: int = 12,
) -> pd.DataFrame:
    frame = pd.DataFrame({"date": dates.iloc[:-1].reset_index(drop=True), "actual": y_true.iloc[:-1].reset_index(drop=True)})
    for name, artifact in model_artifacts.items():
        preds = pd.Series(artifact.oof_predictions)
        correctness = (preds == frame["actual"]).astype(float)
        correctness[preds.isna()] = np.nan
        frame[name] = correctness.rolling(window=window, min_periods=3).mean()
        frame[f"{name}_prob"] = artifact.oof_probabilities
    return frame


def top_feature_importance(model_artifacts: dict[str, ModelArtifacts], limit: int = 15) -> list[str]:
    aggregate: dict[str, float] = {}
    for artifact in model_artifacts.values():
        for feature, importance in artifact.feature_importance.items():
            aggregate[feature] = aggregate.get(feature, 0.0) + importance
    return [name for name, _ in sorted(aggregate.items(), key=lambda item: item[1], reverse=True)[:limit]]


def correlation_heatmap_data(dataset: pd.DataFrame, feature_columns: list[str], limit: int = 20) -> pd.DataFrame:
    selected = feature_columns[:limit]
    return dataset[selected].corr().fillna(0.0)
