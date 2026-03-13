from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ModelArtifacts:
    cv_accuracy: float
    confusion: list[list[int]]
    oof_probabilities: list[float]
    oof_predictions: list[int]
    feature_importance: dict[str, float]
    latest_probability_up: float
    latest_probability_down: float
    auc: float
    precision: float
    recall: float


def _build_estimators() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=2000)),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=5,
                        min_samples_leaf=8,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def train_direction_models(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> tuple[dict[str, ModelArtifacts], pd.Series]:
    X = dataset[feature_columns]
    y = dataset[target_column].astype(int)
    n_splits = min(5, max(2, len(dataset) // 18))
    if len(dataset) < 30:
        raise ValueError("At least 30 usable rows are required after feature engineering.")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    artifacts: dict[str, ModelArtifacts] = {}

    latest_row = X.iloc[[-1]]
    train_X = X.iloc[:-1]
    train_y = y.iloc[:-1]
    scored_index = dataset.index[:-1]

    for name, estimator in _build_estimators().items():
        oof_probs = pd.Series(index=scored_index, dtype=float)
        oof_preds = pd.Series(index=scored_index, dtype=float)

        for train_idx, test_idx in splitter.split(train_X):
            model = clone(estimator)
            model.fit(train_X.iloc[train_idx], train_y.iloc[train_idx])
            probs = model.predict_proba(train_X.iloc[test_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            oof_probs.iloc[test_idx] = probs
            oof_preds.iloc[test_idx] = preds

        evaluated = oof_probs.dropna().index
        eval_y = train_y.loc[evaluated]
        eval_probs = oof_probs.loc[evaluated]
        eval_preds = oof_preds.loc[evaluated].astype(int)
        confusion = confusion_matrix(eval_y, eval_preds, labels=[0, 1]).tolist()
        accuracy = float((eval_preds == eval_y).mean())
        auc = float(roc_auc_score(eval_y, eval_probs)) if eval_y.nunique() > 1 else 0.5
        tp = int(((eval_preds == 1) & (eval_y == 1)).sum())
        fp = int(((eval_preds == 1) & (eval_y == 0)).sum())
        fn = int(((eval_preds == 0) & (eval_y == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        final_model = clone(estimator)
        final_model.fit(train_X, train_y)
        latest_probability_up = float(final_model.predict_proba(latest_row)[0, 1])
        feature_importance = _extract_feature_importance(final_model, feature_columns)

        artifacts[name] = ModelArtifacts(
            cv_accuracy=accuracy,
            confusion=confusion,
            oof_probabilities=oof_probs.fillna(np.nan).tolist(),
            oof_predictions=oof_preds.fillna(np.nan).tolist(),
            feature_importance=feature_importance,
            latest_probability_up=latest_probability_up,
            latest_probability_down=float(1.0 - latest_probability_up),
            auc=auc,
            precision=float(precision),
            recall=float(recall),
        )

    return artifacts, y


def compute_ensemble(
    dataset: pd.DataFrame,
    target_column: str,
    future_return_column: str,
    model_artifacts: dict[str, ModelArtifacts],
) -> dict[str, float]:
    weights = np.array([max(artifact.cv_accuracy, 1e-6) for artifact in model_artifacts.values()], dtype=float)
    weights = weights / weights.sum()
    probabilities = np.array([artifact.latest_probability_up for artifact in model_artifacts.values()], dtype=float)
    ensemble_up = float(np.dot(weights, probabilities))
    ensemble_down = float(1.0 - ensemble_up)

    future_returns = dataset[future_return_column]
    up_returns = future_returns[dataset[target_column] == 1]
    down_returns = future_returns[dataset[target_column] == 0]
    avg_up = float(up_returns.mean()) if len(up_returns) else 0.0
    avg_down = float(down_returns.mean()) if len(down_returns) else 0.0
    expected_move = ensemble_up * avg_up + ensemble_down * avg_down
    confidence = abs(ensemble_up - 0.5) * 200.0

    return {
        "probability_up": ensemble_up,
        "probability_down": ensemble_down,
        "expected_move": expected_move,
        "confidence": confidence,
        "avg_up_return": avg_up,
        "avg_down_return": avg_down,
    }


def _extract_feature_importance(model: Pipeline, feature_columns: list[str]) -> dict[str, float]:
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    elif hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    else:
        values = np.zeros(len(feature_columns))
    total = float(values.sum()) or 1.0
    normalized = values / total
    return {feature: float(score) for feature, score in zip(feature_columns, normalized)}
