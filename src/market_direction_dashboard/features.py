from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureBundle:
    dataset: pd.DataFrame
    feature_columns: list[str]
    target_column: str
    future_return_column: str


def engineer_features(frame: pd.DataFrame, date_column: str, index_column: str) -> FeatureBundle:
    if index_column not in frame.columns:
        raise ValueError(f"Primary index column '{index_column}' not found in dataset.")

    dataset = frame.copy()
    numeric_columns = [col for col in dataset.columns if col != date_column and pd.api.types.is_numeric_dtype(dataset[col])]
    for column in numeric_columns:
        dataset[f"ret__{column}"] = np.log(dataset[column] / dataset[column].shift(1))

    index_return = f"ret__{index_column}"
    for lag in (1, 2, 3):
        dataset[f"{index_column.lower()}_lag_{lag}"] = dataset[index_return].shift(lag)

    auxiliary_columns = [col for col in numeric_columns if col != index_column]
    for column in auxiliary_columns:
        slug = _slug(column)
        dataset[f"{slug}_ret_lag_1"] = dataset[f"ret__{column}"].shift(1)
        dataset[f"{slug}_ret_lag_2"] = dataset[f"ret__{column}"].shift(2)

    dataset[f"{index_column.lower()}_ma_dev_3"] = dataset[index_column] / dataset[index_column].rolling(3).mean() - 1.0
    dataset[f"{index_column.lower()}_ma_dev_6"] = dataset[index_column] / dataset[index_column].rolling(6).mean() - 1.0
    dataset[f"{index_column.lower()}_mom_3"] = dataset[index_column] / dataset[index_column].shift(3) - 1.0
    dataset[f"{index_column.lower()}_mom_6"] = dataset[index_column] / dataset[index_column].shift(6) - 1.0
    dataset[f"{index_column.lower()}_vol_3"] = dataset[index_return].rolling(3).std()

    us_10y, domestic_10y = detect_yield_columns(dataset.columns, index_column)
    if us_10y and domestic_10y:
        dataset["yield_spread"] = dataset[domestic_10y] - dataset[us_10y]

    future_return_column = f"{index_column.lower()}_future_return"
    dataset[future_return_column] = dataset[index_return].shift(-1)
    target_column = "target_up"
    dataset[target_column] = (dataset[future_return_column] > 0).astype(float)

    protected = {date_column, index_column, target_column, future_return_column}
    feature_columns = [
        col
        for col in dataset.columns
        if col not in protected and pd.api.types.is_numeric_dtype(dataset[col]) and not col.startswith(index_column.lower() + "_future")
    ]
    dataset = dataset.dropna(subset=feature_columns + [target_column, future_return_column]).reset_index(drop=True)
    return FeatureBundle(
        dataset=dataset,
        feature_columns=feature_columns,
        target_column=target_column,
        future_return_column=future_return_column,
    )


def detect_yield_columns(columns: pd.Index, index_column: str) -> tuple[str | None, str | None]:
    lowered = {str(col).lower(): col for col in columns}
    us_match = None
    domestic_match = None
    for lower, original in lowered.items():
        compact = re.sub(r"[^a-z0-9]+", " ", lower)
        if "us" in compact and "10" in compact and ("yr" in compact or "year" in compact or "bond" in compact or "treasury" in compact):
            us_match = original
        elif "10" in compact and ("yr" in compact or "year" in compact or "bond" in compact or "yield" in compact):
            domestic_match = original
    if domestic_match == us_match:
        domestic_match = None
    if domestic_match and index_column.lower() in domestic_match.lower():
        return us_match, domestic_match
    return us_match, domestic_match


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
