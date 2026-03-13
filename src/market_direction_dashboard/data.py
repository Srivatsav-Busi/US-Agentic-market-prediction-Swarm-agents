from __future__ import annotations

from pathlib import Path

import pandas as pd


DATE_CANDIDATES = ("date", "month", "datetime", "timestamp")


def _normalize_name(name: str) -> str:
    return str(name).strip().replace("\n", " ")


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("nan", "", case=False, regex=False)
        .str.strip()
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if series.astype(str).str.contains("%", regex=False).any():
        numeric = numeric / 100.0
    return numeric


def load_market_data(path: str | Path, date_column: str | None = None) -> tuple[pd.DataFrame, str]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    frame = frame.rename(columns={col: _normalize_name(col) for col in frame.columns})
    resolved_date_column = date_column or detect_date_column(frame)
    frame[resolved_date_column] = pd.to_datetime(frame[resolved_date_column], errors="coerce")
    frame = frame.dropna(subset=[resolved_date_column]).sort_values(resolved_date_column).reset_index(drop=True)

    for column in frame.columns:
        if column == resolved_date_column:
            continue
        frame[column] = _coerce_numeric(frame[column])

    numeric_columns = [col for col in frame.columns if col != resolved_date_column]
    frame[numeric_columns] = frame[numeric_columns].ffill().bfill()
    return frame, resolved_date_column


def detect_date_column(frame: pd.DataFrame) -> str:
    lowered = {str(col).strip().lower(): col for col in frame.columns}
    for candidate in DATE_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError("Could not infer date column. Use --date-column to specify it explicitly.")


def parse_overrides(values: list[str]) -> dict[str, float]:
    overrides: dict[str, float] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Override must be in COLUMN=value format: {item}")
        key, raw_value = item.split("=", 1)
        overrides[key.strip()] = float(raw_value.strip())
    return overrides


def append_live_row(
    frame: pd.DataFrame,
    date_column: str,
    live_as_of: str | None,
    overrides: dict[str, float],
) -> pd.DataFrame:
    if not overrides and not live_as_of:
        return frame

    next_row = frame.iloc[-1].copy()
    if live_as_of:
        next_row[date_column] = pd.to_datetime(live_as_of)
    else:
        inferred = frame[date_column].diff().median()
        next_row[date_column] = frame[date_column].iloc[-1] + (inferred if pd.notna(inferred) else pd.Timedelta(days=30))

    for column, value in overrides.items():
        if column not in frame.columns:
            frame[column] = frame[column].iloc[-1] if len(frame) else value
            next_row[column] = value
        else:
            next_row[column] = value

    result = pd.concat([frame, pd.DataFrame([next_row])], ignore_index=True)
    result = result.sort_values(date_column).reset_index(drop=True)
    numeric_columns = [col for col in result.columns if col != date_column]
    result[numeric_columns] = result[numeric_columns].ffill().bfill()
    return result
