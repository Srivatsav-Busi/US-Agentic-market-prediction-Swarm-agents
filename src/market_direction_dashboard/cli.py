from __future__ import annotations

import argparse

from .data import parse_overrides
from .pipeline import run_dashboard_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an AI market direction probability dashboard.")
    parser.add_argument("--input", required=True, help="Path to CSV or XLSX market data.")
    parser.add_argument("--output", required=True, help="Path to output HTML file.")
    parser.add_argument("--index-column", required=True, help="Primary index column to predict.")
    parser.add_argument("--date-column", help="Optional explicit date column name.")
    parser.add_argument("--live-as-of", help="Optional live data date.")
    parser.add_argument("--override", action="append", default=[], help="Live override in COLUMN=value format.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    overrides = parse_overrides(args.override)
    run_dashboard_pipeline(
        input_path=args.input,
        output_path=args.output,
        index_column=args.index_column,
        date_column=args.date_column,
        live_as_of=args.live_as_of,
        overrides=overrides,
    )


if __name__ == "__main__":
    main()
