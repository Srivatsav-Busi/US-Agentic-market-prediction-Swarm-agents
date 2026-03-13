from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from market_direction_dashboard.data import append_live_row, load_market_data, parse_overrides
from market_direction_dashboard.features import engineer_features
from market_direction_dashboard.pipeline import run_dashboard_pipeline


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "examples" / "sample_market_data.csv"


class PipelineTests(unittest.TestCase):
    def test_data_load_and_features(self) -> None:
        frame, date_column = load_market_data(SAMPLE)
        bundle = engineer_features(frame, date_column, "NIFTY")
        self.assertEqual(date_column, "Date")
        self.assertIn("target_up", bundle.dataset.columns)
        self.assertGreater(len(bundle.feature_columns), 10)
        self.assertGreater(len(bundle.dataset), 20)

    def test_parse_overrides_and_append(self) -> None:
        overrides = parse_overrides(["NIFTY=24158", "INDIA_VIX=23.36"])
        frame, date_column = load_market_data(SAMPLE)
        appended = append_live_row(frame, date_column, "2026-03-13", overrides)
        self.assertEqual(str(appended.iloc[-1][date_column].date()), "2026-03-13")
        self.assertEqual(float(appended.iloc[-1]["NIFTY"]), 24158.0)

    def test_dashboard_pipeline_outputs_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dashboard.html"
            result = run_dashboard_pipeline(
                input_path=SAMPLE,
                output_path=output,
                index_column="NIFTY",
                live_as_of="2026-03-13",
                overrides={"NIFTY": 24158.0, "INDIA_VIX": 23.36, "USD_INR": 92.30},
            )
            self.assertTrue(output.exists())
            self.assertIn("ensemble", result)
            self.assertIn("monte_carlo", result)


if __name__ == "__main__":
    unittest.main()
