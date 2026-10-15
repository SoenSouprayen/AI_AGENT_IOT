from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.analysis.audit_raw_dataset import RAW_MEANS, aggregate_raw, compare_aggregation
from scripts.ML.evaluate_csv_agent import decision_metrics, evaluate_dataset
from src.prediction.geometry_schema import MODES


class CSVStudyTests(unittest.TestCase):
    def raw_frame(self):
        rows = []
        values = [15, 1000, 7600, 7000, -100, 200, -4, 5, -120, 10, 500]
        for seed in (42, 43):
            for mode in MODES:
                for packet in range(2):
                    rows.append({
                        **dict(zip(RAW_MEANS, values)), "N": 10000,
                        "environment": "urban", "scenario_seed": seed,
                        "replication": 1, "mode": mode, "max_elevation_deg": 90,
                        "success": packet == 0 if mode == "DR8" else False,
                    })
        return pd.DataFrame(rows)

    def aggregate(self, frame, chunksize=3):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            frame.to_csv(path, index=False)
            return aggregate_raw(path, chunksize)

    def test_chunked_counts_means_and_false_values(self):
        rebuilt, report = self.aggregate(self.raw_frame())
        single, _ = self.aggregate(self.raw_frame(), 1000)
        pd.testing.assert_frame_equal(rebuilt, single)
        self.assertEqual(report["raw_rows"], 32)
        self.assertTrue((rebuilt["n_packets"] == 2).all())
        self.assertTrue((rebuilt.loc[rebuilt["mode"] == "DR8", "per"] == 0.5).all())
        self.assertTrue((rebuilt.loc[rebuilt["mode"] == "DR9", "per"] == 1).all())
        self.assertTrue((rebuilt["visibility_window_mean_s"] == 500).all())

    def test_comparison_detects_missing_and_changed_results(self):
        rebuilt, _ = self.aggregate(self.raw_frame())
        self.assertTrue(compare_aggregation(rebuilt, rebuilt)["matches_existing"])
        altered = rebuilt.copy()
        altered.loc[0, "n_packets"] += 1
        self.assertFalse(compare_aggregation(rebuilt, altered)["matches_existing"])
        self.assertFalse(compare_aggregation(rebuilt, rebuilt.iloc[1:])["matches_existing"])

    def test_finer_bins_and_ninety_degree_boundary(self):
        raw = self.raw_frame()
        second = raw.copy()
        raw["elevation_deg"] = 15.5
        second["elevation_deg"] = 16.5
        boundary = raw.copy()
        boundary["elevation_deg"] = 90
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            pd.concat([raw, second, boundary]).to_csv(path, index=False)
            fine, report = aggregate_raw(path, chunksize=3, bin_size=1)
            coarse, _ = aggregate_raw(path, chunksize=3, bin_size=5)
            self.assertEqual(set(fine["elevation_bin"]), {15, 16, 89})
            self.assertEqual(set(coarse["elevation_bin"]), {15, 85})
            self.assertEqual(report["elevation_bin_size_deg"], 1)
            self.assertEqual(fine["n_packets"].sum(), coarse["n_packets"].sum())
            with self.assertRaises(ValueError):
                aggregate_raw(path, bin_size=2)

    def test_raw_rejects_missing_invalid_boolean_and_nonfinite(self):
        for column, value in (("success", "unknown"), ("snr_db", np.inf),
                              ("elevation_deg", -1), ("distance_km", np.nan)):
            frame = self.raw_frame().astype({column: object})
            frame.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                self.aggregate(frame)

    def test_grouped_predictions_and_metrics(self):
        rebuilt, _ = self.aggregate(self.raw_frame())
        report, predictions = evaluate_dataset(rebuilt, folds=2, trees=5)
        for split in report["splits"]:
            self.assertFalse(set(split["train_seeds"]) & set(split["test_seeds"]))
        self.assertTrue((predictions.groupby("scenario_seed")["fold"].nunique() == 1).all())
        self.assertEqual(report["metrics"]["N_rule_learned_on_train"]["decision_accuracy"], 1)
        self.assertTrue(np.isfinite(predictions["RF_mode_N"]).all())

    def test_optimal_ties_count_as_correct(self):
        rebuilt, _ = self.aggregate(self.raw_frame())
        rebuilt["per"] = 0.5
        metrics = decision_metrics(rebuilt, None, np.full(len(rebuilt), "SF12", dtype=object))
        self.assertEqual(metrics["decision_accuracy"], 1)
        self.assertEqual(metrics["max_regret"], 0)

    def test_evaluation_checks_groups_and_common_features(self):
        rebuilt, _ = self.aggregate(self.raw_frame())
        with self.assertRaises(ValueError):
            evaluate_dataset(rebuilt, folds=3)
        rebuilt.loc[0, "distance_mean_km"] += 1
        with self.assertRaisesRegex(ValueError, "communes"):
            evaluate_dataset(rebuilt, folds=2)


if __name__ == "__main__":
    unittest.main()
