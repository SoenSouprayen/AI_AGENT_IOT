from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from scripts.ML.train_geometry_per_model import train_model
from src.decision.modulation_selector import select_best_modulation
from src.prediction.geometry_schema import FEATURES, GEOMETRY_FEATURES, MODES, make_frame
from src.prediction.per_predictor import PERPredictor


class GeometryAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        cls.model_path = Path(cls.directory.name) / "geometry.joblib"
        cls.geometry = dict(zip(GEOMETRY_FEATURES, [45, 800, 7600, 7000, -100, -200, -4, 5]))
        rows = []
        for N in (10000, 50000, 100000):
            for index, mode in enumerate(MODES):
                row = make_frame(cls.geometry, (mode,), N).iloc[0].to_dict()
                row["per"] = 0.1 + index / 10
                rows.append(row)
        cls.dataset_path = Path(cls.directory.name) / "dataset.csv"
        pd.DataFrame(rows).to_csv(cls.dataset_path, index=False)
        train_model(cls.dataset_path, cls.model_path)
        cls.predictor = PERPredictor(cls.model_path)

    def test_round_trip_and_selection(self):
        result = select_best_modulation(self.geometry, 10000, self.predictor)
        self.assertEqual(tuple(result["per_by_mode"]), MODES)
        self.assertEqual(result["predicted_per"], min(result["per_by_mode"].values()))
        self.assertEqual(result["best_mode"], min(MODES, key=result["per_by_mode"].get))
        self.assertTrue(all(0 <= value <= 1 for value in result["per_by_mode"].values()))
        for mode in MODES:
            self.assertAlmostEqual(
                self.predictor.predict_per(self.geometry, mode, 10000),
                result["per_by_mode"][mode],
            )
        self.assertEqual(list(self.predictor.pipeline.feature_names_in_), list(FEATURES))

    def test_invalid_N(self):
        for value in (0, -1, 1.5, True, "10000", np.nan, np.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.predictor.predict_per(self.geometry, "DR8", value)

    def test_geometry_and_mode_validation(self):
        missing = self.geometry.copy()
        missing.pop("distance_mean_km")
        with self.assertRaises(ValueError):
            self.predictor.predict_per(missing, "DR8", 10000)
        for feature, value in (
            ("elevation_mean_deg", 91), ("distance_mean_km", 0),
            ("orbital_velocity_mean_ms", -1), ("relative_velocity_mean_ms", -1),
            ("doppler_peak_rate_mean_hz_s", -1), ("doppler_shift_mean_hz", np.inf),
            ("doppler_rate_mean_hz_s", True),
        ):
            with self.subTest(feature=feature), self.assertRaises(ValueError):
                self.predictor.predict_per({**self.geometry, feature: value}, "DR8", 10000)
        with self.assertRaises(ValueError):
            self.predictor.predict_per(self.geometry, "DR10", 10000)

    def test_out_of_domain(self):
        with self.assertRaisesRegex(ValueError, "hors domaine"):
            self.predictor.predict_per(self.geometry, "DR8", 25000)
        with self.assertRaisesRegex(ValueError, "hors domaine"):
            self.predictor.predict_per({**self.geometry, "elevation_mean_deg": 40}, "DR8", 10000)
        predictor = PERPredictor(self.model_path, allow_out_of_domain=True)
        with self.assertWarns(UserWarning):
            predictor.predict_per(self.geometry, "DR8", 25000)

    def test_incompatible_pipeline_and_no_overwrite(self):
        incompatible = Path(self.directory.name) / "C.joblib"
        joblib.dump(Pipeline([]), incompatible)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            PERPredictor(incompatible)
        with self.assertRaises(FileExistsError):
            train_model(self.dataset_path, self.model_path)
        for name in ("best_per_predictor.joblib", "per_predictor.joblib"):
            with self.assertRaises(ValueError):
                train_model(self.dataset_path, Path(self.directory.name) / name)

    def test_bad_target_and_missing_mode(self):
        dataset = pd.read_csv(self.dataset_path)
        bad_path = Path(self.directory.name) / "bad.csv"
        dataset.loc[0, "per"] = 2
        dataset.to_csv(bad_path, index=False)
        with self.assertRaisesRegex(ValueError, "per"):
            train_model(bad_path, Path(self.directory.name) / "bad.joblib")
        dataset = pd.read_csv(self.dataset_path)
        dataset[dataset["mode"] != "SF12"].to_csv(bad_path, index=False)
        with self.assertRaisesRegex(ValueError, "huit modes"):
            train_model(bad_path, Path(self.directory.name) / "bad.joblib")

    def test_tie_breaking(self):
        class EqualPredictor:
            def predict_modes(self, geometry, N, modes):
                return {mode: 0.2 for mode in modes}

        result = select_best_modulation(self.geometry, 10000, EqualPredictor())
        self.assertEqual(result["best_mode"], "DR8")
        self.assertEqual(result["tied_modes"], list(MODES))


if __name__ == "__main__":
    unittest.main()
