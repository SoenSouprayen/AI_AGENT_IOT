import unittest

import pandas as pd

from scripts.analysis.audit_agent_dataset import audit_dataset
from scripts.plan_simulations import make_plan
from src.prediction.geometry_schema import GEOMETRY_FEATURES, MODES


class SimulationPreparationTests(unittest.TestCase):
    def dataset(self):
        rows = []
        for load in (10000, 100000):
            for mode in MODES:
                optimal = "DR8" if load == 10000 else "DR9"
                successes = 90 if mode == optimal else 50
                rows.append({
                    **dict(zip(GEOMETRY_FEATURES, [15, 1000, 7600, 7000, -100, 200, -4, 5])),
                    "N": load, "mode": mode, "environment": "urban",
                    "scenario_seed": 42, "replication": 1, "elevation_bin": 15,
                    "n_packets": 100, "n_success": successes, "per": 1 - successes / 100,
                })
        return pd.DataFrame(rows)

    def test_plan_counts_and_reproducibility(self):
        rows = make_plan([10000, 50000, 100000], velocity_kind="speed_norm")
        self.assertEqual(len(rows), 3240)
        self.assertEqual(rows, make_plan([10000, 50000, 100000], velocity_kind="speed_norm"))
        self.assertEqual(sum(row["split"] == "test" for row in rows), 1200)
        self.assertTrue(all("per" not in row for row in rows))

    def test_plan_keeps_related_rows_together(self):
        frame = pd.DataFrame(make_plan([10000, 100000], velocity_kind="radial_signed"))
        self.assertTrue((frame.groupby("scenario_id")["split"].nunique() == 1).all())
        self.assertTrue((frame.groupby("scenario_id")["scenario_seed"].nunique() == 1).all())
        self.assertTrue((frame.groupby("condition_id")["mode"].nunique() == 8).all())
        self.assertEqual(frame.drop_duplicates("scenario_id")["scenario_seed"].nunique(), 135)

    def test_invalid_plan(self):
        for kwargs in ({"loads": [0]}, {"loads": [True]}, {"loads": [1, 1]},
                       {"loads": [1], "repetitions": 0},
                       {"loads": [1], "holdout_groups": 135},
                       {"loads": [1], "velocity_kind": "unspecified"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(**({"velocity_kind": "speed_norm"} | kwargs))

    def test_audit_rule_and_mode_specific_features(self):
        frame = self.dataset()
        frame.loc[0, "doppler_rate_mean_hz_s"] = -3
        report = audit_dataset(frame)
        self.assertTrue(report["N_only_rule_suffices_on_observed_data"])
        self.assertEqual(report["modes_optimal_for_every_condition_at_N"]["100000"], ["DR9"])
        self.assertEqual(report["geometry_spread_across_modes"]["doppler_rate_mean_hz_s"]["conditions_different"], 1)

    def test_audit_ties_and_no_N_only_winner(self):
        frame = self.dataset()
        second = frame.copy()
        second["replication"] = 2
        second["n_success"] = 50
        second.loc[second["mode"] == "SF7", "n_success"] = 95
        second["per"] = 1 - second["n_success"] / second["n_packets"]
        self.assertFalse(audit_dataset(pd.concat([frame, second]))["N_only_rule_suffices_on_observed_data"])
        frame["n_success"] = 90
        frame["per"] = 0.1
        self.assertEqual(audit_dataset(frame)["conditions_with_optimal_ties"], 2)

    def test_audit_rejects_inconsistent_results(self):
        for column, value in (("per", 10), ("per", 0.5), ("n_packets", 0),
                              ("n_success", 101), ("n_success", float("nan"))):
            frame = self.dataset()
            frame.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                audit_dataset(frame)
        with self.assertRaises(ValueError):
            audit_dataset(self.dataset().iloc[1:])
        with self.assertRaises(ValueError):
            audit_dataset(pd.concat([self.dataset(), self.dataset()]))


if __name__ == "__main__":
    unittest.main()
