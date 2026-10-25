"""塔吊图谱阈值与关联数据的离线回归检查。"""

import unittest
import csv
from pathlib import Path

from risk_scoring import score_by_indicator_name


class TowerGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parent / "Knowledge Graph"

        def read(name):
            with (root / name).open(encoding="utf-8-sig", newline="") as stream:
                return list(csv.DictReader(stream))

        cls.nodes = {
            "SafetyFactor": read("SafetyFactor.csv"),
            "Indicator": read("Indicator.csv"),
        }
        cls.relations = {
            "HAS_INDICATOR": read("HAS_INDICATOR.csv"),
            "ACTIVE_IN": read("ACTIVE_IN.csv"),
        }

    def test_every_indicator_relation_uses_its_own_thresholds(self):
        names = {row["id"]: row["name"] for row in self.nodes["Indicator"]}
        for row in self.relations["HAS_INDICATOR"]:
            with self.subTest(factor=row["start_id"], indicator=row["end_id"]):
                def score(value):
                    return score_by_indicator_name(
                        names[row["end_id"]], value,
                        float(row["normal_threshold"]),
                        float(row["warning_threshold"]),
                        float(row["danger_threshold"]), row["direction"]
                    )

                self.assertAlmostEqual(score(float(row["normal_threshold"])), 0)
                self.assertAlmostEqual(score(float(row["warning_threshold"])), 0.6)
                self.assertAlmostEqual(score(float(row["danger_threshold"])), 1)

    def test_first_factor_has_tower_crane_relations(self):
        factor = next(row for row in self.nodes["SafetyFactor"] if row["id"] == "F001")
        self.assertIn("设备", factor["name"])
        self.assertIn(
            ("F001", "S001"),
            {(row["start_id"], row["end_id"]) for row in self.relations["ACTIVE_IN"]},
        )
        self.assertEqual(
            {"I001", "I038"},
            {row["end_id"] for row in self.relations["HAS_INDICATOR"]
             if row["start_id"] == "F001"},
        )


if __name__ == "__main__":
    unittest.main()
