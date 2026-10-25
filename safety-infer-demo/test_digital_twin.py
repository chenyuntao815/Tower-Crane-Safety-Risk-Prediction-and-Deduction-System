import unittest

from digital_twin_service import bundled_graph_edges, simulate_twin


BASE = {
    "load_t": 5, "radius_m": 35, "wind_m_s": 8, "slew_deg": 35,
    "event": "normal", "tower_height_m": 48, "jib_length_m": 55,
    "attachment_heights_m": [18, 36], "mast_ei_kn_m2": 2_500_000,
    "rated_moment_kn_m": 2400, "mast_moment_capacity_kn_m": 3000,
    "attachment_capacity_kn": 180, "attachment_lever_arm_m": 5,
    "rated_load_t": 12, "max_free_height_m": 25,
    "projected_area_m2": 85, "drag_coefficient": 1.3,
}


class DigitalTwinTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.edges = bundled_graph_edges()

    def run_scenario(self, **changes):
        return simulate_twin({**BASE, **changes}, self.edges, "test")

    def test_measures_reduce_structural_response_and_risk(self):
        cases = self.run_scenario()["scenarios"]
        baseline, lower_load, shorter_radius = (case["future"] for case in cases)
        for improved in (lower_load, shorter_radius):
            self.assertLess(improved["displacement_mm"], baseline["displacement_mm"])
            self.assertLess(improved["mast_moment_kn_m"], baseline["mast_moment_kn_m"])
            self.assertLessEqual(improved["risk_probability"], baseline["risk_probability"])

    def test_installing_attachment_reduces_free_height_and_displacement(self):
        scenario = self.run_scenario(event="attach")["scenarios"][0]
        current, future = scenario["series"][0], scenario["future"]
        self.assertGreater(len(future["attachments_m"]), len(current["attachments_m"]))
        self.assertLess(future["free_height_m"], current["free_height_m"])
        self.assertLess(future["displacement_mm"], current["displacement_mm"])

    def test_jacking_and_dismantling_change_geometry(self):
        jack = self.run_scenario(event="jacking")["scenarios"][0]
        dismantle = self.run_scenario(event="dismantle")["scenarios"][0]
        self.assertGreater(jack["future"]["tower_height_m"], jack["series"][0]["tower_height_m"])
        self.assertLess(dismantle["future"]["tower_height_m"], dismantle["series"][0]["tower_height_m"])
        self.assertLess(len(dismantle["future"]["attachments_m"]), len(dismantle["series"][0]["attachments_m"]))

    def test_probabilities_and_time_samples(self):
        result = self.run_scenario(wind_m_s=20)
        self.assertEqual([s["time_s"] for s in result["scenarios"][0]["series"]], list(range(0, 61, 5)))
        for scenario in result["scenarios"]:
            for point in scenario["series"]:
                self.assertGreaterEqual(point["risk_probability"], 0)
                self.assertLessEqual(point["risk_probability"], 1)
                self.assertTrue(all(0 <= risk["prob"] <= 1 for risk in point["risks"]))

    def test_graph_link_strength_affects_bayesian_result(self):
        ordinary = self.run_scenario()["scenarios"][0]["future"]["risk_probability"]
        no_links = simulate_twin(BASE, [], "test")["scenarios"][0]["future"]["risk_probability"]
        self.assertGreater(ordinary, no_links)


if __name__ == "__main__":
    unittest.main()
