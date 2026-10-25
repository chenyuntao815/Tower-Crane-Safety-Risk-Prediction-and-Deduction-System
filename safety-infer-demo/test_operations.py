"""Physics, common-noise comparison, constraints and validation regressions."""
import copy
import json
import math
import unittest
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from operation_models import OperationRequest
from operation_service import (simulate_operations, control_profile, validate_predictions,
                               _ensemble, _rollout, _derivative, wilson)
from digital_twin_service import bundled_graph_edges


class OperationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.req = OperationRequest(samples=64, horizon_s=8)
        cls.result = simulate_operations(cls.req, bundled_graph_edges(), "test")

    def test_shared_initial_state_wind_and_ensemble(self):
        cases = self.result["scenarios"]
        self.assertEqual(len({s["ensemble_id"] for s in cases}), 1)
        for case in cases[1:]:
            self.assertEqual(case["series"][0], cases[0]["series"][0])
            self.assertEqual([x["wind_m_s"] for x in case["series"]], [x["wind_m_s"] for x in cases[0]["series"]])
        p1, x1, w1, id1, _ = _ensemble(self.req)
        p2, x2, w2, id2, _ = _ensemble(self.req)
        self.assertEqual(id1, id2)
        np.testing.assert_array_equal(x1, x2)
        np.testing.assert_array_equal(w1, w2)
        changed = self.req.model_copy(update={"seed": self.req.seed+1})
        self.assertNotEqual(id1, _ensemble(changed)[3])

    def test_time_window_events_monotone_and_joint_not_independence(self):
        for case in self.result["scenarios"]:
            values = [p["cumulative_probability"] for p in case["series"]]
            self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))
            self.assertEqual(case["risk"]["probability"], values[-1])
            for point in case["series"]:
                self.assertGreaterEqual(point["cumulative_probability"], point["instantaneous_probability"])
                self.assertTrue(all(point["cumulative_probability"] >= p for p in point["event_probabilities"].values()))
            self.assertEqual(case["risk"]["hits"]/64, case["risk"]["probability"])
        json.dumps(self.result, allow_nan=False)

    def test_acceleration_profiles_and_no_direction_reversal(self):
        for action in ["early_deceleration", "smooth_brake", "immediate_brake"]:
            for sign in (-1, 1):
                for t in np.linspace(0, 8, 51):
                    w, a, acc = control_profile(action, t, np.array([sign*.05]), self.req.controls)
                    self.assertGreaterEqual(sign*w[0], 0)
                    if 0 < t < .7:
                        h=1e-5
                        plus=control_profile(action, t+h, np.array([sign*.05]), self.req.controls)
                        minus=control_profile(action, t-h, np.array([sign*.05]), self.req.controls)
                        self.assertAlmostEqual((plus[1][0]-minus[1][0])/(2*h), w[0], places=6)
                        self.assertAlmostEqual((plus[0][0]-minus[0][0])/(2*h), acc[0], places=6)

    def test_quiet_model_recovers_analytic_pendulum(self):
        req = OperationRequest(samples=64, horizon_s=4, state={"wind_m_s":0,"slew_rate_deg_s":0,"swing_x_deg":2,"swing_z_deg":0}, controls={"remaining_slew_deg":0})
        p, state, wind, _, _ = _ensemble(req)
        p["length"][:] = 18
        p["omega"][:] = 0
        p["omega"][:] = 0
        p["swing_damping"][:] = 0
        state[:, :4] = [math.radians(2), 0, 0, 0]
        # Remove mast coupling using a rigid equivalent mast and very small mass.
        p["ei"][:] = 1e25
        state[:, 4:] = 0
        omega = math.sqrt(9.81/18)
        h = .01
        for k in range(400):
            t=k*h
            k1=_derivative(req,p,state,t,"maintain",np.zeros(64))
            k2=_derivative(req,p,state+h*k1/2,t+h/2,"maintain",np.zeros(64))
            k3=_derivative(req,p,state+h*k2/2,t+h/2,"maintain",np.zeros(64))
            k4=_derivative(req,p,state+h*k3,t+h,"maintain",np.zeros(64))
            state += h*(k1+2*k2+2*k3+k4)/6
        self.assertAlmostEqual(state[0,0], math.radians(2)*math.cos(omega*4), places=7)
        self.assertAlmostEqual(state[0,1], 0, places=7)

    def test_abrupt_brake_is_not_assumed_safer(self):
        baseline=self.result["scenarios"][0]
        hard=self.result["scenarios"][3]
        self.assertGreater(hard["peaks"]["swing_deg"]["p95"], baseline["peaks"]["swing_deg"]["p95"])

    def test_initial_exceedance_shared_and_no_feasible_action(self):
        req=OperationRequest(samples=64,horizon_s=4,state={"load_t":15},limits={"swing_deg":.01})
        value=simulate_operations(req)
        self.assertEqual(value["recommendation"]["status"], "no_feasible_action")
        for s in value["scenarios"]:
            self.assertFalse(s["constraints"]["feasible"])
            self.assertEqual(s["risk"]["probability"],1)

    def test_zero_hits_have_nonzero_upper_bound_and_budget_gate(self):
        self.assertGreater(wilson(0,64)[1],0)
        for s in self.result["scenarios"]:
            self.assertLessEqual(s["risk"]["ci95"][0],s["risk"]["probability"])
            self.assertGreaterEqual(s["risk"]["ci95"][1],s["risk"]["probability"])
        if self.result["recommendation"]["status"]=="within_budget":
            s=next(s for s in self.result["scenarios"] if s["id"]==self.result["recommendation"]["action_id"])
            self.assertLessEqual(s["risk"]["ci95"][1],self.req.controls.risk_budget)

    def test_disabled_obstacle_and_missing_attachment(self):
        req=OperationRequest(samples=64,horizon_s=4,equipment={"attachment_heights_m":[],"max_free_height_m":60})
        value=simulate_operations(req)
        for s in value["scenarios"]:
            self.assertEqual(s["risk"]["events"]["clearance"]["probability"],0)
            self.assertEqual(s["risk"]["events"]["attachment"]["probability"],0)
            self.assertEqual(s["series"][0]["attachment_axial_kn"]["mean"],0)

    def test_displacement_measurement_reduces_posterior_uncertainty(self):
        measured=OperationRequest(state={"mast_x_mm":30,"mast_z_mm":0})
        prior=_ensemble(OperationRequest())[-1]
        post=_ensemble(measured)[-1]
        self.assertLess(post["mast_posterior_std_mm"][0],prior["mast_posterior_std_mm"][0])

    def test_invalid_requests_are_rejected(self):
        for change in [{"state":{"radius_m":60}},{"state":{"wind_m_s":float('nan')}},{"equipment":{"attachment_heights_m":[36,18]}},{"samples":100000},
                       {"state":{"slew_rate_deg_s":-3}},{"horizon_s":100},{"limits":{"swing_source":""}}]:
            with self.assertRaises(ValidationError):OperationRequest(**change)

    def test_risk_link_weights_cannot_change_solver_probability(self):
        req=OperationRequest(samples=64,horizon_s=4)
        a=simulate_operations(req,[])
        b=simulate_operations(req,[{"factor_id":"F028","risk_id":"x","risk_name":"test","trigger_prob":1}])
        self.assertEqual([s["risk"] for s in a["scenarios"]],[s["risk"] for s in b["scenarios"]])

    def test_obstacle_reduction_uses_shared_samples_and_meets_budget(self):
        req=OperationRequest(samples=128,obstacle={"enabled":True,"x_m":10,"z_m":33,"radius_m":2},limits={"attachment_axial_kn":280})
        value=simulate_operations(req)
        baseline,early=value["scenarios"][:2]
        self.assertGreater(baseline["risk"]["events"]["clearance"]["probability"],early["risk"]["events"]["clearance"]["probability"])
        self.assertGreater(early["comparison"]["risk_reduction_pp"],50)
        self.assertEqual(value["recommendation"]["action_id"],"early_deceleration")
        self.assertEqual(value["recommendation"]["status"],"within_budget")

    def test_rk4_step_refinement_preserves_predictions(self):
        req=OperationRequest(samples=64,horizon_s=4)
        p,x,w,fp,_=_ensemble(req)
        ordinary,_=_rollout(req,p,x,w,"smooth_brake","test",fp)
        with patch('operation_service.DT',.01),patch('operation_service.OUTPUT_STRIDE',20):
            refined,_=_rollout(req,p,x,w,"smooth_brake","test",fp)
        a,b=ordinary['series'],refined['series']
        self.assertEqual(len(a),len(b))
        self.assertLess(max(abs(x['swing_deg']['mean']-y['swing_deg']['mean']) for x,y in zip(a,b)),.001)
        self.assertLess(max(abs(x['mast_moment_kn_m']['mean']-y['mast_moment_kn_m']['mean']) for x,y in zip(a,b)),.1)


class ValidationTests(unittest.TestCase):
    def test_metrics_and_endpoint_bins(self):
        rows=[{"risk_probability":p,"observed_exceedance":y,"pred_swing_deg":3,"observed_swing_deg":2,"p05_swing_deg":1,"p95_swing_deg":4} for p,y in [(0,0),(.5,1),(1,1)]]
        report=validate_predictions(rows)
        self.assertAlmostEqual(report["brier_score"],1/12,places=6)
        self.assertEqual(sum(b["count"] for b in report["reliability_bins"]),3)
        self.assertEqual(report["prediction_error"]["swing_deg"]["mae"],1)
        self.assertEqual(report["prediction_error"]["swing_deg"]["p05_p95_coverage"],1)

    def test_reject_invalid_or_incomplete_trials(self):
        for row in [{"risk_probability":1.1,"observed_exceedance":1},{"risk_probability":.1,"observed_exceedance":2},{"risk_probability":.2},
                    {"pred_swing_deg":2},{"other":1},{"pred_swing_deg":float('nan'),"observed_swing_deg":1}]:
            with self.assertRaises(ValueError):validate_predictions([row])


if __name__=="__main__":unittest.main()
