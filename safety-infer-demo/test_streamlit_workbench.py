"""Interaction checks for the redesigned workbench, without live services."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest


def response(data):
    return SimpleNamespace(status_code=200, json=lambda: {"code": 0, "data": data}, raise_for_status=lambda: None)


def graph_response(url, **kwargs):
    if url.endswith("/factors"):
        return response([{"id": "F028", "name": "不利天气"}])
    if url.endswith("/stages"):
        return response([{"id": "S001", "name": "正常吊装"}])
    if url.endswith("/indicators"):
        return response([{"id": "I016", "name": "风速", "unit": "m/s", "data_type": "float",
                          "normal_threshold": 8, "warning_threshold": 12, "danger_threshold": 20,
                          "direction": "positive"}])
    return response([])


INFERENCE = {"factor_id": "F028", "factor_name": "不利天气", "stage_id": "S001", "stage_name": "正常吊装",
             "activation_score": .4, "dynamic_weight": .5,
             "risk_probs": [{"risk_id": "R001", "risk_name": "吊物失稳", "prob": .3, "risk_level": "高"}],
             "accident_probs": [{"accident_id": "A001", "accident_name": "物体打击", "prob": .1}]}


class WorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.get_patch = patch("requests.get", side_effect=graph_response)
        self.get_patch.start()
        self.addCleanup(self.get_patch.stop)

    def app(self):
        app = AppTest.from_file("streamlit_app.py", default_timeout=15).run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_initial_workspace_focuses_on_risk_inference(self):
        app = self.app()
        self.assertEqual(app.radio(key="workspace").value, "安全风险推演")
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertEqual(app.selectbox(key="selected_factor").value, "不利天气")
        self.assertFalse(app.button(key="run_inference").disabled)

    def test_history_prediction_remains_accessible(self):
        app = self.app()
        app.number_input(key="indicator_I016").set_value(14.5).run()
        app.radio(key="workspace").set_value("历史数据预测").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.get("file_uploader")), 1)
        self.assertEqual(app.number_input(key="prediction_horizon").value, 1)
        app.radio(key="workspace").set_value("安全风险推演").run()
        self.assertEqual(len(app.exception), 0)
        self.assertFalse(app.button(key="run_inference").disabled)
        self.assertEqual(app.number_input(key="indicator_I016").value, 14.5)

    def test_changed_observation_reaches_api_and_results_can_be_reset(self):
        app = self.app()
        app.number_input(key="indicator_I016").set_value(14.5).run()
        with patch("requests.post", return_value=response(INFERENCE)) as post:
            app.button(key="run_inference").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(post.call_args.kwargs["json"]["indicators"], [{"indicator_id": "I016", "value": 14.5}])
        self.assertEqual(len(app.tabs), 5)
        self.assertGreaterEqual(len(app.metric), 4)
        app.button(key="reset_inference").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state.last_result)
        self.assertEqual(app.number_input(key="indicator_I016").value, 14.5)

    def test_unavailable_graph_keeps_workbench_and_prediction_usable(self):
        with patch("requests.get", side_effect=requests.exceptions.ConnectionError("offline")):
            app = self.app()
            self.assertTrue(app.button(key="run_inference").disabled)
            self.assertEqual(len(app.error), 0)
            app.radio(key="workspace").set_value("历史数据预测").run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.get("file_uploader")), 1)

    def test_ai_result_is_rendered_immediately_and_survives_reruns(self):
        app = self.app()
        with patch('requests.post', return_value=response(INFERENCE)):
            app.button(key='run_inference').click().run()
        decision = {'situation_analysis':'根据计算结果分析', 'main_warnings':[],
                    'prevention_measures':['平缓操作'], 'emergency_response':'按预案处置', 'monitoring_focus':'摆角'}
        with patch('requests.post', return_value=response(decision)) as post:
            app.button(key='generate_decision').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(post.call_args.args[0].endswith('/infer/llm-decision'))
        self.assertEqual(app.session_state.last_decision, decision)
        self.assertTrue(any(item.value == decision['situation_analysis'] for item in app.info))
        app.run()
        self.assertEqual(app.session_state.last_decision, decision)

    def test_ai_error_remains_visible_after_reruns_and_can_be_retried(self):
        app = self.app()
        with patch('requests.post', return_value=response(INFERENCE)):
            app.button(key='run_inference').click().run()
        failed = SimpleNamespace(status_code=200, json=lambda: {'code':1, 'message':'后端未配置 DEEPSEEK_API_KEY'})
        with patch('requests.post', return_value=failed):
            app.button(key='generate_decision').click().run()
        app.run()
        self.assertTrue(any('未配置' in item.value for item in app.error))
        decision = {'situation_analysis':'重试成功', 'main_warnings':[], 'prevention_measures':[],
                    'emergency_response':'按预案处置', 'monitoring_focus':'摆角'}
        with patch('requests.post', return_value=response(decision)):
            app.button(key='generate_decision').click().run()
        self.assertIsNone(app.session_state.last_decision_error)
        self.assertEqual(app.session_state.last_decision, decision)

    def test_config_diagnostics_use_backend_endpoint_without_calling_model(self):
        app = self.app()
        with patch('requests.post', return_value=response(INFERENCE)):
            app.button(key='run_inference').click().run()
        status = {'status':'missing_key', 'message':'后端未配置密钥', 'model':'deepseek-flash', 'connection_tested':False}
        def get(url, **kwargs):
            return response(status) if url.endswith('/infer/llm-status') else graph_response(url, **kwargs)
        with patch('requests.get', side_effect=get), patch('requests.post') as post:
            app.button(key='check_llm_config').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state.llm_configuration, status)
        self.assertTrue(any('未配置密钥' in item.value for item in app.warning))
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
