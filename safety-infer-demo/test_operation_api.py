"""API integration tests use the bundled graph snapshot, without external services."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from app import app
from digital_twin_service import bundled_graph_edges
from inference_service import LLMDecisionService


class OperationApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client=TestClient(app)

    def test_defaults_simulation_and_assets(self):
        response=self.client.get('/api/v1/operations/defaults')
        self.assertEqual(response.status_code,200)
        config=response.json()['data']
        config.update(samples=64,horizon_s=4)
        with patch('app.operation_graph',return_value=(bundled_graph_edges(),'test CSV')):
            response=self.client.post('/api/v1/operations/simulate',json=config)
        self.assertEqual(response.status_code,200)
        data=response.json()['data']
        self.assertEqual(len(data['scenarios']),4)
        self.assertEqual(data['graph_source'],'test CSV')
        for path in ['/twin/','/twin/operations.html','/twin/operations.mjs','/twin/operation_scene.mjs','/twin/operations.css','/twin/legacy.html','/twin/vendor/package/build/three.module.mjs']:
            self.assertEqual(self.client.get(path).status_code,200,path)

    def test_validation_errors_and_valid_audit(self):
        self.assertEqual(self.client.post('/api/v1/operations/simulate',json={'samples':4}).status_code,422)
        bad=self.client.post('/api/v1/operations/validate',json={'rows':[{'risk_probability':1.1,'observed_exceedance':1}]})
        self.assertEqual(bad.status_code,422)
        good=self.client.post('/api/v1/operations/validate',json={'rows':[{'risk_probability':.2,'observed_exceedance':0}]})
        self.assertEqual(good.status_code,200)
        self.assertEqual(good.json()['data']['brier_score'],.04)

    def test_advice_recomputes_and_preserves_authoritative_result(self):
        decision={'situation_analysis':'已有摆动','main_warnings':['急刹会激发摆动'],'prevention_measures':['核对约束'], 'emergency_response':'按应急预案处置','monitoring_focus':'摆角与风速'}
        with patch('app.operation_graph',return_value=([],'test')),patch('app.llm_service.generate_decision',return_value=decision) as llm:
            response=self.client.post('/api/v1/operations/advice',json={'samples':64,'horizon_s':4})
        self.assertEqual(response.status_code,200)
        self.assertEqual(response.json()['data']['explanation'],decision)
        self.assertIn('recommendation',response.json()['data'])
        evidence=llm.call_args.args[0]['operation_evidence']
        self.assertEqual(len(evidence['measures']),4)
        self.assertIn('probability_definition',evidence)

    def test_client_cannot_supply_probabilities_to_new_advice_endpoint(self):
        response=self.client.post('/api/v1/operations/advice',json={'risk_probability':.0001})
        self.assertEqual(response.status_code,422)

    def test_configuration_endpoint_returns_safe_status_without_generating_advice(self):
        service=LLMDecisionService(api_key='offline-private-key',use_local=False)
        with patch('app.llm_service',service),patch.object(service,'generate_decision') as generate:
            response=self.client.get('/api/v1/infer/llm-status')
        self.assertEqual(response.status_code,200)
        data=response.json()['data']
        self.assertEqual(data['status'],'configured')
        self.assertFalse(data['connection_tested'])
        self.assertNotIn('offline-private-key',response.text)
        generate.assert_not_called()

    def test_missing_key_is_reported_by_both_advice_endpoints_without_fake_output(self):
        service=LLMDecisionService(use_local=False)
        with patch('app.llm_service',service),patch('inference_service.get_setting',side_effect=lambda name,default=None:default),patch('app.operation_graph',return_value=([],'test')):
            for path,payload in [('/api/v1/operations/advice',{'samples':64,'horizon_s':4}),
                                 ('/api/v1/infer/llm-decision',{'infer_result':{}})]:
                with self.subTest(path=path):
                    response=self.client.post(path,json=payload)
                    self.assertEqual(response.status_code,200)
                    body=response.json()
                    self.assertEqual(body['code'],1)
                    self.assertIn('DEEPSEEK_API_KEY',body['message'])
                    self.assertIsNone(body['data'])


if __name__=='__main__':unittest.main()
