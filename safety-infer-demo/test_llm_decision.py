"""DeepSeek 决策调用的离线回归测试，不访问网络。"""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from inference_service import LLMDecisionService
from runtime_config import get_setting


class DeepSeekDecisionTests(unittest.TestCase):
    def setUp(self):
        # Never use a developer's real configuration or credentials in tests.
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.env_path = Path(directory.name) / '.env'
        env_file = patch('runtime_config.ENV_PATH', self.env_path)
        env_file.start()
        self.addCleanup(env_file.stop)
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        for name in ('DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL', 'USE_LOCAL_LLM'):
            os.environ.pop(name, None)

    def make_sdk(self, content=None, error=None):
        calls = {}

        def create(**kwargs):
            calls["request"] = kwargs
            if error:
                raise error
            message = types.SimpleNamespace(content=content)
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=message)]
            )

        def openai_client(**kwargs):
            calls["client"] = kwargs
            return types.SimpleNamespace(
                chat=types.SimpleNamespace(
                    completions=types.SimpleNamespace(create=create)
                )
            )

        module = types.ModuleType("openai")
        module.OpenAI = openai_client
        return module, calls

    def test_official_chat_completion_arguments_and_result(self):
        decision = {
            "situation_analysis": "已分析",
            "main_warnings": [],
            "prevention_measures": [],
            "emergency_response": "停机",
            "monitoring_focus": "检查塔机",
        }
        sdk, calls = self.make_sdk(content=json.dumps(decision))
        with patch.dict(sys.modules, {"openai": sdk}):
            actual = LLMDecisionService(api_key="test-key").generate_decision({})
        self.assertEqual(actual, decision)
        self.assertEqual(calls["client"]["base_url"], "https://api.deepseek.com")
        self.assertEqual(calls["request"]["model"], "deepseek-flash")
        self.assertEqual(calls["request"]["reasoning_effort"], "high")
        self.assertEqual(
            calls["request"]["extra_body"], {"thinking": {"type": "enabled"}}
        )
        self.assertIs(calls["request"]["stream"], False)
        self.assertNotIn("temperature", calls["request"])
        self.assertNotIn("max_tokens", calls["request"])

    def test_api_error_is_not_replaced_with_preset_decision(self):
        sdk, _ = self.make_sdk(error=RuntimeError("authentication failed"))
        with patch.dict(sys.modules, {"openai": sdk}):
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                LLMDecisionService(api_key="test-key").generate_decision({})

    def test_invalid_json_is_reported(self):
        sdk, _ = self.make_sdk(content="无法生成 JSON")
        with patch.dict(sys.modules, {"openai": sdk}):
            with self.assertRaisesRegex(ValueError, "JSON"):
                LLMDecisionService(api_key="test-key").generate_decision({})

    def test_operation_explanation_cannot_author_probabilities(self):
        decision = {"situation_analysis":"风险概率1%", "main_warnings":[], "prevention_measures":[],
                    "emergency_response":"检查", "monitoring_focus":"摆角"}
        sdk, _ = self.make_sdk(content=json.dumps(decision))
        with patch.dict(sys.modules, {"openai": sdk}):
            with self.assertRaisesRegex(ValueError, "数值"):
                LLMDecisionService(api_key="test-key").generate_decision({"operation_evidence":{}})

    def test_project_env_is_read_on_each_call_with_process_environment_priority(self):
        service = LLMDecisionService()
        self.assertEqual(service.api_key, '')
        self.env_path.write_text('DEEPSEEK_API_KEY=offline-key\nDEEPSEEK_MODEL=deepseek-flash\nUSE_LOCAL_LLM=false\n', encoding='utf-8-sig')
        self.assertEqual(service.api_key, 'offline-key')
        self.assertEqual(service.model, 'deepseek-flash')
        self.assertFalse(service.use_local)
        self.env_path.write_text('DEEPSEEK_API_KEY=replaced-offline-key\n', encoding='utf-8')
        self.assertEqual(service.api_key, 'replaced-offline-key')
        os.environ['DEEPSEEK_API_KEY'] = 'process-key'
        self.assertEqual(service.api_key, 'process-key')
        os.environ['DEEPSEEK_API_KEY'] = ''
        self.assertEqual(service.api_key, '')
        self.assertEqual(get_setting('UNSET_TEST_SETTING', 'fallback'), 'fallback')

    def test_configuration_check_does_not_contact_provider_or_return_key(self):
        with patch('inference_service.importlib.util.find_spec', return_value=object()), patch.object(LLMDecisionService, '_call_openai') as call:
            missing = LLMDecisionService().configuration_status()
            status = LLMDecisionService(api_key='offline-secret').configuration_status()
        self.assertEqual(missing['status'], 'missing_key')
        self.assertEqual(status['status'], 'configured')
        self.assertFalse(status['connection_tested'])
        self.assertNotIn('offline-secret', json.dumps(status))
        call.assert_not_called()

    def test_provider_http_errors_are_actionable_and_do_not_expose_raw_body(self):
        for code, expected in ((401, '身份验证失败'), (402, '额度不足'), (429, '受限'), (503, '服务暂时异常')):
            with self.subTest(code=code):
                error = RuntimeError('private-provider-body')
                error.status_code = code
                sdk, _ = self.make_sdk(error=error)
                with patch.dict(sys.modules, {'openai': sdk}):
                    with self.assertRaisesRegex(RuntimeError, expected) as raised:
                        LLMDecisionService(api_key='offline-key').generate_decision({})
                self.assertNotIn('private-provider-body', str(raised.exception))

    def test_real_sdk_serializes_official_request_using_offline_transport(self):
        import httpx
        import openai
        actual_client = openai.OpenAI
        captured = []
        decision = {'situation_analysis':'依据计算结果分析', 'main_warnings':[], 'prevention_measures':[],
                    'emergency_response':'按预案处置', 'monitoring_focus':'摆角'}

        def handler(request):
            captured.append(request)
            return httpx.Response(200, json={
                'id':'offline-completion', 'object':'chat.completion', 'created':0, 'model':'deepseek-flash',
                'choices':[{'index':0, 'finish_reason':'stop', 'message':{'role':'assistant', 'content':json.dumps(decision)}}]
            })

        def make_client(**kwargs):
            return actual_client(**kwargs, http_client=httpx.Client(transport=httpx.MockTransport(handler)))

        with patch.object(openai, 'OpenAI', side_effect=make_client):
            output = LLMDecisionService(api_key='offline-key').generate_decision({'operation_evidence':{}})
        self.assertEqual(output, decision)
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertEqual(str(request.url), 'https://api.deepseek.com/chat/completions')
        body = json.loads(request.content)
        self.assertEqual(body['model'], 'deepseek-flash')
        self.assertEqual(body['reasoning_effort'], 'high')
        self.assertEqual(body['thinking'], {'type':'enabled'})
        self.assertFalse(body['stream'])


if __name__ == "__main__":
    unittest.main()
