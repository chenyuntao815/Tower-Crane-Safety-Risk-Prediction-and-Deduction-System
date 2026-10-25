# -*- coding: utf-8 -*-
"""
可视化服务和LLM决策服务
"""
import os
import json
import re
import logging
import importlib.util
from typing import Dict, List, Any, Optional
from datetime import datetime
from runtime_config import get_setting

logger = logging.getLogger("safety-infer")


class VisualizationService:
    """知识图谱和风险可视化服务"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_knowledge_graph(self, factor_id: str) -> Dict[str, Any]:
        """获取知识图谱数据 - 因素相关的指标、风险、事故"""
        query = """
        MATCH (f:SafetyFactor {id:$factor_id})
        OPTIONAL MATCH (f)-[r1:HAS_INDICATOR]->(i:Indicator)
        OPTIONAL MATCH (f)-[r2:CAUSES]->(risk:SafetyRisk)
        OPTIONAL MATCH (risk)-[r3:LEADS_TO]->(accident:SafetyAccident)
        RETURN 
            f.id AS factor_id, f.name AS factor_name, f.description AS factor_desc,
            i.id AS indicator_id, i.name AS indicator_name,
            toFloat(r1.weight) AS indicator_weight,
            risk.id AS risk_id, risk.name AS risk_name, risk.risk_level AS risk_level,
            toFloat(r2.trigger_prob) AS risk_trigger_prob,
            accident.id AS accident_id, accident.name AS accident_name,
            toFloat(r3.trigger_prob) AS accident_trigger_prob
        """
        
        with self.driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            results = session.run(query, factor_id=factor_id).data()
        
        if not results:
            return {"nodes": [], "edges": [], "error": f"Factor {factor_id} not found"}
        
        nodes = []
        edges = []
        node_ids = set()
        
        # 颜色映射
        color_map = {
            "factor": "#FF6B6B",      # 红色 - 因素
            "indicator": "#4ECDC4",    # 青色 - 指标
            "risk_high": "#FFB6B9",    # 浅红 - 高风险
            "risk_mid": "#FFE66D",     # 黄色 - 中风险
            "risk_low": "#95E1D3",     # 绿色 - 低风险
            "accident": "#FFD700"      # 金色 - 事故
        }
        
        for record in results:
            # 添加因素节点
            factor_id_n = record['factor_id']
            if factor_id_n and factor_id_n not in node_ids:
                nodes.append({
                    "id": factor_id_n,
                    "label": record['factor_name'],
                    "color": color_map["factor"],
                    "title": record['factor_desc'] or "",
                    "size": 30
                })
                node_ids.add(factor_id_n)
            
            # 添加指标节点和边
            if record['indicator_id']:
                ind_id = record['indicator_id']
                if ind_id not in node_ids:
                    nodes.append({
                        "id": ind_id,
                        "label": record['indicator_name'],
                        "color": color_map["indicator"],
                        "title": f"权重: {record['indicator_weight'] or 0:.2f}",
                        "size": 20
                    })
                    node_ids.add(ind_id)
                
                edges.append({
                    "source": factor_id_n,
                    "target": ind_id,
                    "relation": "关联指标",
                    "weight": record['indicator_weight'] or 1.0,
                    "title": f"权重: {record['indicator_weight'] or 0:.2f}"
                })
            
            # 添加风险节点和边
            if record['risk_id']:
                risk_id = record['risk_id']
                if risk_id not in node_ids:
                    risk_level = record['risk_level'] or "中"
                    if risk_level == "高":
                        color = color_map["risk_high"]
                    elif risk_level == "低":
                        color = color_map["risk_low"]
                    else:
                        color = color_map["risk_mid"]
                    
                    nodes.append({
                        "id": risk_id,
                        "label": record['risk_name'],
                        "color": color,
                        "title": f"风险等级: {risk_level}",
                        "size": 25
                    })
                    node_ids.add(risk_id)
                
                edges.append({
                    "source": factor_id_n,
                    "target": risk_id,
                    "relation": "可能导致",
                    "weight": record['risk_trigger_prob'] or 1.0,
                    "title": f"触发概率: {record['risk_trigger_prob'] or 0:.2%}"
                })
            
            # 添加事故节点和边
            if record['accident_id']:
                acc_id = record['accident_id']
                if acc_id not in node_ids:
                    nodes.append({
                        "id": acc_id,
                        "label": record['accident_name'],
                        "color": color_map["accident"],
                        "title": "可能导致的事故",
                        "size": 22
                    })
                    node_ids.add(acc_id)
                
                edges.append({
                    "source": risk_id,
                    "target": acc_id,
                    "relation": "可能引发",
                    "weight": record['accident_trigger_prob'] or 1.0,
                    "title": f"触发概率: {record['accident_trigger_prob'] or 0:.2%}"
                })
        
        unique_edges = {
            (edge["source"], edge["target"], edge["relation"]): edge
            for edge in edges
        }
        return {"nodes": nodes, "edges": list(unique_edges.values()), "count": len(nodes)}
    
    def get_risk_distribution(self, risk_probs: List[Dict]) -> Dict[str, Any]:
        """获取风险分布统计"""
        if not risk_probs:
            return {"error": "No risk data"}
        
        # 风险等级分类
        high_risk = sum(1 for r in risk_probs if r.get("prob", 0) >= 0.7)
        mid_risk = sum(1 for r in risk_probs if 0.3 <= r.get("prob", 0) < 0.7)
        low_risk = sum(1 for r in risk_probs if r.get("prob", 0) < 0.3)
        
        return {
            "total": len(risk_probs),
            "high_risk": high_risk,
            "mid_risk": mid_risk,
            "low_risk": low_risk,
            "distribution": [
                {"level": "高风险", "count": high_risk, "color": "#FF6B6B"},
                {"level": "中风险", "count": mid_risk, "color": "#FFE66D"},
                {"level": "低风险", "count": low_risk, "color": "#95E1D3"}
            ]
        }
    
    def get_risk_matrix(self, factor_name: str, activation_score: float, 
                       risk_probs: List[Dict]) -> Dict[str, Any]:
        """生成风险矩阵数据"""
        matrix_data = []
        
        for risk in risk_probs[:5]:  # 取概率最高的5个风险
            prob = risk.get("prob", 0)
            
            # 可能性分级（0-1映射到1-5）
            possibility = min(5, int(prob * 5) + 1)
            
            # 严重性由图谱风险等级给出，与可能性分别计分
            severity = {"高": 5, "中": 3, "低": 1}.get(
                risk.get("risk_level"), 3
            )
            
            # 风险等级
            risk_score = possibility * severity
            if risk_score >= 16:
                level = "极高"
                color = "#FF4444"
            elif risk_score >= 12:
                level = "高"
                color = "#FF6B6B"
            elif risk_score >= 8:
                level = "中"
                color = "#FFE66D"
            else:
                level = "低"
                color = "#95E1D3"
            
            matrix_data.append({
                "risk_id": risk.get("risk_id", ""),
                "risk_name": risk.get("risk_name", ""),
                "probability": f"{prob:.1%}",
                "possibility_level": possibility,
                "severity_level": severity,
                "risk_score": risk_score,
                "risk_level": level,
                "color": color
            })
        
        return {"matrix": matrix_data, "total_risks": len(risk_probs)}


class LLMDecisionService:
    """大模型决策服务"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 use_local: Optional[bool] = None):
        self._api_key = api_key
        self._model = model
        self._use_local = use_local

    @property
    def api_key(self):
        return (self._api_key or get_setting("DEEPSEEK_API_KEY", "")).strip()

    @property
    def model(self):
        return (self._model or get_setting("DEEPSEEK_MODEL") or "deepseek-flash").strip() or "deepseek-flash"

    @property
    def use_local(self):
        return self._use_local if self._use_local is not None else get_setting("USE_LOCAL_LLM", "false").strip().lower() == "true"

    def configuration_status(self):
        """Configuration diagnostics only: never return a key or claim connectivity."""
        sdk_installed = importlib.util.find_spec("openai") is not None
        configured = bool(self.api_key)
        if self.use_local:
            status, issue = "local_enabled", "当前启用本地 Ollama；使用 DeepSeek 请设置 USE_LOCAL_LLM=false。"
        elif not configured:
            status, issue = "missing_key", "后端未读取到 DEEPSEEK_API_KEY，请在项目根目录 .env 中填写或在启动后端的终端设置。"
        elif not sdk_installed:
            status, issue = "missing_sdk", "后端 Python 环境未安装 openai SDK。"
        else:
            status, issue = "configured", "配置已就绪；尚未验证联网、密钥有效性及账户额度。"
        return {"provider": "ollama" if self.use_local else "deepseek", "model": self.model,
                "api_key_configured": configured, "sdk_installed": sdk_installed,
                "status": status, "message": issue, "connection_tested": False}
    
    def generate_decision(self, infer_result: Dict) -> Dict[str, Any]:
        """使用LLM生成决策建议"""
        prompt = self._build_prompt(infer_result)
        
        response_text = self._call_ollama(prompt) if self.use_local else self._call_openai(prompt)
        decision = self._parse_response(response_text)
        if "operation_evidence" in infer_result:
            # Numerical advice is rendered from the solver, never authored by the LLM.
            narrative = " ".join([decision["situation_analysis"], *decision["main_warnings"],
                                  *decision["prevention_measures"], decision["emergency_response"], decision["monitoring_focus"]])
            if re.search(r"[0-9%％]|百分之|千分之", narrative):
                raise ValueError("AI 解释含自行生成的数值，请重试；概率与代价以计算面板为准")
        logger.info(
            "LLM call succeeded, provider=%s, model=%s",
            "ollama" if self.use_local else "deepseek",
            self.model,
        )
        return decision
    
    def _build_prompt(self, infer_result: Dict) -> str:
        if "operation_evidence" in infer_result:
            return (
                "你解释塔吊运行操作比较的计算结果。计算器已完成措施筛选，你不得重新选措施、改变风险排序或声称安全放行。"
                "概率为短时模型条件下的联合超限频率，不是事故发生率；图谱只用于风险链解释。"
                "所有数值由界面直接显示，输出只写定性解释，不得写任何数字、百分比、百分之或千分之表述。"
                "只依据下面的结构化证据解释推荐、风险原因、适用条件、作业时间与停机取舍。"
                "如status为budget_unmet或no_feasible_action，明确无法满足预算或操作约束；不得编造传感器数据或试验校准。\n"
                + json.dumps(infer_result["operation_evidence"], ensure_ascii=False)
                + "\n只返回JSON：situation_analysis（字符串）、main_warnings（字符串数组）、prevention_measures（字符串数组）、"
                "emergency_response（字符串）、monitoring_focus（字符串）。"
            )
        if "twin_scenarios" in infer_result:
            scenarios = infer_result["twin_scenarios"]
            return (
                "你是塔式起重机施工安全专家。以下是数字孪生示范参数和简化力学、图谱关系得到的情景推演。"
                "概率未经现场数据校准，不得称为真实事故发生率。请比较措施，说明需现场复核的条件。\n"
                + json.dumps(scenarios, ensure_ascii=False)
                + "\n只返回 JSON，字段必须为 situation_analysis (字符串), main_warnings (字符串数组), "
                "prevention_measures (字符串数组), emergency_response (字符串), monitoring_focus (字符串)。"
            )
        """构建LLM提示词"""
        risk_list = "\n".join([
            f"- {r['risk_name']} (ID: {r['risk_id']}): 风险概率 {r['prob']:.2%}"
            for r in infer_result.get("risk_probs", [])[:5]
        ])
        
        accident_list = "\n".join([
            f"- {a['accident_name']}: 预估概率 {a['prob']:.2%}"
            for a in infer_result.get("accident_probs", [])[:3]
        ])
        indicator_list = "\n".join(
            f"- {i['indicator_name']} ({i['indicator_id']}): {i['value']} {i.get('unit') or ''}"
            for i in infer_result.get("indicator_values", [])
        )
        
        prompt = f"""你是一位塔式起重机施工安全专家。基于以下图谱推演结果，请提供与因素、阶段和指标对应的安全管理建议。概率是图谱权重计算的情景估计，不能当作经过实测校准的事故发生率。

【激活因素】
- 因素名称: {infer_result.get('factor_name', 'N/A')}
- 因素ID: {infer_result.get('factor_id', 'N/A')}
- 激活度: {infer_result.get('activation_score', 0):.2%}
- 施工阶段: {infer_result.get('stage_name', infer_result.get('stage_id', 'N/A'))}

【输入指标】
{indicator_list}

【推演风险清单（概率排序）】
{risk_list}

【可能导致的事故链】
{accident_list}

【动态权重】
当前动态权重: {infer_result.get('dynamic_weight', 0):.3f}

请按照以下JSON格式（严格遵守）返回分析结果，不要有其他文字：

{{
    "situation_analysis": "用2-3句话总结当前安全形势（简洁专业）",
    "main_warnings": [
        "第一个最严重的风险及其潜在后果",
        "第二个重要风险及其潜在后果",
        "第三个需要关注的风险及其潜在后果"
    ],
    "prevention_measures": [
        "立即采取的行动1",
        "立即采取的行动2",
        "立即采取的行动3",
        "短期防控措施1",
        "短期防控措施2"
    ],
    "emergency_response": "如果风险激发，应立即采取的应急措施（1-2句）",
    "monitoring_focus": "重点监测的指标和频率"
}}

只返回JSON，不要有任何其他说明文字。"""
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """按 DeepSeek 官方的 OpenAI SDK 兼容格式调用 Chat Completions。"""
        api_key = self.api_key
        if not api_key:
            raise RuntimeError("后端未配置 DEEPSEEK_API_KEY。请在项目根目录 .env 中填写，或在启动端口 8000 后端的终端设置；仅在前端终端设置不会传给后端。")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai SDK，请执行 python -m pip install -r requirements.txt") from exc

        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
            timeout=120.0,
            max_retries=0,
        )
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是塔式起重机施工安全风险分析专家。请按用户要求输出 JSON。"},
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                reasoning_effort="high",
                extra_body={"thinking": {"type": "enabled"}},
            )
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            messages = {
                400: "DeepSeek 拒绝请求参数（HTTP 400），请核对模型、SDK 版本及输入内容。",
                401: "DeepSeek 身份验证失败（HTTP 401），请检查后端 API Key 是否有效。",
                402: "DeepSeek 账户额度不足（HTTP 402），请检查账户余额。",
                403: "DeepSeek 拒绝访问（HTTP 403），请检查账户权限。",
                404: "DeepSeek 模型或接口不存在（HTTP 404），请检查 DEEPSEEK_MODEL。",
                429: "DeepSeek 请求频率或配额受限（HTTP 429），请稍后重试。",
            }
            if status in messages:
                raise RuntimeError(messages[status]) from None
            if status and status >= 500:
                raise RuntimeError(f"DeepSeek 服务暂时异常（HTTP {status}），请稍后重试。") from None
            if type(exc).__name__ == "APITimeoutError":
                raise RuntimeError("DeepSeek 在 120 秒内未完成响应，请稍后重试。") from None
            if type(exc).__name__ == "APIConnectionError":
                raise RuntimeError("后端无法连接 https://api.deepseek.com，请检查后端运行环境的网络或代理。") from None
            if isinstance(exc, TypeError):
                raise RuntimeError("openai SDK 不支持当前调用参数，请在后端使用的 Python 环境升级 openai SDK。") from None
            raise
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("DeepSeek API 返回了空内容，请检查模型名称、额度及服务状态")
        if getattr(response.choices[0], "finish_reason", None) == "length":
            raise RuntimeError("DeepSeek 输出被截断，未生成完整建议，请缩短输入后重试。")
        return response.choices[0].message.content.strip()
    
    def _call_ollama(self, prompt: str) -> str:
        """调用本地Ollama模型"""
        try:
            import requests
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama2-chinese",  # 或 "llama2"
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.7
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except Exception as e:
            logger.error(f"Ollama调用失败: {str(e)}")
            raise
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """解析LLM响应"""
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start < 0 or json_end <= json_start:
            raise ValueError("DeepSeek 响应不包含 JSON 决策结果")
        try:
            data = json.loads(response_text[json_start:json_end])
        except json.JSONDecodeError as exc:
            raise ValueError("DeepSeek 返回的决策 JSON 无法解析，请重试") from exc
        required = ("situation_analysis", "main_warnings", "prevention_measures",
                    "emergency_response", "monitoring_focus")
        if not isinstance(data, dict) or any(key not in data for key in required):
            raise ValueError("DeepSeek 返回的决策字段不完整，请重试")
        if (not all(isinstance(data[key], str) for key in
                    ("situation_analysis", "emergency_response", "monitoring_focus"))
                or not all(isinstance(data[key], list) and
                           all(isinstance(item, str) for item in data[key])
                           for key in ("main_warnings", "prevention_measures"))):
            raise ValueError("DeepSeek 返回的决策字段类型不正确，请重试")
        return data
    
class HistoryManager:
    """推演历史记录管理"""
    
    def __init__(self, max_records: int = 100):
        self.records: List[Dict] = []
        self.max_records = max_records
    
    def add_record(self, infer_request: Dict, infer_result: Dict, llm_decision: Optional[Dict] = None) -> None:
        """添加推演记录"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "factor_id": infer_request.get("factor_id"),
            "factor_name": infer_result.get("factor_name"),
            "stage_id": infer_request.get("stage_id"),
            "activation_score": infer_result.get("activation_score"),
            "top_risk": infer_result.get("risk_probs", [{}])[0].get("risk_name", "N/A"),
            "llm_decision": llm_decision is not None
        }
        
        self.records.insert(0, record)
        
        # 限制记录数量
        if len(self.records) > self.max_records:
            self.records = self.records[:self.max_records]
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """获取历史记录"""
        return self.records[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        if not self.records:
            return {}
        
        factors = {}
        avg_activation = 0
        high_risk_count = 0
        
        for record in self.records:
            factor = record.get("factor_name", "未知")
            factors[factor] = factors.get(factor, 0) + 1
            avg_activation += record.get("activation_score", 0)
            if record.get("activation_score", 0) >= 0.7:
                high_risk_count += 1
        
        return {
            "total_inferences": len(self.records),
            "avg_activation_score": avg_activation / len(self.records),
            "high_risk_count": high_risk_count,
            "factors_distribution": factors
        }
