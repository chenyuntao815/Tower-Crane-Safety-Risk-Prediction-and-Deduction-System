# -*- coding: utf-8 -*-
import os
import logging
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from inference_service import VisualizationService, LLMDecisionService, HistoryManager
from risk_scoring import clip, score_by_indicator_name
from digital_twin_service import FACTOR_IDS, EVENTS, bundled_graph_edges, simulate_twin
from operation_models import OperationRequest, ValidationRequest
from operation_service import simulate_operations, validate_predictions

# =========================
# 1) 配置与日志
# =========================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger("safety-infer")

app = FastAPI(title="塔式起重机安全风险预测与推演 API", version="3.0.0")

# 添加CORS中间件，支持前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# 初始化服务
vis_service = VisualizationService(driver)
llm_service = LLMDecisionService()
history_manager = HistoryManager()


# =========================
# 2) 请求/响应模型
# =========================
class IndicatorInput(BaseModel):
    indicator_id: str = Field(..., description="指标ID")
    value: float = Field(..., description="指标值")


class InferRequest(BaseModel):
    factor_id: str
    stage_id: str
    indicators: List[IndicatorInput]
    lambda_factor: float = 1.0

    @field_validator("lambda_factor")
    def check_lambda(cls, v):
        if v < 0:
            raise ValueError("lambda_factor 必须 >= 0")
        return v


class ApiResponse(BaseModel):
    code: int
    message: str
    # 不同接口的数据既可能是列表（因素/阶段/指标），也可能是字典（推演结果）。
    data: Any = None


class TwinRequest(BaseModel):
    load_t: float = Field(5.0, ge=0, le=30)
    radius_m: float = Field(35.0, ge=2, le=80)
    wind_m_s: float = Field(8.0, ge=0, le=40)
    slew_deg: float = Field(35.0, ge=-360, le=360)
    event: str = "normal"
    tower_height_m: float = Field(48.0, ge=12, le=120)
    jib_length_m: float = Field(55.0, ge=15, le=80)
    attachment_heights_m: List[float] = Field(default_factory=lambda: [18.0, 36.0])
    mast_ei_kn_m2: float = Field(2_500_000, gt=0)
    rated_moment_kn_m: float = Field(2400, gt=0)
    mast_moment_capacity_kn_m: float = Field(3000, gt=0)
    attachment_capacity_kn: float = Field(180, gt=0)
    attachment_lever_arm_m: float = Field(5, gt=0)
    rated_load_t: float = Field(12, gt=0)
    max_free_height_m: float = Field(25, gt=0)
    projected_area_m2: float = Field(85, gt=0)
    drag_coefficient: float = Field(1.3, gt=0)

    @field_validator("event")
    @classmethod
    def valid_event(cls, value):
        if value not in EVENTS:
            raise ValueError("未知工序事件")
        return value

    @field_validator("attachment_heights_m")
    @classmethod
    def valid_attachments(cls, values):
        if len(values) > 8 or any(v <= 0 for v in values) or values != sorted(set(values)):
            raise ValueError("附着高度须为升序、不重复且大于零，最多 8 道")
        return values


# =========================
# 4) Cypher 查询
# =========================
Q_FACTOR_INDICATORS = """
MATCH (f:SafetyFactor {id:$factor_id})-[r:HAS_INDICATOR]->(i:Indicator)
RETURN f.id AS factor_id, f.name AS factor_name, coalesce(toFloat(f.initial_weight),0.0) AS w0,
       i.id AS indicator_id, i.name AS indicator_name, i.unit AS indicator_unit,
       coalesce(toFloat(r.weight),0.0) AS alpha,
       r.direction AS direction,
       toFloat(coalesce(r.normal_threshold, i.normal_high)) AS normal_t,
       toFloat(coalesce(r.warning_threshold, i.warning_threshold)) AS warning_t,
       toFloat(coalesce(r.danger_threshold, i.danger_threshold)) AS danger_t
"""

Q_STAGE_COEF = """
MATCH (f:SafetyFactor {id:$factor_id})
MATCH (f)-[a:ACTIVE_IN]->(s:ConstructionStage {id:$stage_id})
RETURN toFloat(a.stage_coefficient) AS k_stage, s.name AS stage_name
"""

Q_CAUSES = """
MATCH (f:SafetyFactor {id:$factor_id})-[c:CAUSES]->(r:SafetyRisk)
RETURN r.id AS risk_id, r.name AS risk_name, r.risk_level AS risk_level,
       coalesce(toFloat(c.trigger_prob),0.0) AS trigger_prob,
       coalesce(toFloat(c.influence),1.0) AS influence,
       coalesce(toFloat(c.confidence),1.0) AS confidence
"""

Q_LEADS_TO_FOR_RISKS = """
MATCH (r:SafetyRisk)-[l:LEADS_TO]->(a:SafetyAccident)
WHERE r.id IN $risk_ids
RETURN r.id AS risk_id, a.id AS accident_id, a.name AS accident_name,
       coalesce(toFloat(l.trigger_prob),0.0) AS trigger_prob,
       coalesce(toFloat(l.influence),1.0) AS influence,
       coalesce(toFloat(l.confidence),1.0) AS confidence
"""


# =========================
# 5) 推演主逻辑
# =========================
def run_inference(payload: InferRequest) -> Dict[str, Any]:
    indicator_map = {x.indicator_id: x.value for x in payload.indicators}
    if len(indicator_map) != len(payload.indicators):
        raise HTTPException(status_code=400, detail="指标ID不可重复")

    with driver.session(database=NEO4J_DATABASE) as session:
        rows = list(session.run(Q_FACTOR_INDICATORS, factor_id=payload.factor_id))
        if not rows:
            raise HTTPException(status_code=404, detail=f"factor_id={payload.factor_id} 不存在或未关联指标")

        w0 = rows[0]["w0"]
        factor_name = rows[0]["factor_name"]

        stage_row = session.run(Q_STAGE_COEF, factor_id=payload.factor_id, stage_id=payload.stage_id).single()
        if not stage_row:
            raise HTTPException(status_code=400, detail="所选因素不适用于该塔吊施工阶段")
        k_stage = float(stage_row["k_stage"])
        stage_name = stage_row["stage_name"]

        activation = 0.0
        explain = []
        used_alpha_sum = 0.0

        for r in rows:
            iid = r["indicator_id"]
            alpha = float(r["alpha"] or 0.0)
            if iid not in indicator_map:
                continue

            v = indicator_map[iid]
            s = score_by_indicator_name(
                indicator_name=r["indicator_name"],
                value=v,
                normal_t=r["normal_t"],
                warning_t=r["warning_t"],
                danger_t=r["danger_t"],
                direction=r["direction"]
            )
            contrib = alpha * s
            activation += contrib
            used_alpha_sum += alpha
            explain.append(f"{iid} 值={v} 危险度={s:.3f} 贡献={contrib:.3f}")

        if used_alpha_sum <= 1e-9:
            raise HTTPException(status_code=400, detail="请提供所选因素关联的至少一个指标")
        if used_alpha_sum > 1e-9 and used_alpha_sum < 0.999:
            activation = activation / used_alpha_sum
            explain.append(f"部分指标缺失，按有效alpha={used_alpha_sum:.3f}重归一")

        activation = clip(activation)
        w_dyn = clip(w0 * (1 + payload.lambda_factor * activation) * k_stage)
        explain.append(f"w0={w0:.3f}, activation={activation:.3f}, k_stage={k_stage:.3f}, w_dyn={w_dyn:.3f}")

        cause_rows = list(session.run(Q_CAUSES, factor_id=payload.factor_id))
        risk_probs = []
        risk_map = {}
        for c in cause_rows:
            p = clip(w_dyn * c["trigger_prob"] * c["influence"] * c["confidence"])
            risk_map[c["risk_id"]] = p
            risk_probs.append({
                "risk_id": c["risk_id"],
                "risk_name": c["risk_name"],
                "risk_level": c["risk_level"],
                "prob": round(p, 6)
            })

        chain_probs = []
        accident_tmp = {}
        if risk_map:
            lead_rows = list(session.run(Q_LEADS_TO_FOR_RISKS, risk_ids=list(risk_map.keys())))
            for l in lead_rows:
                p_r = risk_map[l["risk_id"]]
                p_ra = clip(l["trigger_prob"] * l["influence"] * l["confidence"])
                p_chain = clip(p_r * p_ra)

                chain_probs.append({
                    "factor_id": payload.factor_id,
                    "risk_id": l["risk_id"],
                    "accident_id": l["accident_id"],
                    "prob": round(p_chain, 6)
                })

                aid = l["accident_id"]
                if aid not in accident_tmp:
                    accident_tmp[aid] = {
                        "accident_name": l["accident_name"],
                        "terms": []
                    }
                accident_tmp[aid]["terms"].append(p_chain)

        accident_probs = []
        for aid, v in accident_tmp.items():
            prod = 1.0
            for t in v["terms"]:
                prod *= (1 - t)
            p_total = 1 - prod
            accident_probs.append({
                "accident_id": aid,
                "accident_name": v["accident_name"],
                "prob": round(clip(p_total), 6)
            })

        risk_probs.sort(key=lambda item: item["prob"], reverse=True)
        accident_probs.sort(key=lambda item: item["prob"], reverse=True)
        chain_probs.sort(key=lambda item: item["prob"], reverse=True)
        return {
            "factor_id": payload.factor_id,
            "factor_name": factor_name,
            "stage_id": payload.stage_id,
            "stage_name": stage_name,
            "indicator_values": [
                {"indicator_id": r["indicator_id"], "indicator_name": r["indicator_name"],
                 "unit": r["indicator_unit"],
                 "value": indicator_map[r["indicator_id"]]}
                for r in rows if r["indicator_id"] in indicator_map
            ],
            "activation_score": round(activation, 6),
            "dynamic_weight": round(w_dyn, 6),
            "risk_probs": risk_probs,
            "accident_probs": accident_probs,
            "chain_probs": chain_probs,
            "explanations": explain
        }


# =========================
# 6) API 路由 + 异常处理
# =========================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/health/db")
def health_db():
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            msg = session.run("RETURN 'ok' AS msg").single()["msg"]
        return {"status": msg}
    except Exception as e:
        logger.exception("数据库健康检查失败")
        raise HTTPException(status_code=500, detail=f"db health check failed: {str(e)}")

# 添加带API前缀的健康检查端点（向后兼容）
@app.get("/api/v1/health")
def api_health():
    return {"status": "ok"}

@app.get("/api/v1/health/db")
def api_health_db():
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            msg = session.run("RETURN 'ok' AS msg").single()["msg"]
        return {"status": msg}
    except Exception as e:
        logger.exception("数据库健康检查失败")
        raise HTTPException(status_code=500, detail=f"db health check failed: {str(e)}")


TWIN_CAUSES_QUERY = """
MATCH (f:SafetyFactor)-[c:CAUSES]->(r:SafetyRisk)
WHERE f.id IN $factor_ids
RETURN f.id AS factor_id, r.id AS risk_id, r.name AS risk_name,
       coalesce(toFloat(c.trigger_prob),0) AS trigger_prob,
       coalesce(toFloat(c.influence),0) AS influence,
       coalesce(toFloat(c.confidence),0) AS confidence
"""


@app.post("/api/v1/twin/simulate", response_model=ApiResponse)
def twin_simulate(req: TwinRequest):
    """Read graph relationships and run three illustrative future scenarios."""
    if req.radius_m > req.jib_length_m:
        raise HTTPException(status_code=422, detail="吊装幅度不能超过起重臂长度")
    if any(a >= req.tower_height_m for a in req.attachment_heights_m):
        raise HTTPException(status_code=422, detail="附着高度须低于塔身高度")
    try:
        from neo4j import Query
        with driver.session(database=NEO4J_DATABASE) as session:
            edges = session.run(Query(TWIN_CAUSES_QUERY, timeout=3.0), factor_ids=list(FACTOR_IDS)).data()
        if not edges:
            raise ValueError("Neo4j 中没有匹配的塔吊 CAUSES 关系")
        source = "Neo4j 实时只读查询"
    except Exception as exc:
        logger.warning("数字孪生使用 CSV 离线示范关系，Neo4j 查询失败: %s", exc)
        edges = bundled_graph_edges()
        source = "Knowledge Graph/CAUSES.csv 离线示范（Neo4j 未连接或缺少关系）"
    return ApiResponse(code=0, message="ok", data=simulate_twin(req.model_dump(), edges, source))


def operation_graph():
    """Use existing graph only for explanations; never multiply links into physical risk."""
    try:
        from neo4j import Query
        with driver.session(database=NEO4J_DATABASE) as session:
            edges = session.run(Query(TWIN_CAUSES_QUERY, timeout=2), factor_ids=list(FACTOR_IDS)).data()
        if not edges:
            raise ValueError("没有匹配关系")
        return edges, "Neo4j 实时只读关系（仅用于解释）"
    except Exception:
        return bundled_graph_edges(), "本地知识图谱 CSV（仅用于解释，无需连接 Neo4j）"


@app.get("/api/v1/operations/defaults", response_model=ApiResponse)
def operation_defaults():
    return ApiResponse(code=0, message="ok", data=OperationRequest().model_dump())


@app.post("/api/v1/operations/simulate", response_model=ApiResponse)
def operation_simulate(req: OperationRequest):
    try:
        edges, source = operation_graph()
        return ApiResponse(code=0, message="ok", data=simulate_operations(req, edges, source))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/operations/validate", response_model=ApiResponse)
def operation_validate(req: ValidationRequest):
    try:
        return ApiResponse(code=0, message="ok", data=validate_predictions(req.rows))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/operations/advice", response_model=ApiResponse)
def operation_advice(req: OperationRequest):
    """Recompute authoritative numbers rather than accepting client-authored probabilities."""
    try:
        edges, source = operation_graph()
        result = simulate_operations(req, edges, source)
        evidence = {key: result[key] for key in ("model_status", "probability_definition", "recommendation", "warnings", "knowledge_chains")}
        evidence["measures"] = [{key: scenario[key] for key in ("id", "label", "risk", "peaks", "costs", "constraints", "comparison", "knowledge_rule")} for scenario in result["scenarios"]]
        decision = llm_service.generate_decision({"operation_evidence": evidence})
        return ApiResponse(code=0, message="ok", data={"explanation": decision, "recommendation": result["recommendation"], "ensemble_id": result["ensemble"]["id"]})
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("操作解释生成失败: %s", exc)
        return ApiResponse(code=1, message=str(exc))

@app.post("/api/v1/infer/safety", response_model=ApiResponse)
def infer_safety(req: InferRequest):
    try:
        logger.info("infer request: factor_id=%s stage_id=%s", req.factor_id, req.stage_id)
        data = run_inference(req)
        
        # 记录到历史
        history_manager.add_record(req.dict(), data)
        
        return ApiResponse(code=0, message="ok", data=data)
    except HTTPException as e:
        logger.warning("业务异常: %s", e.detail)
        raise e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Neo4jError as e:
        logger.exception("Neo4j异常")
        raise HTTPException(status_code=500, detail=f"Neo4j error: {str(e)}")
    except Exception as e:
        logger.exception("未知异常")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


# =========================
# 7) 新增 - 可视化和决策API
# =========================
@app.get("/api/v1/graph/{factor_id}", response_model=ApiResponse)
def get_knowledge_graph(factor_id: str):
    """获取知识图谱可视化数据"""
    try:
        graph_data = vis_service.get_knowledge_graph(factor_id)
        return ApiResponse(code=0, message="ok", data=graph_data)
    except Exception as e:
        logger.exception(f"获取知识图谱失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/infer/llm-status", response_model=ApiResponse)
def get_llm_status():
    return ApiResponse(code=0, message="ok", data=llm_service.configuration_status())


@app.post("/api/v1/infer/llm-decision", response_model=ApiResponse)
def get_llm_decision(payload: Dict[str, Any]):
    """使用LLM生成决策建议"""
    try:
        infer_result = payload.get("infer_result", {})
        decision = llm_service.generate_decision(infer_result)
        
        # 记录LLM决策
        if history_manager.records:
            history_manager.records[0]["llm_decision"] = True
        
        return ApiResponse(code=0, message="ok", data=decision)
    except Exception as e:
        logger.exception(f"LLM决策失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/factors", response_model=ApiResponse)
def get_all_factors():
    """获取所有安全因素列表"""
    try:
        query = """
        MATCH (f:SafetyFactor)
        RETURN f.id AS id, f.name AS name, f.description AS description, 
               coalesce(toFloat(f.initial_weight), 0.0) AS initial_weight
        ORDER BY f.id
        """
        with driver.session(database=NEO4J_DATABASE) as session:
            results = list(session.run(query).data())
        return ApiResponse(code=0, message="ok", data=results)
    except Exception as e:
        logger.exception(f"获取因素列表失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/stages", response_model=ApiResponse)
def get_all_stages():
    """获取所有施工阶段列表"""
    try:
        query = """
        MATCH (s:ConstructionStage)
        RETURN s.id AS id, s.name AS name, s.description AS description
        ORDER BY s.id
        """
        with driver.session(database=NEO4J_DATABASE) as session:
            results = list(session.run(query).data())
        return ApiResponse(code=0, message="ok", data=results)
    except Exception as e:
        logger.exception(f"获取阶段列表失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/indicators", response_model=ApiResponse)
def get_all_indicators():
    """获取所有指标列表"""
    try:
        query = """
        MATCH (i:Indicator)
        RETURN i.id AS id, i.name AS name, i.unit AS unit,
               i.data_type AS data_type,
               toFloat(i.normal_high) AS normal_threshold,
               toFloat(i.warning_threshold) AS warning_threshold,
               toFloat(i.danger_threshold) AS danger_threshold
        ORDER BY i.id
        """
        with driver.session(database=NEO4J_DATABASE) as session:
            results = list(session.run(query).data())
        return ApiResponse(code=0, message="ok", data=results)
    except Exception as e:
        logger.exception(f"获取指标列表失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/factors/{factor_id}/indicators", response_model=ApiResponse)
def get_factor_indicators(factor_id: str):
    """返回所选因素关联的塔吊指标及该关系的风险方向和阈值。"""
    query = """
    MATCH (f:SafetyFactor {id:$factor_id})-[r:HAS_INDICATOR]->(i:Indicator)
    RETURN i.id AS id, i.name AS name, i.unit AS unit, i.data_type AS data_type,
           r.direction AS direction, toFloat(r.weight) AS weight,
           toFloat(r.normal_threshold) AS normal_threshold,
           toFloat(r.warning_threshold) AS warning_threshold,
           toFloat(r.danger_threshold) AS danger_threshold
    ORDER BY i.id
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            data = session.run(query, factor_id=factor_id).data()
        return ApiResponse(code=0, message="ok", data=data)
    except Exception as e:
        logger.exception("获取因素指标失败")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/factors/{factor_id}/stages", response_model=ApiResponse)
def get_factor_stages(factor_id: str):
    """仅返回该因素在图谱中标为适用的阶段。"""
    query = """
    MATCH (f:SafetyFactor {id:$factor_id})-[a:ACTIVE_IN]->(s:ConstructionStage)
    RETURN s.id AS id, s.name AS name, s.description AS description,
           toFloat(a.stage_coefficient) AS stage_coefficient
    ORDER BY toInteger(s.sequence)
    """
    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            data = session.run(query, factor_id=factor_id).data()
        return ApiResponse(code=0, message="ok", data=data)
    except Exception as e:
        logger.exception("获取适用阶段失败")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/history", response_model=ApiResponse)
def get_inference_history(limit: int = 50):
    """获取推演历史记录"""
    try:
        history = history_manager.get_history(limit)
        return ApiResponse(code=0, message="ok", data=history)
    except Exception as e:
        logger.exception(f"获取历史失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))


@app.get("/api/v1/statistics", response_model=ApiResponse)
def get_statistics():
    """获取统计信息"""
    try:
        stats = history_manager.get_statistics()
        return ApiResponse(code=0, message="ok", data=stats)
    except Exception as e:
        logger.exception(f"获取统计失败: {str(e)}")
        return ApiResponse(code=1, message=str(e))

@app.on_event("shutdown")
def on_shutdown():
    try:
        driver.close()
    except Exception:
        pass


app.mount("/twin", StaticFiles(directory=Path(__file__).resolve().parent / "digital_twin", html=True), name="digital_twin")

