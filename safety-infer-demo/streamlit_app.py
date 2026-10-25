# -*- coding: utf-8 -*-
"""
塔式起重机安全风险预测与推演系统 - Streamlit 前端
"""
import streamlit as st
import requests
import io
import hashlib
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from workbench_ui import (apply_workbench_style, render_header, sidebar_brand,
                          section_label, render_selection_summary, render_welcome, render_footer)

# ==================== 配置 ====================
st.set_page_config(
    page_title="塔式起重机安全风险预测与推演系统",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API基础URL
API_BASE = "http://localhost:8000/api/v1"
PREDICTION_API_BASE = "http://localhost:8001/api/v1"
PREDICTION_TIME_STEPS = 20

apply_workbench_style()

# ==================== 会话状态初始化 ====================
if 'last_result' not in st.session_state:
    st.session_state.last_result = None
if 'factors_list' not in st.session_state:
    st.session_state.factors_list = []
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'last_decision_error' not in st.session_state:
    st.session_state.last_decision_error = None

# ==================== 工具函数 ====================
def load_factors():
    """加载安全因素列表"""
    try:
        st.session_state.catalog_error = None
        response = requests.get(f"{API_BASE}/factors", timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            st.session_state.factors_list = data
            return data
        else:
            st.session_state.catalog_error = f"加载因素列表失败：HTTP {response.status_code}"
    except requests.exceptions.ConnectionError:
        st.session_state.catalog_error = "无法连接图谱服务，请确认后端服务已启动。"
    except requests.exceptions.Timeout:
        st.session_state.catalog_error = "图谱数据读取超时，请稍后重新加载。"
    except Exception as e:
        st.session_state.catalog_error = f"加载因素列表失败：{e}"
    
    st.session_state.factors_list = []
    return []


def load_factor_related(factor_id, relation):
    """获取该因素实际关联的指标或适用阶段。"""
    try:
        response = requests.get(
            f"{API_BASE}/factors/{factor_id}/{relation}", timeout=5
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(payload.get("message", "查询失败"))
        return payload.get("data") or []
    except Exception as exc:
        st.session_state[f"{relation}_error"] = str(exc)
        return []


def run_inference(factor_id, stage_id, indicators_input, lambda_factor):
    """执行推演"""
    try:
        payload = {
            "factor_id": factor_id,
            "stage_id": stage_id,
            "indicators": indicators_input,
            "lambda_factor": lambda_factor
        }
        
        response = requests.post(
            f"{API_BASE}/infer/safety",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json().get("data")
        else:
            st.error(f"推演失败: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"推演异常: {str(e)}")
        return None


def get_llm_decision(infer_result):
    """获取LLM决策"""
    st.session_state.last_decision_error = None
    try:
        payload = {"infer_result": infer_result}
        response = requests.post(
            f"{API_BASE}/infer/llm-decision",
            json=payload,
            timeout=240
        )
        
        body = response.json()
        if response.status_code != 200 or body.get("code") != 0:
            st.session_state.last_decision_error = f"AI 建议生成失败：{body.get('message') or body.get('detail') or response.status_code}"
            return None
        return body.get("data")
    except requests.exceptions.ConnectionError:
        st.session_state.last_decision_error = "无法连接端口 8000 的后端服务，请先启动后端，再生成建议。"
        return None
    except requests.exceptions.Timeout:
        st.session_state.last_decision_error = "等待后端建议响应超时，请检查后端日志或稍后重试。"
        return None
    except Exception as e:
        st.session_state.last_decision_error = f"AI 建议生成失败：{e}"
        return None


def render_llm_configuration():
    with st.expander("DeepSeek 配置与诊断"):
        st.caption("密钥由端口 8000 的后端读取。配置检查不会发送付费模型请求，也不会返回密钥内容。")
        if st.button("检查后端配置", key="check_llm_config"):
            try:
                response = requests.get(f"{API_BASE}/infer/llm-status", timeout=5)
                response.raise_for_status()
                body = response.json()
                if body.get("code") != 0 or not isinstance(body.get("data"), dict):
                    raise RuntimeError("后端未返回完整的模型配置状态")
                st.session_state.llm_configuration = body["data"]
            except Exception:
                st.session_state.llm_configuration = {"status": "unavailable", "message": "无法获取配置状态，请确认后端已更新并启动。"}
        status = st.session_state.get("llm_configuration")
        if status:
            if status["status"] == "configured":
                st.success(status["message"])
            else:
                st.warning(status["message"])
            if status.get("model"):
                st.caption(f"后端模型：{status['model']}")
        st.caption("可以复制项目根目录的 .env.example 为 .env，在本机填写 DEEPSEEK_API_KEY；不要将密钥发到聊天中。")


def get_knowledge_graph(factor_id):
    """获取知识图谱"""
    try:
        response = requests.get(
            f"{API_BASE}/graph/{factor_id}",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json().get("data")
        else:
            st.error(f"获取图谱失败: HTTP {response.status_code}")
            return None
    except Exception as e:
        st.error(f"获取图谱异常: {str(e)}")
        return None


def request_prediction(uploaded_file, horizon):
    """把上传的 CSV 发给独立 CNN-LSTM-AT 预测服务。"""
    response = requests.post(
        f"{PREDICTION_API_BASE}/predict",
        params={"horizon": int(horizon)},
        files={"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")},
        timeout=180,
    )
    if response.status_code != 200:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"HTTP {response.status_code}: {detail}")
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(payload.get("message", "预测服务返回失败"))
    result = pd.DataFrame(payload.get("data", {}).get("rows", []))
    if result.empty or not {"time", "output", "kind"}.issubset(result.columns):
        raise RuntimeError("预测服务没有返回完整的结果")
    return result


def render_prediction_panel():
    """保留历史 CSV 上传与 CNN-LSTM-AT 预测入口。"""
    st.markdown("## 历史数据预测")
    st.caption("上传包含 output、input1–input8 的历史 CSV；time 列可选。当前模型字段含义待后续定义。")
    with st.expander("数据上传与预测设置", expanded=True):
        uploaded_file = st.file_uploader(
            "上传历史数据 CSV", type=["csv"], key="prediction_upload"
        )
        horizon = st.number_input(
            "预测未来步数", min_value=1, max_value=100, value=1, step=1,
            key="prediction_horizon"
        )
        signature = (
            uploaded_file.name,
            hashlib.sha256(uploaded_file.getvalue()).hexdigest(),
            int(horizon),
        ) if uploaded_file is not None else None
        if st.session_state.get("prediction_signature") != signature:
            st.session_state.prediction_signature = signature
            st.session_state.prediction_result = None
        history = None
        if uploaded_file is not None:
            try:
                history = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
                st.caption(f"已读取 {len(history)} 行；至少需要 {PREDICTION_TIME_STEPS} 行。")
                st.dataframe(history.head(5), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"CSV 读取失败：{exc}")
        if history is not None and st.button(
            "开始参数预测", type="primary", use_container_width=True,
            key="run_prediction"
        ):
            try:
                with st.spinner("正在请求预测服务……"):
                    st.session_state.prediction_result = request_prediction(
                        uploaded_file, int(horizon)
                    )
                st.success(f"预测完成，共生成 {int(horizon)} 个未来数据点。")
            except requests.exceptions.ConnectionError:
                st.error("无法连接预测服务，请确认端口 8001 已启动。")
            except Exception as exc:
                st.error(f"预测失败：{exc}")

        result = st.session_state.prediction_result
        if isinstance(result, pd.DataFrame) and not result.empty:
            historical = result[result["kind"] == "历史"]
            future = result[result["kind"] == "预测"]
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=historical["time"], y=historical["output"],
                mode="lines", name="历史 output"
            ))
            future_x, future_y = future["time"].tolist(), future["output"].tolist()
            if not historical.empty:
                future_x.insert(0, historical.iloc[-1]["time"])
                future_y.insert(0, historical.iloc[-1]["output"])
            figure.add_trace(go.Scatter(
                x=future_x, y=future_y, mode="lines+markers", name="预测 output"
            ))
            figure.update_layout(template="crane_workbench", xaxis_title="时间", yaxis_title="output", height=420)
            st.plotly_chart(figure, use_container_width=True)
            st.info("多步预测时，未来 input1–input8 沿用历史数据最后一行的值。")
            st.dataframe(future, use_container_width=True, hide_index=True)
            st.download_button(
                "下载预测结果 CSV",
                data=future.to_csv(index=False).encode("utf-8-sig"),
                file_name="predicted_future_output.csv",
                mime="text/csv",
                use_container_width=True,
            )


def visualize_knowledge_graph(graph_data):
    """使用Pyvis可视化知识图谱"""
    try:
        if not graph_data or not graph_data.get("nodes"):
            st.warning("没有图谱数据")
            return
        
        from pyvis.network import Network

        # 移除physics参数以兼容不同版本的Pyvis
        net = Network(
            height="600px",
            directed=True,
            notebook=False
        )
        
        # 添加节点
        for node in graph_data.get("nodes", []):
            net.add_node(
                node["id"],
                label=node["label"],
                color=node.get("color", "#4ECDC4"),
                size=node.get("size", 20),
                title=node.get("title", "")
            )
        
        # 添加边
        for edge in graph_data.get("edges", []):
            net.add_edge(
                edge["source"],
                edge["target"],
                title=edge.get("title", ""),
                label=edge.get("relation", "")
            )
        
        # 配置物理引擎（在创建后配置）
        net.show_buttons(filter_=['physics'])
        net.write_html("graph.html")
        
        # 显示图谱
        with open("graph.html", 'r', encoding='utf-8') as f:
            html_string = f.read()
        st.components.v1.html(html_string, height=600, scrolling=True)
    except Exception as e:
        st.error(f"图谱可视化失败: {str(e)}")


# ==================== 主界面 ====================
render_header()
# Horizontal radio is compatible with the project's pinned Streamlit 1.28.
workspace = st.radio("工作空间", ["安全风险推演", "历史数据预测"], horizontal=True,
                     label_visibility="collapsed", key="workspace")
previous_workspace = st.session_state.get("active_workspace", "安全风险推演")
if previous_workspace != workspace and previous_workspace == "安全风险推演":
    # Widgets absent from a view are cleaned up by Streamlit; retain their inputs.
    st.session_state.workbench_inputs = {
        key: value for key, value in st.session_state.items()
        if key == "selected_factor" or key == "lambda_factor" or key.startswith(("indicator_", "stage_"))
    }
st.session_state.active_workspace = workspace
saved_inputs = st.session_state.get("workbench_inputs", {})

with st.sidebar:
    sidebar_brand()
    if workspace == "安全风险推演":
        section_label("01", "分析对象")
        if not st.session_state.factors_list:
            with st.spinner("读取图谱因素…"):
                load_factors()
        factors = st.session_state.factors_list or []
        if factors:
            st.markdown(f'<div class="catalog-status">知识图谱已加载 · {len(factors)} 项安全因素</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="catalog-status pending">图谱数据未就绪 · 请连接服务后重新加载</div>', unsafe_allow_html=True)
            if st.button("重新加载数据", key="reload_catalog"):
                load_factors()
                st.rerun()
            if st.session_state.get("catalog_error"):
                with st.expander("连接详情"):
                    st.caption(st.session_state.catalog_error)
        factors_options = {f["name"]: f["id"] for f in factors}
        if factors_options:
            factor_names = list(factors_options)
            saved_factor = saved_inputs.get("selected_factor")
            factor_name = st.selectbox("安全因素", options=factor_names,
                                       index=factor_names.index(saved_factor) if saved_factor in factor_names else 0,
                                       key="selected_factor")
            factor_id = factors_options[factor_name]
        else:
            factor_name = None
            factor_id = st.text_input("安全因素 ID", value="F001")
        factor_stages = load_factor_related(factor_id, "stages")
        stages_options = {s["name"]: s["id"] for s in factor_stages}
        if stages_options:
            stage_names = list(stages_options)
            saved_stage = saved_inputs.get(f"stage_{factor_id}")
            stage_name = st.selectbox("施工阶段", options=stage_names,
                                      index=stage_names.index(saved_stage) if saved_stage in stage_names else 0,
                                      key=f"stage_{factor_id}")
            stage_id = stages_options[stage_name]
        else:
            stage_name = None
            stage_id = st.text_input("施工阶段 ID", value="S001")
        section_label("02", "监测指标")
        indicators_to_show = load_factor_related(factor_id, "indicators")
        indicator_inputs = []
        if not indicators_to_show:
            st.caption("关联指标尚未获取，数据就绪后将在这里显示输入项。")
        for indicator in indicators_to_show:
            ind_id = indicator["id"]
            unit = indicator.get("unit") or ""
            normal = float(indicator.get("normal_threshold") or 0)
            is_integer = indicator.get("data_type") == "integer"
            saved_value = saved_inputs.get(f"indicator_{ind_id}", normal)
            value = st.number_input(
                f"{indicator['name']} · {unit}" if unit else indicator['name'],
                value=int(saved_value) if is_integer else float(saved_value),
                min_value=0 if is_integer else 0.0, step=1 if is_integer else 0.5,
                help=(f"正常 {normal:g}；预警 {indicator['warning_threshold']}；"
                      f"危险 {indicator['danger_threshold']}；"
                      f"{'数值越低风险越高' if indicator.get('direction') == 'negative' else '数值越高风险越高'}"),
                key=f"indicator_{ind_id}"
            )
            indicator_inputs.append({"indicator_id": ind_id, "value": value})
        with st.expander("高级参数"):
            lambda_factor = st.slider("因素权重倍数", min_value=0.5, max_value=2.0,
                                      value=float(saved_inputs.get("lambda_factor", 1.0)), step=0.1,
                                      help="调整因素激活度对动态权重的影响。", key="lambda_factor")
        section_label("03", "运行推演")
        run_col, reset_col = st.columns([3, 1])
        with run_col:
            if st.button("开始风险推演", type="primary", key="run_inference",
                         disabled=not (indicators_to_show and factor_stages)):
                with st.spinner("正在计算关联风险与事故链…"):
                    inferred = run_inference(factor_id, stage_id, indicator_inputs, lambda_factor)
                if inferred:
                    st.session_state.last_result = inferred
                    st.session_state.factor_id = factor_id
                    st.session_state.last_decision = None
                    st.session_state.last_decision_error = None
                    st.rerun()
        with reset_col:
            if st.button("重置", key="reset_inference", help="清空本次推演结果，保留输入参数"):
                st.session_state.last_result = None
                st.session_state.last_decision = None
                st.session_state.last_decision_error = None
                st.rerun()
        st.caption("输入值用于当前情景分析，不会修改原始知识图谱。")
    else:
        section_label("01", "历史数据预测")
        st.caption("在右侧上传历史 CSV，设置预测步数并查看变化趋势。")
        section_label("02", "数据要求")
        st.caption("至少 20 行记录，包含 output 与 input1–input8 数值列。time 列可选。")
        st.caption("当前预测模型独立运行，不参与图谱情景概率计算。")
    st.markdown('<div class="sidebar-section"><span>↗</span>运行操作比较</div>', unsafe_allow_html=True)
    st.markdown("[打开塔吊数字孪生页面](http://localhost:8000/twin/)")

# ==================== 主内容区 ====================
if workspace == "历史数据预测":
    render_prediction_panel()
elif st.session_state.last_result:
    result = st.session_state.last_result
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["推演概览", "知识关联", "风险分析", "分析建议", "历史记录"])
    
    # TAB 1: 推演结果
    with tab1:
        st.markdown("## 推演结果概览")
        
        # 关键指标卡片
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            activation = result.get("activation_score", 0)
            st.metric(
                "激活度",
                f"{activation:.1%}",
                delta=f"动态权重: {result.get('dynamic_weight', 0):.3f}"
            )
        
        with col2:
            st.metric("因素名称", result.get("factor_name", "N/A")[:15])
        
        with col3:
            st.metric("施工阶段", result.get("stage_name", result.get("stage_id", "N/A")))
        
        with col4:
            risk_count = len(result.get("risk_probs", []))
            st.metric("风险数量", risk_count)
        
        st.caption(" ")
        
        # 风险分布图表
        st.markdown("### 关联风险分布")
        risk_probs = result.get("risk_probs", [])
        
        if risk_probs:
            risks_df = pd.DataFrame(risk_probs)
            
            # 按概率排序
            risks_df = risks_df.sort_values("prob", ascending=False)
            
            # 创建柱状图
            fig = px.bar(
                risks_df,
                x="risk_name",
                y="prob",
                color="prob",
                color_continuous_scale=["#dcece5", "#91bba9", "#d8a06f", "#b96656"],
                labels={"prob": "风险概率", "risk_name": "风险类型"},
                title="各风险类型的激活概率",
                height=400
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                hovermode="x unified",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示风险列表
            st.markdown("### 风险详情")
            risks_display = risks_df[["risk_id", "risk_name", "prob"]].copy()
            risks_display["风险概率"] = risks_display["prob"].apply(lambda x: f"{x:.1%}")
            risks_display = risks_display[["risk_id", "risk_name", "风险概率"]]
            risks_display.columns = ["风险ID", "风险名称", "概率"]
            st.dataframe(risks_display, use_container_width=True, hide_index=True)
        
        st.caption(" ")
        
        # 事故链传播
        accident_probs = result.get("accident_probs", [])
        if accident_probs:
            st.markdown("### 事故链传播分析")
            
            accident_df = pd.DataFrame(accident_probs)
            accident_df = accident_df.sort_values("prob", ascending=False)
            accident_df["事故概率"] = accident_df["prob"].apply(lambda x: f"{x:.1%}")
            
            fig_accident = px.bar(
                accident_df,
                x="accident_name",
                y="prob",
                color="prob",
                color_continuous_scale=["#f0e6ce", "#d5b675", "#c87961"],
                labels={"prob": "事故概率", "accident_name": "事故类型"},
                title="可能导致的事故概率",
                height=400
            )
            fig_accident.update_layout(xaxis_tickangle=-45, showlegend=False)
            st.plotly_chart(fig_accident, use_container_width=True)
    
    # TAB 2: 知识图谱
    with tab2:
        st.markdown("## 知识图谱可视化")
        
        if st.button("加载知识图谱", use_container_width=True):
            with st.spinner("加载中..."):
                graph_data = get_knowledge_graph(st.session_state.factor_id)
                if graph_data:
                    st.success(f"已加载 {graph_data.get('count', 0)} 个节点")
                    visualize_knowledge_graph(graph_data)
                    
                    st.markdown("### 节点说明")
                    st.markdown("""
                    - 🔴 **红色** - 安全因素
                    - 🔷 **青色** - 监测指标
                    - 🔸 **浅红** - 风险因素
                    - 🟡 **金色** - 可能事故
                    """)
    
    # TAB 3: 风险分析
    with tab3:
        st.markdown("## 风险分析")
        
        risk_probs = result.get("risk_probs", [])
        if risk_probs:
            # 风险等级分类
            high_risks = [r for r in risk_probs if r["prob"] >= 0.7]
            mid_risks = [r for r in risk_probs if 0.3 <= r["prob"] < 0.7]
            low_risks = [r for r in risk_probs if r["prob"] < 0.3]
            
            # 风险分布饼图
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = px.pie(
                    values=[len(high_risks), len(mid_risks), len(low_risks)],
                    names=["高风险", "中风险", "低风险"],
                    color_discrete_map={
                        "高风险": "#c87969",
                        "中风险": "#d7b06d",
                        "低风险": "#78ac99"
                    },
                    title="风险等级分布"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # 风险统计
            with col2:
                st.markdown("### 风险统计")
                st.metric("高风险", f"{len(high_risks)} 项")
                st.metric("中风险", f"{len(mid_risks)} 项")
                st.metric("低风险", f"{len(low_risks)} 项")
            
            # 风险矩阵
            st.markdown("### 风险矩阵")
            
            matrix_data = []
            for idx, risk in enumerate(risk_probs[:6], 1):
                prob = risk.get("prob", 0)
                
                # 可能性 (1-5)
                possibility = min(5, int(prob * 5) + 1)
                # 严重性来自图谱风险等级，不由发生概率推算
                severity = {"高": 5, "中": 3, "低": 1}.get(
                    risk.get("risk_level"), 3
                )
                # 风险评分
                risk_score = possibility * severity
                
                if risk_score >= 16:
                    level = "极高"
                    color = "🔴"
                elif risk_score >= 12:
                    level = "高"
                    color = "🟠"
                elif risk_score >= 8:
                    level = "中"
                    color = "🟡"
                else:
                    level = "低"
                    color = "🟢"
                
                matrix_data.append({
                    "序号": idx,
                    "风险": risk.get("risk_name", ""),
                    "概率": f"{prob:.1%}",
                    "可能性": "★" * possibility,
                    "严重性": "★" * severity,
                    "等级": f"{color} {level}"
                })
            
            matrix_df = pd.DataFrame(matrix_data)
            st.dataframe(matrix_df, use_container_width=True, hide_index=True)
    
    # TAB 4: AI决策
    with tab4:
        st.markdown("## 分析与行动建议")
        render_llm_configuration()
        
        if st.button("生成分析建议", type="primary", key="generate_decision"):
            st.session_state.last_decision = None
            with st.spinner("DeepSeek 正在分析……"):
                decision = get_llm_decision(result)
                
                if decision:
                    st.session_state.last_decision = decision
        
        # 显示已缓存的决策
        if st.session_state.get('last_decision'):
            decision = st.session_state.last_decision
            
            # 情况分析
            st.markdown("### 安全形势")
            situation = decision.get("situation_analysis", "")
            if situation:
                st.info(situation)
            
            # 主要警告
            st.markdown("### 主要风险")
            warnings = decision.get("main_warnings", [])
            for i, warning in enumerate(warnings, 1):
                st.error(f"**{i}. {warning}**")
            
            # 防控措施
            st.markdown("### 防控措施")
            measures = decision.get("prevention_measures", [])
            
            col1, col2 = st.columns(2)
            immediate = measures[:3] if len(measures) > 3 else measures
            short_term = measures[3:] if len(measures) > 3 else []
            
            with col1:
                st.markdown("**立即行动**")
                for measure in immediate:
                    st.success(f"• {measure}")
            
            with col2:
                st.markdown("**短期措施**")
                for measure in short_term:
                    st.success(f"• {measure}")
            
            # 应急响应
            st.markdown("### 应急响应")
            emergency = decision.get("emergency_response", "")
            if emergency:
                st.markdown(f"<div class='danger-box'>{emergency}</div>", unsafe_allow_html=True)
            
            # 监测重点
            st.markdown("### 监测重点")
            monitoring = decision.get("monitoring_focus", "")
            if monitoring:
                st.markdown(f"<div class='warning-box'>{monitoring}</div>", unsafe_allow_html=True)
        elif st.session_state.get("last_decision_error"):
            st.error(st.session_state.last_decision_error)
        else:
            st.info("点击上方按钮生成分析建议")
    
    # TAB 5: 历史记录
    with tab5:
        st.markdown("## 推演历史")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("加载历史记录", use_container_width=True):
                try:
                    response = requests.get(f"{API_BASE}/history?limit=20", timeout=5)
                    if response.status_code == 200:
                        history = response.json().get("data", [])
                        st.session_state.history = history
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
        
        with col2:
            if st.button("加载统计信息", use_container_width=True):
                try:
                    response = requests.get(f"{API_BASE}/statistics", timeout=5)
                    if response.status_code == 200:
                        stats = response.json().get("data", {})
                        st.session_state.statistics = stats
                except Exception as e:
                    st.error(f"加载失败: {str(e)}")
        
        with col3:
            if st.button("清空会话", use_container_width=True):
                st.session_state.last_result = None
                st.session_state.last_decision = None
                st.session_state.last_decision_error = None
                st.rerun()
        
        st.caption(" ")
        
        # 显示历史
        if hasattr(st.session_state, 'history'):
            history_df = pd.DataFrame(st.session_state.history)
            if not history_df.empty:
                st.dataframe(history_df, use_container_width=True)
            else:
                st.info("暂无历史记录")
        
        # 显示统计
        if hasattr(st.session_state, 'statistics'):
            stats = st.session_state.statistics
            
            st.markdown("### 统计信息")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("总推演次数", stats.get("total_inferences", 0))
            
            with col2:
                avg_activation = stats.get("avg_activation_score", 0)
                st.metric("平均激活度", f"{avg_activation:.1%}")
            
            with col3:
                st.metric("高风险推演", stats.get("high_risk_count", 0))
            
            # 因素分布
            factors_dist = stats.get("factors_distribution", {})
            if factors_dist:
                st.markdown("### 因素分布")
                fig_factor = px.bar(
                    x=list(factors_dist.keys()),
                    y=list(factors_dist.values()),
                    labels={"x": "安全因素", "y": "推演次数"},
                    title="各因素的推演频率"
                )
                st.plotly_chart(fig_factor, use_container_width=True)

else:
    render_selection_summary(factor_name, stage_name, len(indicators_to_show))
    render_welcome(bool(indicators_to_show and factor_stages))

render_footer()
