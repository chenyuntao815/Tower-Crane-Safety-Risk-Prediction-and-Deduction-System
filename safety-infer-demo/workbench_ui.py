"""Presentation helpers for the Streamlit workbench; no inference logic."""
from html import escape
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
import streamlit as st


def apply_workbench_style():
    css = (Path(__file__).parent / "assets" / "workbench.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    pio.templates["crane_workbench"] = go.layout.Template(
        layout=dict(
            font=dict(family="Microsoft YaHei, Segoe UI, sans-serif", color="#415664", size=12),
            paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
            colorway=["#147D73", "#D5A04B", "#5385AC", "#C87163", "#86B8A8"],
            margin=dict(l=24, r=24, t=52, b=32),
            title=dict(font=dict(size=15, color="#203442"), x=0.02),
            xaxis=dict(showgrid=False, zeroline=False, automargin=True),
            yaxis=dict(gridcolor="#EDF1F3", zeroline=False, automargin=True),
            legend=dict(orientation="h", y=-0.18, x=0),
            hoverlabel=dict(bgcolor="#203442", font_color="white"),
        )
    )
    px.defaults.template = "crane_workbench"


def render_header():
    st.markdown('''
    <div class="workbench-topline"><div><span class="brand-square">T</span>
    <span class="brand-label">CRANE SAFETY <b> / </b> 塔吊安全研究平台</span></div>
    <a href="http://localhost:8000/twin/" target="_blank" rel="noopener noreferrer">三维数字孪生 <span>↗</span></a></div>
    <div class="workbench-hero"><div class="hero-copy">
    <div class="eyebrow">RISK ANALYSIS WORKSPACE</div>
    <h1>塔吊安全风险推演工作台</h1>
    <p>从现场观测到风险链分析，让每一次工况评估都有依据。</p>
    <div class="hero-tags"><span>知识图谱驱动</span><span>多指标情景分析</span><span>可追溯的决策依据</span></div>
    </div><div class="hero-diagram" aria-hidden="true"><div class="diagram-label">ANALYSIS PATH</div>
    <div class="diagram-node">安全因素 <i>01</i></div><div class="diagram-connector"></div>
    <div class="diagram-node light">监测指标 <i>02</i></div><div class="diagram-connector"></div>
    <div class="diagram-node">风险与事故链 <i>03</i></div></div></div>
    ''', unsafe_allow_html=True)


def sidebar_brand():
    st.markdown('''<div class="sidebar-brand"><span class="brand-square">T</span>
    <div><strong>推演参数</strong><small>SCENARIO CONFIGURATION</small></div></div>''', unsafe_allow_html=True)


def section_label(number, title):
    st.markdown(f'<div class="sidebar-section"><span>{escape(str(number))}</span>{escape(title)}</div>', unsafe_allow_html=True)


def render_selection_summary(factor, stage, indicator_count):
    values = [("当前安全因素", factor or "尚未选择"), ("适用施工阶段", stage or "等待关联数据"),
              ("待分析监测指标", f"{indicator_count} 项" if indicator_count else "等待指标数据")]
    st.markdown('<div class="selection-summary">' + ''.join(
        f'<div><small>{label}</small><strong>{escape(value)}</strong></div>' for label, value in values
    ) + '</div>', unsafe_allow_html=True)


def render_welcome(ready):
    st.markdown('''<div class="welcome-card"><div class="welcome-kicker">开始一次新的工况评估</div>
    <h2>选择因素，输入观测，查看风险。</h2><p>在左侧配置当前施工场景，运行后将在这里呈现风险分布、事故传播链和分析建议。</p>
    <div class="start-steps"><div><span>01</span><strong>选择分析对象</strong><small>安全因素与适用施工阶段</small></div>
    <div><span>02</span><strong>填写监测数据</strong><small>依据图谱阈值输入现场观测</small></div>
    <div><span>03</span><strong>运行风险推演</strong><small>查看关联风险与建议依据</small></div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="section-heading"><h3>一次推演，串联完整分析</h3><span>分析结果将在运行后生成</span></div>', unsafe_allow_html=True)
    cards = [
        ("01", "风险分布", "对比关联风险的情景概率，聚焦需要优先关注的环节。", "关联风险 · 分布统计"),
        ("02", "知识关联", "沿因素、监测指标与事故关系，追溯推演结果的依据。", "知识图谱 · 事故链"),
        ("03", "行动建议", "结合当前结果生成防控、应急和监测建议，辅助人工研判。", "措施解释 · 监测重点"),
    ]
    for column, (number, title, body, foot) in zip(st.columns(3), cards):
        with column:
            st.markdown(f'<div class="feature-card"><span class="feature-number">{number}</span><h3>{title}</h3><p>{body}</p><small>{foot}</small></div>', unsafe_allow_html=True)
    if not ready:
        st.caption("图谱数据尚未就绪。连接图谱服务后，在左侧重新加载数据即可开始；历史数据预测可通过上方入口使用。")


def render_footer():
    st.markdown('''<div class="workbench-footer"><span>CRANE SAFETY / 研究与情景评估</span>
    <span>图谱情景概率未经实测校准，使用时请结合设备资料与现场复核。</span></div>''', unsafe_allow_html=True)
