"""Illustrative, reduced-order tower-crane digital twin.

The mechanics and noisy-OR link strengths are scenario estimates, not a
structural design check or calibrated accident frequencies.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path


FACTOR_IDS = ("F015", "F016", "F017", "F022", "F028", "F033")
FACTOR_NAMES = {
    "F015": "顶升工序失衡", "F016": "附着连接缺陷", "F017": "自由高度超限",
    "F022": "超载或超力矩", "F028": "不利天气", "F033": "拆卸顺序失控",
}
EVENTS = {"normal": "正常吊装", "jacking": "顶升", "attach": "安装附着", "dismantle": "拆卸"}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def bundled_graph_edges() -> list[dict]:
    """Read-only snapshot for an explicitly identified offline demonstration."""
    root = Path(__file__).resolve().parent / "Knowledge Graph"
    with (root / "SafetyRisk.csv").open(encoding="utf-8-sig", newline="") as file:
        names = {row["id"]: row["name"] for row in csv.DictReader(file)}
    with (root / "CAUSES.csv").open(encoding="utf-8-sig", newline="") as file:
        return [
            {"factor_id": row["start_id"], "risk_id": row["end_id"],
             "risk_name": names.get(row["end_id"], row["end_id"]),
             "trigger_prob": float(row["trigger_prob"]),
             "influence": float(row["influence"]), "confidence": float(row["confidence"])}
            for row in csv.DictReader(file) if row["start_id"] in FACTOR_IDS
        ]


def _round(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


def _phase(t: float, start: float = 0.2, end: float = 0.75) -> float:
    return clamp((t - start) / (end - start))


def _risk_probs(factors: dict[str, float], edges: list[dict]) -> list[dict]:
    # Independent-cause noisy-OR with zero leak. Link values are graph-derived
    # illustrative strengths, not statistically calibrated CPT entries.
    risks: dict[str, dict] = {}
    for edge in edges:
        fid = edge["factor_id"]
        if fid not in factors:
            continue
        rid = edge["risk_id"]
        entry = risks.setdefault(rid, {"risk_id": rid, "risk_name": edge["risk_name"], "survival": 1.0})
        link = clamp(float(edge["trigger_prob"]) * float(edge["influence"]) * float(edge["confidence"]))
        entry["survival"] *= 1 - clamp(factors[fid]) * link
    return sorted(
        ({"risk_id": v["risk_id"], "risk_name": v["risk_name"],
          "prob": _round(1 - v["survival"], 5)} for v in risks.values()),
        key=lambda x: x["prob"], reverse=True,
    )


def _sample(config: dict, event: str, time_s: int, horizon_s: int, edges: list[dict]) -> dict:
    t = time_s / horizon_s
    h0 = config["tower_height_m"]
    attachments = [a for a in config["attachment_heights_m"] if a < h0]
    transition = _phase(t)
    if event == "jacking":
        height = h0 + 3 * transition
    elif event == "dismantle":
        height = h0 - 3 * transition
        if t >= 0.45 and attachments:
            attachments.pop()
    else:
        height = h0
    if event == "attach" and t >= 0.55:
        new_brace = max(0.0, h0 - 6.0)
        if not attachments or new_brace > attachments[-1] + 1:
            attachments.append(new_brace)

    attachments = [a for a in attachments if a < height]

    free_height = height - (attachments[-1] if attachments else 0.0)
    load_kn = config["load_t"] * 9.81
    radius = config["radius_m"]
    # Demonstration gust envelope over the 60 s horizon (12% peak increase).
    wind = config["wind_m_s"] * (1 + 0.12 * math.sin(math.pi * t))
    # q = 0.613 v² N/m²; area and drag are editable demonstration parameters.
    wind_kn = 0.000613 * wind**2 * config["drag_coefficient"] * config["projected_area_m2"]
    lift_moment = load_kn * radius
    wind_moment = wind_kn * height / 2
    mast_moment = lift_moment + wind_moment
    # Cantilever end moment + lateral point load on the unsupported segment.
    ei = config["mast_ei_kn_m2"]
    displacement_mm = 1000 * (lift_moment * free_height**2 / (2 * ei)
                               + wind_kn * free_height**3 / (3 * ei))
    attach_axial_kn = (mast_moment / (config["attachment_lever_arm_m"] * len(attachments))
                       if attachments else 0.0)
    jib_util = lift_moment / config["rated_moment_kn_m"]
    mast_util = mast_moment / config["mast_moment_capacity_kn_m"]
    attach_util = attach_axial_kn / config["attachment_capacity_kn"] if attachments else 0.0
    load_util = config["load_t"] / config["rated_load_t"]
    max_util = max(jib_util, mast_util, attach_util, load_util)
    free_util = free_height / config["max_free_height_m"]
    factors = {
        "F022": clamp((max(jib_util, load_util) - 0.55) / 0.7),
        "F028": clamp((wind - 8) / 18),
        "F017": clamp((free_util - 0.65) / 0.7),
        "F015": (0.18 + 0.35 * math.sin(math.pi * transition)) if event == "jacking" else 0.0,
        "F016": (0.16 + 0.27 * math.sin(math.pi * transition)) if event == "attach" else 0.0,
        "F033": (0.2 + 0.34 * math.sin(math.pi * transition)) if event == "dismantle" else 0.0,
    }
    factors = {key: _round(clamp(value), 5) for key, value in factors.items()}
    risks = _risk_probs(factors, edges)
    overall_risk = 1 - math.prod(1 - item["prob"] for item in risks)
    components = [
        {"id": "TC-MAST", "name": "塔身", "force_kn": _round(wind_kn), "moment_kn_m": _round(mast_moment), "utilization": _round(mast_util)},
        {"id": "TC-JIB", "name": "起重臂", "force_kn": _round(load_kn), "moment_kn_m": _round(lift_moment), "utilization": _round(jib_util)},
        {"id": "TC-HOIST", "name": "起升及吊物", "force_kn": _round(load_kn), "moment_kn_m": _round(lift_moment), "utilization": _round(load_util)},
        {"id": "TC-BASE", "name": "基础（示意）", "force_kn": _round(load_kn), "moment_kn_m": _round(mast_moment), "utilization": _round(mast_util)},
    ]
    components.extend(
        {"id": f"TC-ATT-{i+1}", "name": f"第{i+1}道附着", "force_kn": _round(attach_axial_kn),
         "moment_kn_m": 0.0, "utilization": _round(attach_util), "elevation_m": _round(a)}
        for i, a in enumerate(attachments)
    )
    planned_slew = config["slew_deg"] + 20 * t
    return {
        "time_s": time_s, "tower_height_m": _round(height), "wind_m_s": _round(wind), "attachments_m": [_round(a) for a in attachments],
        "free_height_m": _round(free_height), "slew_deg": _round(planned_slew), "load_x_m": _round(radius * math.cos(math.radians(planned_slew))),
        "load_z_m": _round(radius * math.sin(math.radians(planned_slew))),
        "displacement_mm": _round(displacement_mm), "mast_moment_kn_m": _round(mast_moment),
        "jib_moment_kn_m": _round(lift_moment), "attachment_axial_kn": _round(attach_axial_kn),
        "max_utilization": _round(max_util), "risk_probability": _round(overall_risk, 5),
        "factor_activations": factors, "risks": risks, "components": components,
    }


def simulate_twin(config: dict, edges: list[dict], graph_source: str) -> dict:
    event = config["event"]
    horizon = 60
    times = range(0, horizon + 1, 5)
    variants = [
        ("baseline", "保持原方案", config),
        ("lower_load", "减小吊重 25%", {**config, "load_t": config["load_t"] * 0.75}),
        ("shorter_radius", "缩小幅度 20%", {**config, "radius_m": config["radius_m"] * 0.8}),
    ]
    scenarios = []
    for key, label, params in variants:
        series = [_sample(params, event, second, horizon, edges) for second in times]
        scenarios.append({"id": key, "label": label, "load_t": _round(params["load_t"]),
                          "radius_m": _round(params["radius_m"]), "series": series,
                          "future": series[-1]})
    baseline = scenarios[0]
    initial = baseline["series"][0]
    mappings = [
        {"indicator_id": "I006", "name": "塔身垂直度偏差（等效）", "value": _round(initial["displacement_mm"] / config["tower_height_m"]), "unit": "‰", "kind": "derived"},
        {"indicator_id": "I014", "name": "载荷力矩利用率", "value": _round(initial["jib_moment_kn_m"] / config["rated_moment_kn_m"] * 100), "unit": "%", "kind": "derived"},
        {"indicator_id": "I015", "name": "额定载荷利用率", "value": _round(config["load_t"] / config["rated_load_t"] * 100), "unit": "%", "kind": "derived"},
        {"indicator_id": "I016", "name": "风速", "value": config["wind_m_s"], "unit": "m/s", "kind": "input"},
        {"indicator_id": "I034", "name": "自由高度利用率", "value": _round(initial["free_height_m"] / config["max_free_height_m"] * 100), "unit": "%", "kind": "derived"},
    ]
    return {"model_status": "示范参数与简化力学；风险概率未经现场数据校准", "graph_source": graph_source,
            "event": event, "event_name": EVENTS[event], "horizon_s": horizon,
            "indicator_mappings": mappings,
            "factor_names": FACTOR_NAMES, "scenarios": scenarios}
