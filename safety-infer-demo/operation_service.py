"""Paired stochastic rollout of a reduced crane model. No graph-derived probabilities.

World X/Z small-angle pendulum coupled to two equivalent mast modal coordinates;
RK4, 0.02 s internal step. Parameters are research defaults, not device certification.
"""
from __future__ import annotations

import hashlib
import math
import json
from pathlib import Path
import numpy as np

from operation_models import OperationRequest
from digital_twin_service import FACTOR_NAMES

MODEL_VERSION = "paired-crane-1.0"
DT = .02
OUTPUT_STRIDE = 10
ACTIONS = [("maintain", "维持速度"), ("early_deceleration", "提前减速"),
           ("smooth_brake", "平缓制动"), ("immediate_brake", "立即制动")]
EVENT_NAMES = {"swing": "摆角超限", "moment": "塔身弯矩超限", "attachment": "附着轴力超限", "clearance": "安全距离不足"}


def _r(value):
    return round(float(value), 6)


def wilson(hits, n):
    p, z = hits / n, 1.95996398454
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [_r(max(0, center-half)), _r(min(1, center+half))]


def control_profile(action, t, omega0, c):
    """Finite, C1 velocity ramps; analytical angle integral, never reverse slew."""
    if action == "maintain":
        return omega0, omega0*t, np.zeros_like(omega0)
    duration = c.early_deceleration_s if action == "early_deceleration" else (
        c.smooth_braking_s if action == "smooth_brake" else c.immediate_braking_s)
    final = c.reduced_speed_fraction if action == "early_deceleration" else 0
    u = min(1, max(0, t/duration))
    fraction = 1 - (1-final)*(3*u*u-2*u*u*u)
    accel = -(1-final)*(6*u-6*u*u)/duration
    angle = duration*(u-(1-final)*(u**3-.5*u**4)) + final*max(0, t-duration)
    return omega0*fraction, omega0*angle, omega0*accel


def _ensemble(req):
    s, e, u, n = req.state, req.equipment, req.uncertainty, req.samples
    rng = np.random.default_rng(req.seed)
    uniform = lambda half: rng.uniform(-half, half, n)
    relative = lambda base: base * (1 + uniform(u.relative_parameter_error))
    params = {
        "length": relative(s.rope_length_m), "mass": relative(s.load_t*1000),
        "ei": relative(e.mast_ei_kn_m2), "freq": relative(e.mast_frequency_hz)*2*np.pi,
        "mast_damping": relative(e.mast_damping), "swing_damping": relative(e.swing_damping),
        "drag": relative(e.drag_coefficient),
        "phi": np.radians(s.slew_deg+uniform(u.slew_error_deg)),
        "omega": np.radians(s.slew_rate_deg_s+uniform(u.slew_rate_error_deg_s)),
        "wind_dir": np.radians(s.wind_direction_deg+uniform(u.wind_direction_error_deg)),
    }
    # Measurement errors are bounded uniforms; parameter multipliers are fixed per path.
    theta = np.radians(np.column_stack([s.swing_x_deg+uniform(u.swing_error_deg), s.swing_z_deg+uniform(u.swing_error_deg)]))
    vel = np.radians(np.column_stack([s.swing_rate_x_deg_s+uniform(u.swing_rate_error_deg_s), s.swing_rate_z_deg_s+uniform(u.swing_rate_error_deg_s)]))
    knots = int(req.horizon_s/.2)+1
    rho = math.exp(-.2/u.gust_correlation_s)
    gust = np.empty((knots, n))
    gust[0] = rng.normal(0, u.gust_std_m_s, n)
    for k in range(1, knots):
        gust[k] = rho*gust[k-1] + math.sqrt(1-rho*rho)*rng.normal(0, u.gust_std_m_s, n)
    bias = uniform(u.wind_bias_m_s)
    wind = np.maximum(0, s.wind_m_s + bias + np.clip(gust, -3*u.gust_std_m_s, 3*u.gust_std_m_s))
    er = np.column_stack([np.cos(params["phi"]), np.sin(params["phi"])])
    wd = np.column_stack([np.cos(params["wind_dir"]), np.sin(params["wind_dir"])])
    free = e.tower_height_m - (e.attachment_heights_m[-1] if e.attachment_heights_m else 0)
    static_moment = params["mass"][:, None]*.00981*s.radius_m*er
    tower_force = (.000613*wind[0]**2*params["drag"]*e.tower_wind_area_m2)[:, None]*wd
    prior = static_moment*free**2/(2*params["ei"][:, None]) + tower_force*free**3/(3*params["ei"][:, None])
    # Scalar Bayesian displacement updates; one snapshot cannot identify EI/frequency.
    posterior = prior.copy()
    stds, innovations = [], []
    for axis, observation in enumerate([s.mast_x_mm, s.mast_z_mm]):
        pv, mv = u.mast_prior_std_mm**2, u.mast_measurement_std_mm**2
        gain = pv/(pv+mv) if observation is not None else 0
        posterior[:, axis] += gain*((observation or 0)/1000-prior[:, axis])
        std = math.sqrt((1-gain)*pv)
        stds.append(std)
        innovations.append(None if observation is None else _r(observation - np.mean(prior[:, axis])*1000))
        posterior[:, axis] += np.clip(rng.normal(0, std/1000, n), -3*std/1000, 3*std/1000)
    state = np.column_stack([theta, vel, posterior, np.tile([s.mast_rate_x_mm_s/1000, s.mast_rate_z_mm_s/1000], (n, 1))])
    # Rope samples that cross the ground are outside the ensemble's validity.
    if np.any(params["length"]+e.payload_radius_m >= e.tower_height_m):
        raise ValueError("绳长不确定范围可能使吊物触地，请缩短绳长或减小参数误差")
    digest = hashlib.sha256(state.tobytes()+wind.tobytes()+b"".join(v.tobytes() for v in params.values())).hexdigest()[:16]
    estimation = {"swing_state": "手动观测值及有界测量误差", "mast_posterior_std_mm": [_r(x) for x in stds],
                  "mast_innovation_mm": innovations, "parameter_status": "参数使用先验范围；单帧输入不识别刚度和阻尼",
                  "wind_process": "0.2 s 相关高斯阵风，±3σ截断，负风速截为零"}
    return params, state, wind, digest, estimation


def _forces(req, p, state, t, action, wind):
    e, s = req.equipment, req.state
    omega, angle, alpha = control_profile(action, t, p["omega"], req.controls)
    phi = p["phi"]+angle
    er = np.column_stack([np.cos(phi), np.sin(phi)])
    et = np.column_stack([-np.sin(phi), np.cos(phi)])
    wd = np.column_stack([np.cos(p["wind_dir"]), np.sin(p["wind_dir"])])
    tower_wind = (.000613*wind**2*p["drag"]*e.tower_wind_area_m2)[:, None]*wd
    payload_wind = (.613*wind**2*p["drag"]*e.payload_wind_area_m2)[:, None]*wd
    # Tension includes a second-order swing-speed term. Horizontal suspension reaction.
    tension = p["mass"]*9.81*(1 + p["length"]*np.sum(state[:, 2:4]**2, axis=1)/9.81)
    reaction = tension[:, None]*state[:, :2]/1000
    lift = tension[:, None]*s.radius_m*er/1000
    moment_vector = lift + tower_wind*e.tower_height_m/2 + reaction*e.tower_height_m
    free = e.tower_height_m - (e.attachment_heights_m[-1] if e.attachment_heights_m else 0)
    equilibrium = (lift*free**2/(2*p["ei"][:, None])
                   + (tower_wind+reaction)*free**3/(3*p["ei"][:, None]))
    # Modal inertial moment on the unsupported section, relative to forced equilibrium.
    modal = (state[:, 4:6]-equilibrium)*2*p["ei"][:, None]/free**2
    moment = np.linalg.norm(moment_vector+modal, axis=1)
    attachments = len(e.attachment_heights_m)
    axial = moment/(e.attachment_lever_arm_m*attachments) if attachments else np.zeros(req.samples)
    pivot_accel = s.radius_m*(-omega[:, None]**2*er + alpha[:, None]*et)
    position = s.radius_m*er + state[:, 4:6] + p["length"][:, None]*state[:, :2]
    return equilibrium, pivot_accel, payload_wind, moment, axial, position, phi, omega


def _derivative(req, p, state, t, action, wind):
    equilibrium, pivot_accel, payload_wind, *_ = _forces(req, p, state, t, action, wind)
    mast_accel = p["freq"][:, None]**2*(equilibrium-state[:, 4:6]) - 2*p["mast_damping"][:, None]*p["freq"][:, None]*state[:, 6:8]
    w = np.sqrt(9.81/p["length"])
    swing_accel = (-w[:, None]**2*state[:, :2] - 2*p["swing_damping"][:, None]*w[:, None]*state[:, 2:4]
                   + (payload_wind/p["mass"][:, None]-pivot_accel-mast_accel)/p["length"][:, None])
    return np.column_stack([state[:, 2:4], swing_accel, state[:, 6:8], mast_accel])


def _band(values):
    lo, hi = np.quantile(values, [.05, .95])
    return {"mean": _r(np.mean(values)), "p05": _r(lo), "p95": _r(hi)}


def _constraints(req, action):
    s, e, c = req.state, req.equipment, req.controls
    reasons = []
    if s.load_t > e.rated_load_t or s.load_t*9.81*s.radius_m > e.rated_lift_moment_kn_m:
        reasons.append("当前吊重或静态起重力矩超过输入的设备额定值")
    free = e.tower_height_m - (e.attachment_heights_m[-1] if e.attachment_heights_m else 0)
    if free > e.max_free_height_m:
        reasons.append("当前自由高度超过输入限值")
    if action == "maintain":
        acceleration, jerk, stop = 0, 0, None
    else:
        duration = c.early_deceleration_s if action == "early_deceleration" else (c.smooth_braking_s if action == "smooth_brake" else c.immediate_braking_s)
        delta = (1-c.reduced_speed_fraction) if action == "early_deceleration" else 1
        # Check the maximum possible initial speed, not just its central estimate.
        speed = abs(s.slew_rate_deg_s) + req.uncertainty.slew_rate_error_deg_s
        acceleration, jerk = 1.5*speed*delta/duration, 6*speed*delta/duration**2
        stop = None if action == "early_deceleration" else duration
        emergency = action == "immediate_brake"
        if emergency and not c.emergency_enabled:
            reasons.append("未允许应急制动候选操作")
        if acceleration > (c.emergency_acceleration_deg_s2 if emergency else c.max_acceleration_deg_s2):
            reasons.append("制动所需峰值角减速度超过操作约束")
        if jerk > (c.emergency_jerk_deg_s3 if emergency else c.max_jerk_deg_s3):
            reasons.append("制动所需角加加速度超过操作约束")
    return {"feasible": not reasons, "reasons": reasons, "peak_acceleration_deg_s2": _r(acceleration),
            "peak_jerk_deg_s3": _r(jerk), "planned_stop_time_s": stop,
            "condition": "应急制动已授权且设备制动能力满足输入约束" if action == "immediate_brake" else "额定载荷、自由高度及正常操作加速度约束均满足"}


def _rollout(req, p, initial, wind_knots, action, label, fingerprint):
    state = initial.copy()
    n, steps = req.samples, int(req.horizon_s/DT)
    hits = np.zeros((n, 4), dtype=bool)
    peak_swing = np.zeros(n)
    peak_moment = np.zeros(n)
    peak_axial = np.zeros(n)
    min_clearance = np.full(n, np.inf)
    max_angle = 0
    series, paths = [], []
    for k in range(steps+1):
        t = k*DT
        j, f = min(int(t/.2), len(wind_knots)-2), (t/.2) % 1
        if k == steps:
            j, f = len(wind_knots)-2, 1
        wind = wind_knots[j]*(1-f)+wind_knots[j+1]*f
        eq, acc, pw, moment, axial, position, phi, omega = _forces(req, p, state, t, action, wind)
        swing = np.degrees(np.linalg.norm(state[:, :2], axis=1))
        max_angle = max(max_angle, float(np.max(swing)))
        clearance = np.linalg.norm(position-[req.obstacle.x_m, req.obstacle.z_m], axis=1)-req.obstacle.radius_m-req.equipment.payload_radius_m
        instantaneous = np.column_stack([swing > req.limits.swing_deg, moment > req.limits.mast_moment_kn_m,
            axial > req.limits.attachment_axial_kn, (clearance < req.limits.min_clearance_m) & req.obstacle.enabled])
        hits |= instantaneous
        peak_swing = np.maximum(peak_swing, swing)
        peak_moment = np.maximum(peak_moment, moment)
        peak_axial = np.maximum(peak_axial, axial)
        min_clearance = np.minimum(min_clearance, clearance)
        if k % OUTPUT_STRIDE == 0:
            any_hit, instant_hit = np.any(hits, axis=1), np.any(instantaneous, axis=1)
            point = {"time_s": _r(t), "slew_deg": _r(np.mean(np.degrees(phi))), "slew_rate_deg_s": _r(np.mean(np.degrees(omega))),
                "wind_m_s": _band(wind), "swing_deg": _band(swing), "swing_x_deg": _r(np.mean(np.degrees(state[:, 0]))),
                "swing_z_deg": _r(np.mean(np.degrees(state[:, 1]))), "mast_moment_kn_m": _band(moment), "attachment_axial_kn": _band(axial),
                "mast_x_mm": _band(state[:, 4]*1000), "mast_z_mm": _band(state[:, 5]*1000),
                "load_x_m": _band(position[:, 0]), "load_z_m": _band(position[:, 1]),
                "load_y_m": _band(req.equipment.tower_height_m-p["length"]*(1-.5*np.sum(state[:, :2]**2, axis=1))),
                "clearance_m": _band(clearance) if req.obstacle.enabled else None,
                "instantaneous_probability": _r(np.mean(instant_hit)), "cumulative_probability": _r(np.mean(any_hit)),
                "cumulative_ci95": wilson(int(any_hit.sum()), n),
                "event_probabilities": {key: _r(np.mean(hits[:, i])) for i, key in enumerate(EVENT_NAMES)}}
            series.append(point)
            paths.append(position[:24].copy())
        if k < steps:
            wm = wind_knots[j]*(1-min(1,f+DT/.4))+wind_knots[j+1]*min(1,f+DT/.4)
            we = wind_knots[j]*(1-min(1,f+DT/.2))+wind_knots[j+1]*min(1,f+DT/.2)
            k1 = _derivative(req, p, state, t, action, wind)
            k2 = _derivative(req, p, state+DT*k1/2, t+DT/2, action, wm)
            k3 = _derivative(req, p, state+DT*k2/2, t+DT/2, action, wm)
            k4 = _derivative(req, p, state+DT*k3, t+DT, action, we)
            state += DT*(k1+2*k2+2*k3+k4)/6
            if not np.isfinite(state).all() or np.max(np.abs(state[:, :2])) > .7:
                raise ValueError("推演超出小角度模型适用范围；请降低速度/风速或缩短预测时段")
    any_hit = np.any(hits, axis=1)
    probability = float(np.mean(any_hit))
    constraints = _constraints(req, action)
    c, speed = req.controls, abs(req.state.slew_rate_deg_s)
    progress = np.array([abs(control_profile(action, point["time_s"], np.array([math.radians(speed)]), c)[1][0])*180/math.pi for point in series])
    target = abs(c.remaining_slew_deg)
    reached = np.flatnonzero(progress >= target)
    completion = series[int(reached[0])]["time_s"] if len(reached) else None
    remaining = max(0, target-progress[-1])
    stopped = action in ("smooth_brake", "immediate_brake") and c.remaining_slew_deg != 0
    # Counterfactual resumption at the input initial speed is an explicit cost assumption.
    estimated_time = completion if completion is not None else req.horizon_s + remaining/max(speed, .01) + (c.restart_delay_s if stopped else 0)
    stop_cost = c.stop_cost if stopped else 0
    time_cost, risk_cost = estimated_time*c.time_cost_per_s, probability*c.risk_cost
    costs = {"completion_within_horizon_s": completion, "estimated_completion_s": _r(estimated_time),
             "remaining_slew_deg": _r(remaining), "time_cost": _r(time_cost), "stop_cost": _r(stop_cost),
             "risk_cost": _r(risk_cost), "total": _r(time_cost+stop_cost+risk_cost),
             "assumption": "未完成时按预测结束后恢复初始角速度估计；制动且任务未完成时增加重启等待，未模拟重启过程"}
    return {"id": action, "label": label, "ensemble_id": fingerprint, "series": series,
            "trajectory_samples_xz_m": np.round(np.stack(paths, axis=1), 4).tolist(),
            "risk": {"probability": _r(probability), "ci95": wilson(int(any_hit.sum()), n), "hits": int(any_hit.sum()),
                     "events": {key: {"probability": _r(np.mean(hits[:, i])), "ci95": wilson(int(hits[:, i].sum()), n)} for i, key in enumerate(EVENT_NAMES)},
                     "peak_instantaneous_probability": _r(max(x["instantaneous_probability"] for x in series))},
            "peaks": {"swing_deg": _band(peak_swing), "mast_moment_kn_m": _band(peak_moment), "attachment_axial_kn": _band(peak_axial),
                      "min_clearance_m": _band(min_clearance) if req.obstacle.enabled else None},
            "constraints": constraints, "costs": costs,
            "validity": {"max_sample_swing_deg": _r(max_angle), "small_angle_valid": max_angle <= 15}}, any_hit


def simulate_operations(req: OperationRequest, edges=None, graph_source="本地只读知识图谱 CSV"):
    params, state, wind, fingerprint, estimation = _ensemble(req)
    scenarios, hit_sets = [], []
    for action, label in ACTIONS:
        scenario, hits = _rollout(req, params, state, wind, action, label, fingerprint)
        scenarios.append(scenario)
        hit_sets.append(hits)
    knowledge = json.loads((Path(__file__).resolve().parent / "operation_knowledge.json").read_text(encoding="utf-8"))
    for scenario, rule in zip(scenarios, knowledge["measures"]):
        if scenario["id"] != rule["id"]:
            raise ValueError("本地措施知识规则与计算操作不一致")
        scenario["knowledge_rule"] = rule
    for scenario, hits in zip(scenarios, hit_sets):
        differences = hit_sets[0].astype(float)-hits.astype(float)
        # Paired standard error uses only discordant outcomes under shared realizations.
        delta = float(np.mean(differences))
        half = 1.96*float(np.std(differences, ddof=1))/math.sqrt(req.samples)
        scenario["comparison"] = {"risk_reduction_pp": _r(100*delta),
            "paired_delta_ci95_pp": [_r(100*max(-1,delta-half)), _r(100*min(1,delta+half))],
            "discordant_samples": int(np.sum(differences != 0)),
            "estimated_time_delta_s": _r(scenario["costs"]["estimated_completion_s"]-scenarios[0]["costs"]["estimated_completion_s"])}
    eligible = [s for s in scenarios if s["constraints"]["feasible"] and s["validity"]["small_angle_valid"]]
    # Upper confidence bound guards against zero-hit false assurance in small ensembles.
    budget_ok = [s for s in eligible if s["risk"]["ci95"][1] <= req.controls.risk_budget]
    chosen = min(budget_ok, key=lambda s: s["costs"]["total"]) if budget_ok else (
        min(eligible, key=lambda s: (s["risk"]["ci95"][1], s["costs"]["total"])) if eligible else None)
    status = "within_budget" if budget_ok else ("budget_unmet" if chosen else "no_feasible_action")
    recommendation = {"status": status, "action_id": chosen["id"] if chosen else None,
        "label": chosen["label"] if chosen else "没有满足操作约束的候选措施", "risk_budget": req.controls.risk_budget,
        "rule": "先筛选操作约束与模型适用范围；风险95%上界满足预算时最小化风险损失+时间+停机代价，否则仅展示最低风险候选供复核",
        "risk_reduction_pp": chosen["comparison"]["risk_reduction_pp"] if chosen else None,
        "text": (f"比较结果：{chosen['label']}。预测时段内任一超限概率 {chosen['risk']['probability']:.1%}，"
                  f"相对维持速度{'降低' if chosen['comparison']['risk_reduction_pp'] >= 0 else '增加'} {abs(chosen['comparison']['risk_reduction_pp']):.1f} 个百分点；"
                  f"预计任务时间 {chosen['costs']['estimated_completion_s']:.1f} s，停机代价 {chosen['costs']['stop_cost']:.1f}。") if chosen else "请调整载荷、设备限值或操作能力后重新推演。"}
    if status == "budget_unmet":
        recommendation["text"] += " 所有可行措施的风险95%上界均未达到预算，不能据此放行作业。"
    relevant = {"F028", "F022"}
    if req.equipment.tower_height_m-(req.equipment.attachment_heights_m[-1] if req.equipment.attachment_heights_m else 0) > req.equipment.max_free_height_m*.8:
        relevant.add("F017")
    chains = [{"factor_id": edge["factor_id"], "factor_name": FACTOR_NAMES.get(edge["factor_id"], edge["factor_id"]),
               "risk_id": edge["risk_id"], "risk_name": edge["risk_name"], "relationship": "CAUSES", "use": "解释关联，未参与超限概率计算"}
              for edge in (edges or []) if edge["factor_id"] in relevant]
    warnings = ["当前无设备试验数据：概率是给定模型与不确定范围下的超限频率，尚未实测校准。",
                "附着轴力采用等效力臂与平均分配，不包含单根撑杆几何、局部应力或疲劳验算。"]
    if any(not s["validity"]["small_angle_valid"] for s in scenarios):
        warnings.append("部分候选样本摆角超过15°，超出本小角度模型适用范围，已从建议筛选中排除。")
    if not req.equipment.attachment_heights_m:
        warnings.append("当前无附着，附着轴力事件停用；须核对自由高度与基础承载能力。")
    return {"model_version": MODEL_VERSION, "model_status": "研究演示 · 未实测校准", "config": req.model_dump(),
            "ensemble": {"id": fingerprint, "seed": req.seed, "samples": req.samples, "internal_dt_s": DT,
                         "common_random_numbers": True, "output_dt_s": DT*OUTPUT_STRIDE},
            "probability_definition": "P(在[0,H]至少一次摆角/塔身弯矩/附着轴力超限，或安全距离不足)，联合事件按同一路径直接计数",
            "uncertainty_note": "P05–P95为逐时刻预测分位区间；Wilson 95%区间仅表示蒙特卡洛计数误差，配对差值区间为近似正态区间，均不代表实测校准误差",
            "estimation": estimation, "graph_source": graph_source, "knowledge_chains": chains, "measure_knowledge_source": knowledge["source"],
            "event_names": EVENT_NAMES, "scenarios": scenarios, "recommendation": recommendation, "warnings": warnings}


def validate_predictions(rows):
    """Audit held-out, independent runs. Does not fit or claim a calibrated model."""
    probabilities, outcomes = [], []
    pairs = {"swing_deg": [], "mast_moment_kn_m": [], "attachment_axial_kn": []}
    intervals = {key: [] for key in pairs}
    for index, row in enumerate(rows, 1):
        if any(v is not None and not math.isfinite(float(v)) for v in row.values()):
            raise ValueError(f"第{index}行含非有限数值")
        p, outcome = row.get("risk_probability"), row.get("observed_exceedance")
        if (p is None) != (outcome is None):
            raise ValueError(f"第{index}行的risk_probability与observed_exceedance须成对填写")
        if p is not None:
            if not 0 <= p <= 1 or outcome not in (0, 1):
                raise ValueError(f"第{index}行概率须为0–1，超限结果须为0或1")
            probabilities.append(p)
            outcomes.append(outcome)
        for key in pairs:
            pred, measured = row.get("pred_"+key), row.get("observed_"+key)
            if (pred is None) != (measured is None):
                raise ValueError(f"第{index}行{key}的预测与实测值须成对填写")
            lo, hi = row.get("p05_"+key), row.get("p95_"+key)
            if (lo is None) != (hi is None) or (lo is not None and (measured is None or lo > hi)):
                raise ValueError(f"第{index}行{key}预测区间缺少实测值或上下界无效")
            if pred is not None:
                pairs[key].append(pred-measured)
                if lo is not None:
                    intervals[key].append(lo <= measured <= hi)
    if not probabilities and not any(pairs.values()):
        raise ValueError("没有可核验的概率/超限结果或预测/实测成对字段，请使用CSV模板列名")
    bins = []
    p, y = np.array(probabilities), np.array(outcomes)
    ece = 0
    for i in range(10):
        mask = (p >= i/10) & ((p < (i+1)/10) if i < 9 else (p <= 1))
        count = int(mask.sum())
        if count:
            avg, observed = float(p[mask].mean()), float(y[mask].mean())
            ece += count/max(1,len(p))*abs(avg-observed)
            bins.append({"lower": i/10, "upper": (i+1)/10, "count": count, "mean_predicted": _r(avg),
                         "observed_frequency": _r(observed), "ci95": wilson(int(y[mask].sum()), count)})
    return {"runs": len(probabilities), "brier_score": _r(np.mean((p-y)**2)) if len(p) else None,
            "ece": _r(ece) if len(p) else None, "reliability_bins": bins,
            "prediction_error": {key: {"count": len(values), "mae": _r(np.mean(np.abs(values))),
                "rmse": _r(np.sqrt(np.mean(np.array(values)**2))), "interval_count": len(intervals[key]),
                "p05_p95_coverage": _r(np.mean(intervals[key])) if intervals[key] else None} for key, values in pairs.items() if values},
            "status": "仅验证，不自动修改模型或概率", "note": "概率校验每行应为一个独立运行时段，采用相同限值、预测时长和联合超限定义；不可将相邻时刻当作独立试验。误差校验可单独提供逐时刻记录。"}
