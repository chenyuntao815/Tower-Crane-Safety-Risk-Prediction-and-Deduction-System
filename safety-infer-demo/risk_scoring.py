"""依据塔吊图谱 HAS_INDICATOR 关系阈值计算指标危险度。"""

import math
from typing import Optional


def clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def linear_segment(value, x0, x1, y0, y1):
    return y0 + (value - x0) / (x1 - x0) * (y1 - y0)


def score_by_indicator_name(
    indicator_name: str,
    value: float,
    normal_t: Optional[float],
    warning_t: Optional[float],
    danger_t: Optional[float],
    direction: str,
) -> float:
    """返回 0-1 危险度；参数名保留用于兼容现有推演调用。"""
    if normal_t is None or warning_t is None or danger_t is None:
        raise ValueError(f"指标 {indicator_name} 缺少评分阈值")
    if not all(math.isfinite(float(v)) for v in (value, normal_t, warning_t, danger_t)):
        raise ValueError(f"指标 {indicator_name} 的输入或阈值不是有限数")
    if direction in ("negative", "higher_better"):
        if not normal_t > warning_t > danger_t:
            raise ValueError(f"指标 {indicator_name} 的下降型阈值顺序错误")
        if value >= normal_t:
            return 0.0
        if value >= warning_t:
            return clip(linear_segment(value, normal_t, warning_t, 0.0, 0.6))
        if value >= danger_t:
            return clip(linear_segment(value, warning_t, danger_t, 0.6, 1.0))
        return 1.0
    if direction in ("positive", "lower_better"):
        if not normal_t < warning_t < danger_t:
            raise ValueError(f"指标 {indicator_name} 的上升型阈值顺序错误")
        if value <= normal_t:
            return 0.0
        if value <= warning_t:
            return clip(linear_segment(value, normal_t, warning_t, 0.0, 0.6))
        if value <= danger_t:
            return clip(linear_segment(value, warning_t, danger_t, 0.6, 1.0))
        return 1.0
    raise ValueError(f"指标 {indicator_name} 的风险方向无效: {direction}")
