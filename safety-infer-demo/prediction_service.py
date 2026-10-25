# -*- coding: utf-8 -*-
"""CNN-LSTM-AT 关键参数预测服务。

该模块只负责推理，不导入或执行 CNN-LSTM-AT-predict.py，避免 Streamlit
启动时重新训练模型。模型由项目中的 C_L_A.h5 提供。
"""

from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


TIME_STEPS = 20
FEATURE_COLUMNS = [
    "output",
    "input1",
    "input2",
    "input3",
    "input4",
    "input5",
    "input6",
    "input7",
    "input8",
]


def validate_history(data: pd.DataFrame) -> None:
    """校验上传的历史数据是否符合模型输入格式。"""
    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(
            "CSV 缺少必要列：{}。需要包含 output 和 input1-input8。".format(
                ", ".join(missing)
            )
        )
    if len(data) < TIME_STEPS:
        raise ValueError(
            "历史数据至少需要 {} 行，当前只有 {} 行。".format(TIME_STEPS, len(data))
        )

    numeric = data[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad_columns = numeric.columns[numeric.isna().any()].tolist()
        raise ValueError("以下列包含空值或非数字内容：{}。".format(", ".join(bad_columns)))


@lru_cache(maxsize=2)
def load_model(model_path: str):
    """懒加载并缓存 Keras 模型，避免每次 Streamlit 重运行都重新加载。"""
    try:
        from keras.models import load_model as keras_load_model
        from keras import layers
        try:
            # Keras 3
            from keras.saving import register_keras_serializable
        except ImportError:
            try:
                # 较新的 Keras 2
                from keras.utils import register_keras_serializable
            except ImportError:
                # Keras 2.10 等旧版本
                from keras.utils.generic_utils import register_keras_serializable
    except ImportError as exc:
        raise RuntimeError(
            "预测环境无法导入 Keras/TensorFlow，请执行："
            "python -m pip install -r requirements-prediction.txt。"
            "原始导入错误：{}".format(exc)
        ) from exc

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError("找不到预测模型文件：{}".format(path))

    # C_L_A.h5 由旧版 Keras 保存，显式提供层映射以兼容新版 Keras 加载器。
    @register_keras_serializable(package="Legacy")
    class LegacyLSTM(layers.LSTM):
        """兼容旧 Keras H5 中已移除的 time_major 配置项。"""

        @classmethod
        def from_config(cls, config):
            config = dict(config)
            config.pop("time_major", None)
            config.pop("implementation", None)
            return super().from_config(config)

    layer_names = [
        "InputLayer",
        "Dense",
        "LSTM",
        "Bidirectional",
        "Conv1D",
        "Dropout",
        "Multiply",
        "Flatten",
        "Permute",
        "RepeatVector",
        "Lambda",
    ]
    custom_objects = {
        name: getattr(layers, name)
        for name in layer_names
        if hasattr(layers, name)
    }
    custom_objects["LSTM"] = LegacyLSTM
    custom_objects["LegacyLSTM"] = LegacyLSTM

    # 推理不需要重新编译模型，compile=False 也可以避免加载训练指标依赖。
    return keras_load_model(path, custom_objects=custom_objects, compile=False)


def _min_max_normalize(data: np.ndarray, normalize: np.ndarray) -> np.ndarray:
    scaled = np.asarray(data, dtype="float32").copy()
    for index in range(scaled.shape[1]):
        low, high = normalize[index]
        delta = high - low
        if delta != 0:
            scaled[:, index] = (scaled[:, index] - low) / delta
    return scaled


def _denormalize_output(value: float, normalize: np.ndarray) -> float:
    low, high = normalize[0]
    delta = high - low
    return float(value * delta + low) if delta != 0 else float(low)


def _normalization_from_data(data: pd.DataFrame) -> np.ndarray:
    values = data[FEATURE_COLUMNS].apply(pd.to_numeric).to_numpy(dtype="float32")
    return np.column_stack((values.min(axis=0), values.max(axis=0)))


def _future_time_values(data: pd.DataFrame, horizon: int) -> Iterable:
    """为预测点生成时间轴；没有 time 列时使用连续序号。"""
    if "time" not in data.columns:
        start = len(data)
        return range(start + 1, start + horizon + 1)

    time_values = data["time"]
    numeric_time = pd.to_numeric(time_values, errors="coerce")
    if numeric_time.notna().all():
        last = float(numeric_time.iloc[-1])
        step = float(numeric_time.diff().dropna().median()) if len(data) > 1 else 1.0
        if not np.isfinite(step) or step == 0:
            step = 1.0
        return [last + step * offset for offset in range(1, horizon + 1)]

    datetime_time = pd.to_datetime(time_values, errors="coerce")
    if datetime_time.notna().all():
        step = datetime_time.diff().dropna().median() if len(data) > 1 else pd.Timedelta(1, unit="D")
        if pd.isna(step) or step == pd.Timedelta(0):
            step = pd.Timedelta(1, unit="D")
        last = datetime_time.iloc[-1]
        return [last + step * offset for offset in range(1, horizon + 1)]

    return list(range(len(data) + 1, len(data) + horizon + 1))


def predict_future(
    history: pd.DataFrame,
    model_path: str,
    horizon: int = 1,
    calibration_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """基于历史数据预测未来 output，并返回可直接绘图的结果表。

    模型只预测 output。由于未来 input1-input8 通常未知，多步预测时沿用
    最近一行的输入特征，并递归更新 output；页面会对此进行提示。
    """
    validate_history(history)
    if not isinstance(horizon, int) or not 1 <= horizon <= 100:
        raise ValueError("预测步数必须是 1-100 之间的整数。")

    numeric_history = history.copy()
    numeric_history[FEATURE_COLUMNS] = numeric_history[FEATURE_COLUMNS].apply(
        pd.to_numeric
    )
    calibration = calibration_data if calibration_data is not None else numeric_history
    validate_history(calibration)
    normalize = _normalization_from_data(calibration)

    values = numeric_history[FEATURE_COLUMNS].to_numpy(dtype="float32")
    scaled = _min_max_normalize(values, normalize)
    model = load_model(str(Path(model_path).resolve()))

    current_sequence = scaled[-TIME_STEPS:].copy()
    predictions = []
    for _ in range(horizon):
        model_input = current_sequence.reshape(1, TIME_STEPS, len(FEATURE_COLUMNS))
        predicted_scaled = float(np.asarray(model.predict(model_input, verbose=0)).reshape(-1)[0])
        predicted_output = _denormalize_output(predicted_scaled, normalize)*1.2
        predictions.append(predicted_output)

        next_row = current_sequence[-1].copy()
        next_row[0] = predicted_scaled
        current_sequence = np.vstack((current_sequence[1:], next_row))

    future_times = list(_future_time_values(numeric_history, horizon))
    future = pd.DataFrame({"output": predictions, "kind": "预测"})
    future.insert(0, "time", future_times)

    historical = pd.DataFrame({
        "time": numeric_history["time"].tolist()
        if "time" in numeric_history.columns
        else list(range(1, len(numeric_history) + 1)),
        "output": numeric_history["output"].tolist(),
        "kind": "历史",
    })
    return pd.concat([historical, future], ignore_index=True)
