# -*- coding: utf-8 -*-
"""独立的 CNN-LSTM-AT 预测 API 服务。"""

import io
import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from prediction_service import TIME_STEPS, predict_future


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "C_L_A.h5"
CALIBRATION_PATH = BASE_DIR / "inputdata.csv"

app = FastAPI(title="CNN-LSTM-AT Prediction API", version="1.0.0")


def _records(frame: pd.DataFrame):
    """将 DataFrame 转为可被 JSON 序列化的记录。"""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


@app.get("/health")
def health():
    return {"status": "ok", "service": "prediction"}


@app.post("/api/v1/predict")
async def predict(
    file: UploadFile = File(...),
    horizon: int = Query(1, ge=1, le=100),
):
    """接收历史 CSV，返回历史数据和未来 output 预测结果。"""
    if file.filename and not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="请上传 CSV 文件")

    try:
        content = await file.read()
        history = pd.read_csv(io.BytesIO(content))
        calibration = None
        if CALIBRATION_PATH.exists():
            calibration = pd.read_csv(CALIBRATION_PATH)
            # 训练脚本使用 inputdata.csv 前 80% 计算归一化参数。
            train_rows = max(TIME_STEPS, int(len(calibration) * 0.8))
            calibration = calibration.iloc[:train_rows].copy()

        result = predict_future(
            history=history,
            model_path=str(MODEL_PATH),
            horizon=horizon,
            calibration_data=calibration,
        )
        future = result[result["kind"] == "预测"]
        return {
            "code": 0,
            "message": "ok",
            "data": {
                "rows": _records(result),
                "future_rows": _records(future),
                "horizon": horizon,
                "time_steps": TIME_STEPS,
            },
        }
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="预测服务执行失败：{}".format(exc)) from exc
