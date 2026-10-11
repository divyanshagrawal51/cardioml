"""src/api.py — FastAPI backend: probabilities + SHAP explanations for CAD / LAD / LCX / RCA."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from preprocess import OUT_DIR, ROOT, build_targets, clean, get_xy, load_raw

MODEL_DIR = ROOT / "models"
TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LOW_CONFIDENCE_AUC = 0.78  # targets below this get a low-confidence flag in the UI

DISCLAIMER = (
    "For decision support and educational purposes only. Not a substitute for "
    "formal diagnostic imaging or clinical judgement."
)

STATE: dict = {}


def risk_level(p: float) -> str:
    return "low" if p < 0.33 else "moderate" if p < 0.66 else "high"


@asynccontextmanager
async def lifespan(app: FastAPI):
    df = build_targets(clean(load_raw()))
    X, _ = get_xy(df)

    features = json.loads((MODEL_DIR / "features.json").read_text())
    X = X[features]

    bundles = {t: joblib.load(MODEL_DIR / f"{t}.joblib") for t in TARGETS}
    # SHAP explains the uncalibrated RF; the displayed probability is the calibrated one
    explainers = {t: shap.TreeExplainer(b["rf"]) for t, b in bundles.items()}

    auc = {}
    res_path = OUT_DIR / "final_results.csv"
    if res_path.exists():
        r = pd.read_csv(res_path)
        r = r[r["variant"] == "calibrated"].set_index("target")
        auc = {t: float(r.loc[t, "roc_auc"]) for t in TARGETS if t in r.index}

    STATE.update(
        X=X,
        features=features,
        medians=X.median(),
        bundles=bundles,
        explainers=explainers,
        auc=auc,
    )
    yield


app = FastAPI(title="CardioML API", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    features: dict[str, float] = Field(
        default_factory=dict,
        description="Encoded feature values (see /schema). Missing features use the training median.",
    )


def _positive_shap(explainer: shap.TreeExplainer, row: pd.DataFrame) -> tuple[np.ndarray, float]:
    sv = explainer.shap_values(row)
    if isinstance(sv, list):  # older shap: list of per-class arrays
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:  # (n, features, classes)
        sv = sv[..., 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return sv[0], base


def _build_row(values: dict[str, float]) -> pd.DataFrame:
    unknown = [k for k in values if k not in STATE["features"]]
    if unknown:
        raise HTTPException(422, f"Unknown features: {unknown}. See /schema.")
    row = STATE["medians"].copy()
    for k, v in values.items():
        row[k] = v
    return pd.DataFrame([row[STATE["features"]]])


def _predict(row: pd.DataFrame, top_k: int) -> dict:
    out = {}
    for t in TARGETS:
        b = STATE["bundles"][t]
        p = float(b["model"].predict_proba(row)[0, 1])
        sv, base = _positive_shap(STATE["explainers"][t], row)

        order = np.argsort(-np.abs(sv))[:top_k]
        contribs = [
            {
                "feature": STATE["features"][i],
                "value": float(row.iloc[0, i]),
                "shap": float(sv[i]),
                "direction": "increases risk" if sv[i] > 0 else "decreases risk",
            }
            for i in order
        ]
        auc = STATE["auc"].get(t)
        out[t] = {
            "probability": round(p, 4),
            "risk_level": risk_level(p),
            "predicted_positive": bool(p >= b["threshold"]),
            "threshold": round(float(b["threshold"]), 2),
            "model_auc": None if auc is None else round(auc, 3),
            "low_confidence": bool(auc is not None and auc < LOW_CONFIDENCE_AUC),
            "base_value": round(base, 4),
            "contributions": contribs,
        }
    return {"disclaimer": DISCLAIMER, "overall_risk": out["CAD"]["probability"], "targets": out}


@app.get("/health")
def health():
    return {"status": "ok", "targets": TARGETS, "n_features": len(STATE["features"])}


@app.get("/schema")
def schema():
    """Feature list with default/min/max, for building the input form."""
    X = STATE["X"]
    return {
        "features": [
            {
                "name": c,
                "default": float(X[c].median()),
                "min": float(X[c].min()),
                "max": float(X[c].max()),
                "binary": bool(X[c].nunique() <= 2),
            }
            for c in STATE["features"]
        ]
    }


@app.get("/sample/{idx}")
def sample(idx: int):
    """A real patient row from the dataset (handy for demos)."""
    X = STATE["X"]
    if not 0 <= idx < len(X):
        raise HTTPException(404, f"idx must be 0..{len(X) - 1}")
    return {"features": {k: float(v) for k, v in X.iloc[idx].items()}}


@app.post("/predict")
def predict(req: PredictRequest, top_k: int = Query(10, ge=1, le=60)):
    row = _build_row(req.features)
    return _predict(row, top_k)