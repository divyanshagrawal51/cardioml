"""src/export_model_card.py — export which ML model is used (type, params, metrics, importances) as JSON."""

from __future__ import annotations

import json

import joblib
import pandas as pd

from preprocess import OUT_DIR, ROOT

MODEL_DIR = ROOT / "models"
EXPORT_DIR = ROOT / "export"
TARGETS = ["CAD", "LAD", "LCX", "RCA"]

DESCRIPTION = {
    "CAD": "Overall coronary artery disease (>=1 of LAD/LCX/RCA stenotic)",
    "LAD": "Left Anterior Descending artery stenosis",
    "LCX": "Left Circumflex artery stenosis",
    "RCA": "Right Coronary Artery stenosis",
}


def main() -> None:
    results = pd.read_csv(OUT_DIR / "final_results.csv")
    EXPORT_DIR.mkdir(exist_ok=True)

    card = {
        "model_family": "RandomForestClassifier (scikit-learn) + sigmoid probability calibration",
        "library": "scikit-learn",
        "task": "Binary classification, one independent model per target (4 models)",
        "input": "Clinical, ECG, lab and echo features (LAD/LCX/RCA/Cath excluded to prevent target leakage)",
        "output": "Calibrated probability (0-1) per target; use probability for 3D colour mapping",
        "validation": "Nested 5-fold CV (inner 3-fold randomized hyperparameter search), out-of-fold metrics",
        "calibration": "CalibratedClassifierCV(method='sigmoid', cv=5)",
        "explainability": "SHAP TreeExplainer on the uncalibrated RandomForest",
        "note": (
            "Random forests have no weight matrix. 'feature_importances' (Gini importance, sums to 1) "
            "is the closest equivalent. For exact weights, call the API."
        ),
        "targets": {},
    }

    for t in TARGETS:
        b = joblib.load(MODEL_DIR / f"{t}.joblib")
        rf = b["rf"]
        feats = b["features"]

        imp = sorted(
            zip(feats, rf.feature_importances_), key=lambda kv: kv[1], reverse=True
        )

        m = results[(results["target"] == t) & (results["variant"] == "calibrated")].iloc[0]
        metrics = {
            k: float(m[k])
            for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "brier"]
        }

        card["targets"][t] = {
            "description": DESCRIPTION[t],
            "model": "RandomForestClassifier",
            "n_estimators": int(rf.n_estimators),
            "n_features": len(feats),
            "hyperparameters": {k: v for k, v in b["best_params"].items()},
            "decision_threshold": round(float(b["threshold"]), 2),
            "metrics_nested_cv_oof": {k: round(v, 3) for k, v in metrics.items()},
            "low_confidence": bool(metrics["roc_auc"] < 0.78),
            "feature_importances": {k: round(float(v), 5) for k, v in imp},
        }

    (EXPORT_DIR / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"Wrote {EXPORT_DIR / 'model_card.json'}")


if __name__ == "__main__":
    main()