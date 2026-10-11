"""src/train.py — CV comparison of baseline models for CAD / LAD / LCX / RCA."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from preprocess import OUT_DIR, RANDOM_STATE, build_targets, clean, get_xy, load_raw

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def get_models() -> dict:
    return {
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
            ),
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
    }


def main() -> None:
    df = build_targets(clean(load_raw()))
    X, y = get_xy(df)  # unscaled; scaling happens inside the pipelines

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

    rows = []
    for target in y.columns:
        for name, model in get_models().items():
            res = cross_validate(model, X, y[target], cv=cv, scoring=SCORING, n_jobs=1)
            row = {"target": target, "model": name}
            for metric in SCORING:
                row[metric] = res[f"test_{metric}"].mean()
            row["roc_auc_std"] = res["test_roc_auc"].std()
            rows.append(row)
            print(f"{target:4s} {name:14s} AUC={row['roc_auc']:.3f} F1={row['f1']:.3f}")

    results = pd.DataFrame(rows).round(3)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUT_DIR / "cv_results.csv", index=False)

    print("\n=== Full results ===")
    print(results.to_string(index=False))

    best = results.loc[results.groupby("target")["roc_auc"].idxmax(), ["target", "model", "roc_auc"]]
    print("\n=== Best model per target (by ROC-AUC) ===")
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()