"""src/train_final.py — nested-CV tuned RandomForest + calibration + thresholds; saves models."""

from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from preprocess import OUT_DIR, RANDOM_STATE, ROOT, build_targets, clean, get_xy, load_raw

MODEL_DIR = ROOT / "models"

PARAM_DIST = {
    "n_estimators": [200, 300],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_features": ["sqrt", 0.3, 0.5],
    "class_weight": ["balanced", "balanced_subsample", None],
}
N_ITER = 8  # random search budget per fit
INNER_FOLDS = 3
SEARCH_JOBS = -1  # parallel inside the search; set to 1 if workers cause trouble


def make_search(seed: int) -> RandomizedSearchCV:
    return RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=PARAM_DIST,
        n_iter=N_ITER,
        scoring="roc_auc",
        cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed),
        random_state=seed,
        n_jobs=SEARCH_JOBS,
        refit=True,
    )


def best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_true, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics_at(y_true: np.ndarray, proba: np.ndarray, thr: float) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "threshold": round(thr, 2),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def main() -> None:
    t_start = time.time()
    df = build_targets(clean(load_raw()))
    X, y = get_xy(df)  # unscaled: tree models don't need scaling

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    outer = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE + 1)
    rows, oof_all = [], {}

    for target in y.columns:
        yt = y[target].to_numpy()
        t0 = time.time()
        print(f"\n=== {target} (positive rate {yt.mean():.3f}) ===", flush=True)

        # 1) nested CV: tuning happens inside every outer training fold -> honest estimate
        print("  [1/4] nested CV ...", flush=True)
        oof_raw = cross_val_predict(
            make_search(RANDOM_STATE), X, yt, cv=outer, method="predict_proba", n_jobs=1
        )[:, 1]
        print(f"        done ({time.time() - t0:.0f}s)", flush=True)

        # 2) tune on full data to get final hyperparameters
        print("  [2/4] final tuning ...", flush=True)
        search = make_search(RANDOM_STATE).fit(X, yt)
        best_rf = search.best_estimator_
        print("        best params:", search.best_params_, flush=True)

        # 3) calibrated out-of-fold probabilities
        print("  [3/4] calibration CV ...", flush=True)
        calibrated = CalibratedClassifierCV(clone(best_rf), method="sigmoid", cv=5)
        oof_cal = cross_val_predict(
            calibrated, X, yt, cv=outer, method="predict_proba", n_jobs=1
        )[:, 1]

        thr = best_threshold(yt, oof_cal)  # threshold picked on OOF probs
        raw_m = metrics_at(yt, oof_raw, 0.5)
        cal_m = metrics_at(yt, oof_cal, thr)
        print(
            f"        raw  @0.50  AUC={raw_m['roc_auc']:.3f} F1={raw_m['f1']:.3f} "
            f"Brier={raw_m['brier']:.3f}\n"
            f"        cal  @{thr:.2f}  AUC={cal_m['roc_auc']:.3f} F1={cal_m['f1']:.3f} "
            f"Recall={cal_m['recall']:.3f} Brier={cal_m['brier']:.3f}",
            flush=True,
        )

        rows.append({"target": target, "variant": "raw", **raw_m})
        rows.append({"target": target, "variant": "calibrated", **cal_m})
        oof_all[f"{target}_oof_cal"] = oof_cal

        # 4) final deployable model: calibrated RF fit on all data
        print("  [4/4] saving model ...", flush=True)
        final = CalibratedClassifierCV(clone(best_rf), method="sigmoid", cv=5).fit(X, yt)
        joblib.dump(
            {
                "model": final,
                "rf": best_rf.fit(X, yt),  # uncalibrated RF, used for SHAP TreeExplainer
                "threshold": thr,
                "features": list(X.columns),
                "best_params": search.best_params_,
            },
            MODEL_DIR / f"{target}.joblib",
        )
        print(f"  {target} finished in {time.time() - t0:.0f}s", flush=True)

    results = pd.DataFrame(rows).round(3)
    results.to_csv(OUT_DIR / "final_results.csv", index=False)
    pd.DataFrame(oof_all).assign(**{t: y[t] for t in y.columns}).to_csv(
        OUT_DIR / "oof_predictions.csv", index=False
    )
    (MODEL_DIR / "features.json").write_text(json.dumps(list(X.columns), indent=2))

    print("\n=== Final results (nested-CV out-of-fold) ===")
    print(results.to_string(index=False))
    print(f"\nModels saved to {MODEL_DIR} | total time {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()