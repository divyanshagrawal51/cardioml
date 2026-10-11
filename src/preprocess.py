"""src/preprocess.py — Z-Alizadeh Sani (extension) preprocessing for cardioml."""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = DATA_DIR / "processed"

TARGETS = ["CAD", "LAD", "LCX", "RCA"]
LEAK_COLS = ["CAD", "LAD", "LCX", "RCA", "Cath"]

RANDOM_STATE = 42
TEST_SIZE = 0.2

TEXT = ["object", "string"]  # pandas 3 uses 'str' dtype, pandas 2 uses 'object'
VHD_MAP = {"n": 0, "mild": 1, "moderate": 2, "severe": 3}
POSITIVE = {"cad", "stenotic", "stenosis", "y", "yes", "true", "1"}


# ---------- loading ----------
def find_data() -> Path:
    files = sorted(
        p
        for p in DATA_DIR.glob("*")
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"} and not p.name.startswith("~$")
    )
    # prefer the extension dataset (xlsx) over the old csv
    files.sort(key=lambda p: p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No data file found in {DATA_DIR}")
    return files[0]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or find_data()
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    print(f"Loaded {path.name}: {df.shape}")
    return df


# ---------- cleaning ----------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # strip whitespace in string columns
    for c in df.select_dtypes(include=TEXT).columns:
        df[c] = df[c].astype(str).str.strip()

    # Sex ("Fmale" is a typo in the original dataset)
    if not pd.api.types.is_numeric_dtype(df["Sex"]):
        df["Sex"] = df["Sex"].str.lower().map({"male": 1, "fmale": 0, "female": 0})

    # VHD: ordinal severity
    if "VHD" in df.columns and not pd.api.types.is_numeric_dtype(df["VHD"]):
        df["VHD"] = df["VHD"].str.lower().map(VHD_MAP)
        if df["VHD"].isna().any():
            warnings.warn("Unmapped VHD values found — check VHD_MAP")

    # remaining Y/N columns -> 1/0 (target cols handled in build_targets)
    skip = set(LEAK_COLS)
    for c in df.select_dtypes(include=TEXT).columns:
        if c in skip:
            continue
        vals = set(df[c].dropna().str.upper().unique())
        if vals <= {"Y", "N"}:
            df[c] = df[c].str.upper().map({"Y": 1, "N": 0})

    return df.drop_duplicates().reset_index(drop=True)


# ---------- targets ----------
def _binarize(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (s > 0).astype(int)
    return s.astype(str).str.strip().str.lower().isin(POSITIVE).astype(int)


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # CAD: use the column if present, otherwise derive from Cath
    if "CAD" in df.columns:
        df["CAD"] = _binarize(df["CAD"])
    elif "Cath" in df.columns:
        df["CAD"] = _binarize(df["Cath"])
    else:
        raise KeyError("Need 'CAD' or 'Cath' column")

    for t in ("LAD", "LCX", "RCA"):
        if t in df.columns:
            df[t] = _binarize(df[t])
        else:
            warnings.warn(f"'{t}' not found in dataset — skipping.")

    return df


# ---------- features ----------
def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [t for t in TARGETS if t in df.columns]
    y = df[targets].copy()
    X = df.drop(columns=[c for c in LEAK_COLS if c in df.columns])

    # anything still non-numeric -> one-hot
    obj = X.select_dtypes(include=TEXT).columns.tolist()
    if obj:
        print("One-hot encoding leftover text columns:", obj)
        X = pd.get_dummies(X, columns=obj, drop_first=True, dtype=int)

    # median-impute any leftover NaNs
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def split_and_scale(X: pd.DataFrame, y: pd.DataFrame):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y["CAD"],
    )

    # scale continuous cols only (binary/ordinal flags stay as-is); fit on train only
    cont = [c for c in X.columns if X[c].nunique() > 6]
    scaler = StandardScaler().fit(X_tr[cont])

    X_tr, X_te = X_tr.copy(), X_te.copy()
    X_tr[cont] = scaler.transform(X_tr[cont])
    X_te[cont] = scaler.transform(X_te[cont])
    return X_tr, X_te, y_tr, y_te, scaler, cont


# ---------- main ----------
def main() -> None:
    df = build_targets(clean(load_raw()))
    X, y = get_xy(df)
    X_tr, X_te, y_tr, y_te, scaler, cont = split_and_scale(X, y)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr.to_csv(OUT_DIR / "X_train.csv", index=False)
    X_te.to_csv(OUT_DIR / "X_test.csv", index=False)
    y_tr.to_csv(OUT_DIR / "y_train.csv", index=False)
    y_te.to_csv(OUT_DIR / "y_test.csv", index=False)

    joblib.dump(
        {"scaler": scaler, "scaled_cols": cont, "features": list(X.columns)},
        OUT_DIR / "scaler.joblib",
    )

    print(f"Features: {X.shape[1]} | train: {len(X_tr)} | test: {len(X_te)}")
    print("Targets:", list(y.columns))
    print(y.mean().round(3).to_string())


if __name__ == "__main__":
    main()