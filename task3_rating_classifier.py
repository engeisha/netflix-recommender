"""
Task 3 – Netflix Audience Rating Classification
================================================
Trains Decision Tree and Random Forest classifiers to predict a grouped
audience-rating category from content metadata + TF-IDF description features.

Rating groups
-------------
  Adult          : TV-MA, R, NC-17
  Teen           : TV-14, PG-13
  Older Kids     : TV-PG, TV-Y7, TV-Y7-FV
  Kids / Family  : TV-Y, TV-G, G, PG
  (rows with NR / UR / duration artefacts are dropped)

Outputs saved to project root
------------------------------
  rating_classifier.pkl   – best fitted sklearn Pipeline
  rating_preprocessor.pkl – dict with metadata needed by the Streamlit page
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "netflix_titles.csv"
CLF_OUT = ROOT / "rating_classifier.pkl"
PRE_OUT = ROOT / "rating_preprocessor.pkl"

# ── rating grouping ────────────────────────────────────────────────────────────
RATING_MAP: dict[str, str] = {
    "TV-MA": "Adult",
    "R":     "Adult",
    "NC-17": "Adult",
    "TV-14": "Teen",
    "PG-13": "Teen",
    "TV-PG": "Older Kids",
    "TV-Y7": "Older Kids",
    "TV-Y7-FV": "Older Kids",
    "TV-Y":  "Kids / Family",
    "TV-G":  "Kids / Family",
    "G":     "Kids / Family",
    "PG":    "Kids / Family",
}


# ── helpers ────────────────────────────────────────────────────────────────────
def parse_duration(dur: str, content_type: str) -> int:
    """Convert '90 min' → 90 or '2 Seasons' → 2; return 0 on failure."""
    try:
        parts = str(dur).split()
        return int(parts[0])
    except (ValueError, IndexError):
        return 0


def load_and_prepare(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop rows with missing / artefact ratings
    df = df[df["rating"].notna()]
    df = df[df["rating"].isin(RATING_MAP)]

    # map to grouped category
    df["rating_group"] = df["rating"].map(RATING_MAP)

    # fill missing text columns
    df["description"] = df["description"].fillna("")
    df["listed_in"]   = df["listed_in"].fillna("Unknown")
    df["country"]     = df["country"].fillna("Unknown")
    df["duration"]    = df.apply(
        lambda r: parse_duration(r["duration"], r["type"]), axis=1
    )

    return df


# ── build sklearn pipeline ─────────────────────────────────────────────────────
def build_pipeline(classifier) -> Pipeline:
    text_pipe = TfidfVectorizer(max_features=100, stop_words="english")

    cat_pipe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    num_cols  = ["release_year", "duration"]
    cat_cols  = ["type", "listed_in", "country"]
    text_col  = "description"

    preprocessor = ColumnTransformer(
        transformers=[
            ("num",  "passthrough", num_cols),
            ("cat",  cat_pipe,       cat_cols),
            ("text", text_pipe,      text_col),
        ],
        remainder="drop",
    )

    return Pipeline([("prep", preprocessor), ("clf", classifier)])


# ── training ───────────────────────────────────────────────────────────────────
def train(path: Path = DATA_PATH):
    print("Loading data …")
    df = load_and_prepare(path)
    print(f"  Dataset: {len(df)} rows | classes: {df['rating_group'].value_counts().to_dict()}")

    feature_cols = ["type", "release_year", "duration", "listed_in", "country", "description"]
    X = df[feature_cols]
    y = df["rating_group"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Decision Tree ──────────────────────────────────────────────────────────
    print("\nTuning Decision Tree …")
    dt_pipe = build_pipeline(DecisionTreeClassifier(random_state=42))
    dt_params = {
        "clf__max_depth":        [5, 10, 15, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf":  [1, 2, 4],
        "clf__criterion":         ["gini", "entropy"],
    }
    dt_search = RandomizedSearchCV(
        dt_pipe, dt_params, n_iter=10, cv=3, scoring="f1_weighted",
        n_jobs=-1, random_state=42, verbose=0,
    )
    dt_search.fit(X_train, y_train)
    dt_best = dt_search.best_estimator_
    dt_pred = dt_best.predict(X_test)
    dt_acc  = accuracy_score(y_test, dt_pred)
    dt_f1   = f1_score(y_test, dt_pred, average="weighted")
    print(f"  DT  → acc={dt_acc:.4f}  f1={dt_f1:.4f}")
    print(classification_report(y_test, dt_pred))

    # ── Random Forest ──────────────────────────────────────────────────────────
    print("Tuning Random Forest …")
    rf_pipe = build_pipeline(RandomForestClassifier(random_state=42, n_jobs=-1))
    rf_params = {
        "clf__n_estimators":      [100, 200],
        "clf__max_depth":         [10, 20],
        "clf__min_samples_split": [2, 5],
        "clf__min_samples_leaf":  [1, 2],
        "clf__max_features":      ["sqrt", "log2"],
    }
    rf_search = RandomizedSearchCV(
        rf_pipe, rf_params, n_iter=8, cv=3, scoring="f1_weighted",
        n_jobs=-1, random_state=42, verbose=0,
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_pred = rf_best.predict(X_test)
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_f1   = f1_score(y_test, rf_pred, average="weighted")
    print(f"  RF  → acc={rf_acc:.4f}  f1={rf_f1:.4f}")
    print(classification_report(y_test, rf_pred))

    # ── pick best ─────────────────────────────────────────────────────────────
    if rf_f1 >= dt_f1:
        best_model, best_name = rf_best, "Random Forest"
        best_acc, best_f1     = rf_acc, rf_f1
    else:
        best_model, best_name = dt_best, "Decision Tree"
        best_acc, best_f1     = dt_acc, dt_f1

    print(f"\n✓ Best model: {best_name}  (acc={best_acc:.4f}  f1={best_f1:.4f})")

    # ── save ──────────────────────────────────────────────────────────────────
    joblib.dump(best_model, CLF_OUT)

    # metadata consumed by the Streamlit page
    meta = {
        "feature_cols":    feature_cols,
        "rating_map":      RATING_MAP,
        "classes":         sorted(y.unique().tolist()),
        "best_model_name": best_name,
        "metrics": {
            "Decision Tree":  {"accuracy": round(dt_acc, 4), "f1": round(dt_f1, 4)},
            "Random Forest":  {"accuracy": round(rf_acc, 4), "f1": round(rf_f1, 4)},
        },
        # derive unique listed_in and country values for the UI
        "listed_in_options": sorted(df["listed_in"].unique().tolist()),
        "country_options":   sorted(df["country"].unique().tolist()),
    }
    joblib.dump(meta, PRE_OUT)

    print(f"Saved → {CLF_OUT}")
    print(f"Saved → {PRE_OUT}")
    return best_model, meta


if __name__ == "__main__":
    train()
