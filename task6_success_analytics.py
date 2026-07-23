"""
Task 6 (Advanced) – Netflix Content Success Analytics Engine
=============================================================
Step 1 : Advanced feature engineering
Step 2 : Build multiple ML models (Logistic Regression, Random Forest,
         Gradient Boosting, XGBoost-style Extra Trees)
Step 3 : Compare model performance (accuracy, F1, ROC-AUC)
Step 4 : Generate automated insights (feature importance, genre/country
         success rates, temporal trends)
Step 5 : Save all artefacts for the Streamlit visual report page

Success proxy
-------------
A title is labelled "High Engagement" (1) if it satisfies ≥2 of:
  • Rating is TV-MA or R  (mature = broader streaming audience)
  • Added in 2019-2021    (recent Netflix originals era)
  • Description length > median
  • Genres include Dramas, International or Thriller/Crime keywords
This gives a balanced, data-driven binary target without external scraping.

Saved artefacts
---------------
  success_models.pkl    – dict of fitted pipelines
  success_data.pkl      – all insight tables + metrics + feature importance
"""

from __future__ import annotations
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parent
DATA_PATH  = ROOT / "netflix_titles.csv"
MODELS_OUT = ROOT / "success_models.pkl"
DATA_OUT   = ROOT / "success_data.pkl"

BAD_RATINGS = {"74 min", "84 min", "66 min"}
HIGH_ENG_RATINGS  = {"TV-MA", "R", "TV-14", "PG-13"}
RECENT_YEARS      = set(range(2018, 2022))
HIGH_ENG_GENRES   = {
    "Dramas", "International Movies", "International TV Shows",
    "Thrillers", "Crime TV Shows", "TV Dramas", "Action & Adventure",
}


# ── Step 1: Feature Engineering ───────────────────────────────────────────────
def parse_duration(row: pd.Series) -> float:
    try:
        return float(str(row["duration"]).split()[0])
    except Exception:
        return 0.0


def build_success_label(df: pd.DataFrame) -> pd.Series:
    """Binary engagement proxy based on 4 observable signals."""
    desc_len_median = df["description"].str.len().median()
    cond1 = df["rating"].isin(HIGH_ENG_RATINGS).astype(int)
    cond2 = df["release_year"].isin(RECENT_YEARS).astype(int)
    cond3 = (df["description"].str.len() > desc_len_median).astype(int)
    cond4 = df["listed_in"].apply(
        lambda g: int(any(genre in str(g) for genre in HIGH_ENG_GENRES))
    )
    score = cond1 + cond2 + cond3 + cond4
    return (score >= 2).astype(int)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df["rating"].isin(BAD_RATINGS)].copy()

    # fill nulls
    df["director"]    = df["director"].fillna("Unknown")
    df["cast"]        = df["cast"].fillna("Unknown")
    df["country"]     = df["country"].fillna("Unknown")
    df["rating"]      = df["rating"].fillna("Unknown")
    df["listed_in"]   = df["listed_in"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    df["date_added"]  = pd.to_datetime(df["date_added"].str.strip(),
                                       format="%B %d, %Y", errors="coerce")

    # numeric features
    df["duration_num"]       = df.apply(parse_duration, axis=1)
    df["desc_length"]        = df["description"].str.len()
    df["desc_word_count"]    = df["description"].str.split().str.len()
    df["is_movie"]           = (df["type"] == "Movie").astype(int)
    df["is_recent"]          = df["release_year"].isin(RECENT_YEARS).astype(int)
    df["month_added"]        = df["date_added"].dt.month.fillna(0).astype(int)
    df["year_added"]         = df["date_added"].dt.year.fillna(0).astype(int)
    df["country_primary"]    = df["country"].str.split(",").str[0].str.strip()
    df["genre_count"]        = df["listed_in"].str.split(",").str.len()
    df["has_known_director"] = (df["director"] != "Unknown").astype(int)
    df["cast_count"]         = df["cast"].str.split(",").str.len().fillna(0)

    # title features
    df["title_length"]       = df["title"].str.len()
    df["title_word_count"]   = df["title"].str.split().str.len()

    # target
    df["success"] = build_success_label(df)

    return df


# ── Step 2: Build ML pipelines ────────────────────────────────────────────────
NUM_COLS  = [
    "duration_num", "release_year", "desc_length", "desc_word_count",
    "is_movie", "is_recent", "month_added", "year_added",
    "genre_count", "has_known_director", "cast_count",
    "title_length", "title_word_count",
]
CAT_COLS  = ["rating", "country_primary", "type"]
TEXT_COL  = "description"
GENRE_COL = "listed_in"


def build_preprocessor():
    return ColumnTransformer([
        ("num",   StandardScaler(),                                    NUM_COLS),
        ("cat",   OneHotEncoder(handle_unknown="ignore", sparse_output=True), CAT_COLS),
        ("text",  TfidfVectorizer(max_features=80, stop_words="english"), TEXT_COL),
        ("genre", TfidfVectorizer(max_features=40, binary=True,
                                  token_pattern=r"[^,]+"),           GENRE_COL),
    ], remainder="drop")


CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, random_state=42, n_jobs=-1
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.08, random_state=42
    ),
    "Extra Trees": ExtraTreesClassifier(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    ),
}


# ── Step 3: Evaluate + compare ────────────────────────────────────────────────
def evaluate(model, X_te, y_te) -> dict:
    preds  = model.predict(X_te)
    probas = model.predict_proba(X_te)[:, 1]
    return {
        "accuracy":  round(accuracy_score(y_te, preds),            4),
        "f1":        round(f1_score(y_te, preds, average="weighted"), 4),
        "precision": round(precision_score(y_te, preds, zero_division=0), 4),
        "recall":    round(recall_score(y_te, preds, zero_division=0),    4),
        "roc_auc":   round(roc_auc_score(y_te, probas),            4),
    }


# ── Step 4: Automated insights ────────────────────────────────────────────────
def generate_insights(df: pd.DataFrame) -> dict:
    insights = {}

    # Success rate by genre
    genre_success = []
    for genre in [
        "International Movies", "Dramas", "Comedies", "International TV Shows",
        "Documentaries", "Action & Adventure", "TV Dramas", "Thrillers",
        "Crime TV Shows", "Children & Family Movies", "Romantic Movies",
        "Independent Movies", "Stand-Up Comedy", "Sci-Fi & Fantasy",
    ]:
        mask = df["listed_in"].str.contains(genre, na=False)
        if mask.sum() < 10:
            continue
        rate = df.loc[mask, "success"].mean()
        cnt  = int(mask.sum())
        genre_success.append({"genre": genre, "success_rate": round(rate, 3),
                               "count": cnt})
    insights["genre_success"] = pd.DataFrame(genre_success).sort_values(
        "success_rate", ascending=False
    )

    # Success rate by country
    top_countries = df["country_primary"].value_counts().head(15).index
    country_data  = []
    for c in top_countries:
        mask = df["country_primary"] == c
        rate = df.loc[mask, "success"].mean()
        cnt  = int(mask.sum())
        country_data.append({"country": c, "success_rate": round(rate, 3), "count": cnt})
    insights["country_success"] = pd.DataFrame(country_data).sort_values(
        "success_rate", ascending=False
    )

    # Annual production trend
    year_df = df[df["release_year"] >= 2010].copy()
    annual  = year_df.groupby("release_year").agg(
        total=("title", "count"),
        success_count=("success", "sum"),
    ).reset_index()
    annual["success_rate"] = (annual["success_count"] / annual["total"]).round(3)
    insights["annual_trend"] = annual

    # Monthly additions trend
    df2 = df.copy()
    df2["ym"] = df2["date_added"].dt.to_period("M")
    monthly = df2.dropna(subset=["date_added"]).groupby("ym").agg(
        total=("title", "count"),
        success_count=("success", "sum"),
    ).reset_index()
    monthly["ym_ts"]        = monthly["ym"].dt.to_timestamp()
    monthly["success_rate"] = (monthly["success_count"] / monthly["total"]).round(3)
    insights["monthly_trend"] = monthly[monthly["ym_ts"] >= "2016-01-01"]

    # Rating breakdown
    rating_df = df.groupby("rating").agg(
        total=("title", "count"),
        success_count=("success", "sum"),
    ).reset_index()
    rating_df["success_rate"] = (rating_df["success_count"] / rating_df["total"]).round(3)
    rating_df = rating_df[~rating_df["rating"].isin(BAD_RATINGS | {"Unknown", "NR", "UR"})]
    insights["rating_breakdown"] = rating_df.sort_values("success_rate", ascending=False)

    # Type split
    insights["type_split"] = df.groupby(["type", "success"]).size().unstack(fill_value=0)

    # Top performing titles (proxy: high engagement score)
    df["eng_score"] = (
        df["desc_length"] / df["desc_length"].max() * 0.3
        + df["is_recent"] * 0.3
        + df["genre_count"] / df["genre_count"].max() * 0.2
        + df["success"] * 0.2
    )
    insights["top_titles"] = df.nlargest(20, "eng_score")[
        ["title", "type", "rating", "release_year", "listed_in", "eng_score"]
    ].reset_index(drop=True)

    # Description length vs success
    insights["desc_stats"] = df.groupby("success")["desc_length"].describe().round(1)

    # KPIs
    insights["kpis"] = {
        "total_titles":      len(df),
        "high_engagement":   int(df["success"].sum()),
        "success_rate":      round(df["success"].mean() * 100, 1),
        "total_movies":      int((df["type"] == "Movie").sum()),
        "total_tv":          int((df["type"] == "TV Show").sum()),
        "unique_countries":  int(df["country_primary"].nunique()),
        "unique_genres":     int(
            len({g.strip() for gs in df["listed_in"].dropna()
                 for g in gs.split(",")})
        ),
        "year_range":        f"{int(df['release_year'].min())}–{int(df['release_year'].max())}",
    }

    return insights


# ── main ───────────────────────────────────────────────────────────────────────
def run():
    print("Step 1: Feature engineering …")
    raw = pd.read_csv(DATA_PATH)
    df  = engineer_features(raw)
    print(f"  {len(df)} rows | success={df['success'].sum()} "
          f"({df['success'].mean()*100:.1f}%)")

    print("\nStep 2: Building ML models …")
    X_df  = df[NUM_COLS + CAT_COLS + [TEXT_COL, GENRE_COL]]
    y     = df["success"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_df, y, test_size=0.2, random_state=42, stratify=y
    )

    fitted_models: dict  = {}
    metrics_table: dict  = {}
    feature_importance: dict = {}

    for name, clf in CLASSIFIERS.items():
        print(f"  Training {name} …", end=" ")
        pipe = Pipeline([
            ("prep", build_preprocessor()),
            ("clf",  clf),
        ])
        pipe.fit(X_tr, y_tr)
        met = evaluate(pipe, X_te, y_te)
        fitted_models[name]  = pipe
        metrics_table[name]  = met
        print(f"acc={met['accuracy']:.4f}  f1={met['f1']:.4f}  "
              f"auc={met['roc_auc']:.4f}")

        # feature importance for tree models
        if hasattr(clf, "feature_importances_"):
            prep    = pipe.named_steps["prep"]
            num_names = NUM_COLS
            try:
                cat_names = list(
                    prep.named_transformers_["cat"].get_feature_names_out(CAT_COLS)
                )
            except Exception:
                cat_names = []
            try:
                text_names = [f"tfidf_{w}" for w in
                              prep.named_transformers_["text"].get_feature_names_out()]
            except Exception:
                text_names = []
            try:
                genre_names = [f"genre_{w}" for w in
                               prep.named_transformers_["genre"].get_feature_names_out()]
            except Exception:
                genre_names = []
            all_names = num_names + cat_names + text_names + genre_names
            imps      = clf.feature_importances_
            n         = min(len(all_names), len(imps))
            fi_df     = pd.DataFrame({
                "feature":    all_names[:n],
                "importance": imps[:n],
            }).sort_values("importance", ascending=False).head(20)
            feature_importance[name] = fi_df

    print("\nStep 3: Model comparison")
    met_df = pd.DataFrame(metrics_table).T
    met_df.index.name = "Model"
    best_model_name = met_df["roc_auc"].idxmax()
    print(met_df.to_string())
    print(f"\n  Best model: {best_model_name}")

    print("\nStep 4: Generating insights …")
    insights = generate_insights(df)
    print(f"  Genre insights: {len(insights['genre_success'])} rows")
    print(f"  Top titles: {len(insights['top_titles'])}")

    print("\nStep 5: Saving artefacts …")
    joblib.dump(fitted_models, MODELS_OUT)
    payload = {
        "metrics":            metrics_table,
        "metrics_df":         met_df,
        "best_model":         best_model_name,
        "feature_importance": feature_importance,
        "insights":           insights,
        "engineered_df":      df,
    }
    joblib.dump(payload, DATA_OUT)
    print(f"  Saved → {MODELS_OUT}")
    print(f"  Saved → {DATA_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    run()
