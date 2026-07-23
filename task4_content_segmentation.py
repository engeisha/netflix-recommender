"""
Task 4 – Netflix Content Segmentation
======================================
Clusters Netflix titles using K-Means on a mixed feature matrix
(TF-IDF description + one-hot genres/rating/type + scaled numerics).
PCA projects the matrix to 2-D for visualization.

Saved artifacts
---------------
  kmeans_model.pkl       – fitted KMeans
  pca_model.pkl          – fitted PCA(n_components=2)
  clustering_pipeline.pkl – fitted ColumnTransformer (used to transform new rows)
  clustered_df.pkl       – DataFrame with all rows + cluster label + PCA coords
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "netflix_titles.csv"

# ── artefact output paths ─────────────────────────────────────────────────────
KMEANS_OUT    = ROOT / "kmeans_model.pkl"
PCA_OUT       = ROOT / "pca_model.pkl"
PIPELINE_OUT  = ROOT / "clustering_pipeline.pkl"
CLUSTERED_OUT = ROOT / "clustered_df.pkl"

# rating artefacts (duration mis-labelled rows) to drop
BAD_RATINGS = {"74 min", "84 min", "66 min"}

CLUSTER_PERSONAS = {
    0: "🌍 Global Drama & Thrillers",
    1: "👶 Kids & Family Entertainment",
    2: "🎬 Hollywood Blockbusters",
    3: "📺 Binge-Worthy TV Series",
    4: "😂 Comedies & Stand-Up",
    5: "🎭 Independent & World Cinema",
    6: "🔪 Crime, Mystery & Horror",
    7: "🌏 International & World TV",
}


# ── preprocessing ──────────────────────────────────────────────────────────────
def parse_duration(row: pd.Series) -> int:
    try:
        val = str(row["duration"]).split()[0]
        return int(val)
    except (ValueError, IndexError):
        return 0


def load_and_clean(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    # drop bad-rating artefact rows
    df = df[~df["rating"].isin(BAD_RATINGS)].copy()

    df["type"]        = df["type"].fillna("Movie")
    df["rating"]      = df["rating"].fillna("Unknown")
    df["listed_in"]   = df["listed_in"].fillna("Unknown")
    df["country"]     = df["country"].fillna("Unknown")
    df["description"] = df["description"].fillna("")

    # simplify country to first listed country
    df["country_primary"] = df["country"].str.split(",").str[0].str.strip()

    # keep top-N countries; bucket rest as "Other"
    top_countries = df["country_primary"].value_counts().head(20).index
    df["country_primary"] = df["country_primary"].where(
        df["country_primary"].isin(top_countries), other="Other"
    )

    # numeric duration (minutes for movies, seasons for TV)
    df["duration_num"] = df.apply(parse_duration, axis=1)
    df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce").fillna(2015)

    return df


# ── feature matrix ─────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame):
    """
    Returns (X_sparse, pipeline_dict) where X_sparse is the combined
    feature matrix and pipeline_dict holds fitted transformers for reuse.
    """
    # 1. TF-IDF on description
    tfidf = TfidfVectorizer(max_features=80, stop_words="english", ngram_range=(1, 2))
    X_text = tfidf.fit_transform(df["description"])

    # 2. OneHot: type, rating, country_primary
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    X_cat = ohe.fit_transform(df[["type", "rating", "country_primary"]])

    # 3. Multi-label genres via binary TF-IDF (binary=True treats each genre as a token)
    genre_tfidf = TfidfVectorizer(max_features=60, binary=True, token_pattern=r"[^,]+")
    X_genre = genre_tfidf.fit_transform(df["listed_in"])

    # 4. Scaled numerics
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(df[["release_year", "duration_num"]])
    # convert to sparse so we can hstack
    from scipy.sparse import csr_matrix
    X_num_sparse = csr_matrix(X_num)

    X = hstack([X_text, X_cat, X_genre, X_num_sparse])

    pipeline = {
        "tfidf":       tfidf,
        "ohe":         ohe,
        "genre_tfidf": genre_tfidf,
        "scaler":      scaler,
    }
    return X, pipeline


def transform_row(row_dict: dict, pipeline: dict):
    """Transform a single new-row dict into the same feature space."""
    from scipy.sparse import csr_matrix
    df_row = pd.DataFrame([row_dict])
    df_row["description"]     = df_row.get("description", pd.Series([""])).fillna("")
    df_row["listed_in"]       = df_row.get("listed_in", pd.Series(["Unknown"])).fillna("Unknown")
    df_row["type"]             = df_row.get("type", pd.Series(["Movie"])).fillna("Movie")
    df_row["rating"]           = df_row.get("rating", pd.Series(["Unknown"])).fillna("Unknown")
    df_row["country_primary"]  = df_row.get("country_primary", pd.Series(["Unknown"])).fillna("Unknown")
    df_row["release_year"]     = pd.to_numeric(df_row.get("release_year", pd.Series([2020])), errors="coerce").fillna(2020)
    df_row["duration_num"]     = pd.to_numeric(df_row.get("duration_num", pd.Series([90])), errors="coerce").fillna(90)

    X_text  = pipeline["tfidf"].transform(df_row["description"])
    X_cat   = pipeline["ohe"].transform(df_row[["type", "rating", "country_primary"]])
    X_genre = pipeline["genre_tfidf"].transform(df_row["listed_in"])
    X_num   = csr_matrix(pipeline["scaler"].transform(df_row[["release_year", "duration_num"]]))
    return hstack([X_text, X_cat, X_genre, X_num])


# ── find optimal k ─────────────────────────────────────────────────────────────
def find_optimal_k(X, k_range=range(3, 10), sample_n=3000) -> int:
    # subsample for speed
    n = X.shape[0]
    idx = np.random.default_rng(42).choice(n, size=min(sample_n, n), replace=False)
    X_s = X[idx]

    best_k, best_score = 6, -1
    print("  k  | silhouette")
    print("  ---+-----------")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5, max_iter=200)
        labels = km.fit_predict(X_s)
        score = silhouette_score(X_s, labels, sample_size=min(1500, len(idx)))
        print(f"  {k:2d} | {score:.4f}")
        if score > best_score:
            best_score, best_k = score, k
    print(f"  → optimal k = {best_k}  (silhouette={best_score:.4f})")
    return best_k


# ── cluster summary ────────────────────────────────────────────────────────────
def cluster_summary(df: pd.DataFrame) -> dict:
    summary = {}
    for cid in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cid]

        # top genres: flatten listed_in tokens
        all_genres = sub["listed_in"].str.split(",").explode().str.strip()
        top_genres = all_genres.value_counts().head(5).index.tolist()

        # top countries
        top_countries = sub["country_primary"].value_counts().head(3).index.tolist()

        # most common rating (non-Unknown)
        ratings = sub["rating"][sub["rating"] != "Unknown"]
        common_rating = ratings.value_counts().index[0] if len(ratings) else "N/A"

        summary[int(cid)] = {
            "count":         int(len(sub)),
            "persona":       CLUSTER_PERSONAS.get(int(cid), f"Cluster {cid}"),
            "top_genres":    top_genres,
            "top_countries": top_countries,
            "common_rating": common_rating,
            "avg_year":      round(float(sub["release_year"].mean()), 1),
            "type_split":    sub["type"].value_counts().to_dict(),
            "sample_titles": sub["title"].dropna().sample(
                min(6, len(sub)), random_state=42
            ).tolist(),
        }
    return summary


# ── main ───────────────────────────────────────────────────────────────────────
def run(data_path: Path = DATA_PATH):
    print("Loading & cleaning data …")
    df = load_and_clean(data_path)
    print(f"  {len(df)} rows after cleaning")

    print("\nBuilding feature matrix …")
    X, pipeline = build_features(df)
    print(f"  Feature matrix shape: {X.shape}")

    print("\nFinding optimal k …")
    k = find_optimal_k(X)

    print(f"\nFitting K-Means (k={k}) …")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    df["cluster"] = kmeans.fit_predict(X)
    print(f"  Cluster counts:\n{df['cluster'].value_counts().sort_index()}")

    print("\nFitting PCA (2 components) …")
    pca = PCA(n_components=2, random_state=42)
    # PCA needs dense; convert in chunks to avoid OOM
    X_dense = X.toarray()
    coords  = pca.fit_transform(X_dense)
    df["pca_x"] = coords[:, 0]
    df["pca_y"] = coords[:, 1]
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_.round(4)}")

    # attach persona label
    df["persona"] = df["cluster"].map(
        lambda c: CLUSTER_PERSONAS.get(int(c), f"Cluster {c}")
    )

    # build summary
    summary = cluster_summary(df)
    pipeline["cluster_summary"] = summary
    pipeline["k"] = k

    print("\nSaving artifacts …")
    joblib.dump(kmeans,   KMEANS_OUT)
    joblib.dump(pca,      PCA_OUT)
    joblib.dump(pipeline, PIPELINE_OUT)
    joblib.dump(df,       CLUSTERED_OUT)

    for p in [KMEANS_OUT, PCA_OUT, PIPELINE_OUT, CLUSTERED_OUT]:
        print(f"  Saved → {p}")

    print("\nDone.")
    return df, kmeans, pca, pipeline


if __name__ == "__main__":
    run()
