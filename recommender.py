"""
recommender.py – Hybrid Weighted Recommendation Engine
=======================================================
Computes three separate cosine similarity matrices:
  1. Plot similarity  – TF-IDF on description
  2. Genre similarity – Binary CountVectorizer on listed_in
  3. Cast/Director    – CountVectorizer on cast + director

Final score = w_plot * sim_plot + w_genre * sim_genre + w_cast * sim_cast

Serendipity mode mixes in titles from adjacent K-Means clusters.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = Path(__file__).resolve().parent / "cleaned_netflix_titles.csv"
CLUSTERED_PKL = Path(__file__).resolve().parent / "clustered_df.pkl"
KMEANS_PKL = Path(__file__).resolve().parent / "kmeans_model.pkl"


# ── cached matrix build ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_matrices(dataset_path: str):
    """Build all three similarity matrices. Cached so we only do this once."""
    df = pd.read_csv(dataset_path)

    # ── 1. Plot TF-IDF ───────────────────────────────────────────────────────
    plot_vec = TfidfVectorizer(stop_words="english", max_features=5000)
    mat_plot = plot_vec.fit_transform(df["description"].fillna(""))

    # ── 2. Genre binary count ────────────────────────────────────────────────
    genre_vec = CountVectorizer(
        tokenizer=lambda x: [g.strip() for g in x.split(",")],
        token_pattern=None,
        binary=True,
    )
    mat_genre = genre_vec.fit_transform(df["listed_in"].fillna(""))

    # ── 3. Cast + Director count ─────────────────────────────────────────────
    df["_cast_dir"] = (
        df["cast"].fillna("") + ", " + df["director"].fillna("")
    )
    cast_vec = CountVectorizer(
        tokenizer=lambda x: [p.strip() for p in x.split(",") if p.strip()],
        token_pattern=None,
    )
    mat_cast = cast_vec.fit_transform(df["_cast_dir"])

    return df, mat_plot, mat_genre, mat_cast


def _load_cluster_map() -> dict[str, int] | None:
    """Return {title_lower: cluster_id} from the saved clustered DataFrame."""
    if not CLUSTERED_PKL.exists():
        return None
    try:
        cdf = joblib.load(CLUSTERED_PKL)
        return dict(zip(cdf["title"].str.lower(), cdf["cluster"].astype(int)))
    except Exception:
        return None


# ── public API ───────────────────────────────────────────────────────────────

def get_recommendations(
    title: str,
    top_n: int = 5,
    w_plot: float = 0.50,
    w_genre: float = 0.30,
    w_cast: float = 0.20,
    serendipity: bool = False,
) -> tuple[pd.DataFrame, float]:
    """
    Returns (recommendations_df, elapsed_ms).

    recommendations_df columns:
      title, genres, rating, release_year, duration, cast,
      similarity_score_percent, plot_pct, genre_pct, cast_pct, is_serendipity
    """
    t0 = time.perf_counter()
    df, mat_plot, mat_genre, mat_cast = _build_matrices(str(DATASET_PATH))

    title_lower = title.lower()
    mask = df["title"].astype(str).str.lower() == title_lower
    if not mask.any():
        raise ValueError(f"Title '{title}' was not found in the dataset")

    idx = df.index[mask][0]

    # ── row-wise similarities ─────────────────────────────────────────────────
    sim_plot  = cosine_similarity(mat_plot[idx],  mat_plot).flatten()
    sim_genre = cosine_similarity(mat_genre[idx], mat_genre).flatten()
    sim_cast  = cosine_similarity(mat_cast[idx],  mat_cast).flatten()

    # ── normalise weights so they always sum to 1.0 ──────────────────────────
    total = w_plot + w_genre + w_cast
    if total == 0:
        w_plot, w_genre, w_cast = 0.50, 0.30, 0.20
        total = 1.0
    w_plot, w_genre, w_cast = w_plot / total, w_genre / total, w_cast / total

    final = w_plot * sim_plot + w_genre * sim_genre + w_cast * sim_cast

    # ── serendipity: inject neighbours from adjacent clusters ─────────────────
    serendipity_indices: set[int] = set()
    if serendipity:
        cluster_map = _load_cluster_map()
        if cluster_map is not None:
            source_cluster = cluster_map.get(title_lower)
            if source_cluster is not None:
                all_clusters = sorted(set(cluster_map.values()))
                n_clusters = len(all_clusters)
                adj_clusters = {
                    (source_cluster - 1) % n_clusters,
                    (source_cluster + 1) % n_clusters,
                }
                title_series = df["title"].astype(str).str.lower()
                for t, c in cluster_map.items():
                    if c in adj_clusters:
                        rows = df.index[title_series == t].tolist()
                        if rows:
                            serendipity_indices.add(rows[0])

    # ── rank candidates ───────────────────────────────────────────────────────
    ranked = final.argsort()[::-1]
    ranked = [i for i in ranked if i != idx]

    if serendipity and serendipity_indices:
        # interleave: take top (top_n - bonus) strict + some serendipity
        strict_n = max(top_n - 2, top_n // 2)
        strict = [i for i in ranked if i not in serendipity_indices][:strict_n]
        novel  = [i for i in ranked if i in serendipity_indices][: top_n - strict_n]
        chosen = strict + novel
        # re-sort so output is still score-ordered within each group
        chosen = sorted(chosen, key=lambda i: final[i], reverse=True)[:top_n]
    else:
        chosen = ranked[:top_n]

    # ── assemble output ───────────────────────────────────────────────────────
    rows_out = []
    for i in chosen:
        f = final[i]
        sp = sim_plot[i]
        sg = sim_genre[i]
        sc = sim_cast[i]

        # component contribution as share of the blended score
        denom = (w_plot * sp + w_genre * sg + w_cast * sc) or 1e-9
        plot_share  = round(w_plot  * sp / denom * 100, 1)
        genre_share = round(w_genre * sg / denom * 100, 1)
        cast_share  = round(w_cast  * sc / denom * 100, 1)

        rows_out.append(
            {
                "title":                    df.loc[i, "title"],
                "genres":                   df.loc[i, "listed_in"],
                "rating":                   df.loc[i, "rating"],
                "release_year":             int(df.loc[i, "release_year"])
                                            if pd.notna(df.loc[i, "release_year"])
                                            else "N/A",
                "duration":                 str(df.loc[i, "duration"]) if "duration" in df.columns else "N/A",
                "cast":                     str(df.loc[i, "cast"])[:80] + "…"
                                            if pd.notna(df.loc[i, "cast"])
                                               and len(str(df.loc[i, "cast"])) > 80
                                            else str(df.loc[i, "cast"])
                                            if pd.notna(df.loc[i, "cast"])
                                            else "N/A",
                "similarity_score_percent": round(f * 100, 1),
                "plot_pct":                 plot_share,
                "genre_pct":                genre_share,
                "cast_pct":                 cast_share,
                "is_serendipity":           i in serendipity_indices,
            }
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return pd.DataFrame(rows_out), round(elapsed_ms, 2)


def get_matrices(dataset_path: str | None = None) -> tuple:
    """
    Return (df, mat_plot, mat_genre, mat_cast) from the cached build.
    Used by graph_viz.py to compute secondary edge weights without
    re-building the matrices.
    """
    path = dataset_path or str(DATASET_PATH)
    return _build_matrices(path)


if __name__ == "__main__":
    query_title = sys.argv[1] if len(sys.argv) > 1 else "Blood & Water"
    recs, ms = get_recommendations(query_title)
    print(f"Top recommendations for '{query_title}' [{ms:.1f} ms]:")
    print(recs[["title", "similarity_score_percent", "plot_pct", "genre_pct", "cast_pct"]].to_string(index=False))
