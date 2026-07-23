"""
Content-Type Classifier page.

Feature pipeline mirrors training:
  release_year        – numeric
  duration            – numeric (minutes or seasons as integer)
  rating_*            – one-hot (18 cols)
  listed_in_*         – one-hot over exact comma-joined genre combos (514 cols)
  target_encoded_rating    – mean(type_encoded) per rating
  target_encoded_listed_in – mean(type_encoded) per listed_in combo
  tfidf_0 … tfidf_49  – TF-IDF on description (max_features=50, rebuilt from CSV)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from ui_theme import inject_global_css

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "cleaned_netflix_titles.csv"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Classifier · Netflix AI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

inject_global_css()

st.markdown(
    """
    <style>
    :root { --bg:#0c0c0c; --panel:#1e1e1e; --text:#f5f5f5;
            --muted:#a0a0a0; --accent:#E50914; --border:#2a2a2a; }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { padding-top: 1rem; padding-bottom: 2.5rem; }
    .result-box {
        border: 1px solid var(--border); border-radius: 18px;
        padding: 1.4rem 1.6rem; margin-top: 1rem; background: #171717;
    }
    .result-label { font-size: 1.8rem; font-weight: 900; margin-bottom: 0.25rem; }
    .movie-color  { color: #E50914; }
    .tvshow-color { color: #3b82f6; }
    .conf-track   { height: 10px; border-radius: 999px; background: #2a2a2a; margin-top: 0.6rem; }
    .conf-fill    { height: 10px; border-radius: 999px; }
    .hint { color: var(--muted); font-size: 0.88rem; margin-top: 0.45rem; }
    label { color: #d0d0d0 !important; }

    /* textarea contrast fix */
    textarea, div[data-baseweb="textarea"] textarea {
        color: #FFFFFF !important;
        background-color: #1F1F1F !important;
        border: 1px solid #333333 !important;
    }
    textarea::placeholder { color: #888888 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:0.12em;margin-bottom:0.3rem;'>🤖 CONTENT TYPE CLASSIFIER</div>"
    "<div style='color:#a0a0a0;font-size:0.93rem;margin-bottom:1.4rem;'>"
    "Predict whether a title is a <strong style='color:#E50914;'>Movie</strong> or "
    "<strong style='color:#3b82f6;'>TV Show</strong> from its metadata.</div>",
    unsafe_allow_html=True,
)


# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    clf_path = ROOT / "best_classifier.pkl"
    vec_path = ROOT / "vectorizer.pkl"
    if not clf_path.exists() or not vec_path.exists():
        return None, None, None, None, None, None
    clf = joblib.load(clf_path)
    feature_cols = joblib.load(vec_path)
    return clf, feature_cols


@st.cache_data
def load_target_encoding_maps():
    """Rebuild target-encoding dicts from the cleaned CSV (same as training)."""
    df = pd.read_csv(DATASET_PATH)
    df["type_encoded"] = (df["type"] == "TV Show").astype(int)
    global_mean = float(df["type_encoded"].mean())
    rating_map = df.groupby("rating")["type_encoded"].mean().to_dict()
    listed_map = df.groupby("listed_in")["type_encoded"].mean().to_dict()
    return rating_map, listed_map, global_mean


@st.cache_resource
def load_tfidf() -> TfidfVectorizer:
    """Refit TF-IDF on full corpus with same params used at training."""
    df = pd.read_csv(DATASET_PATH)
    tfidf = TfidfVectorizer(max_features=50)
    tfidf.fit(df["description"].fillna(""))
    return tfidf


clf, feature_cols = load_artifacts()

if clf is None:
    st.warning(
        "Model files not found. Place `best_classifier.pkl` and `vectorizer.pkl` "
        "in the project root to enable this page.",
        icon="⚠️",
    )
    st.stop()

rating_map, listed_map, global_mean = load_target_encoding_maps()
tfidf_vec = load_tfidf()

# ── derive UI options from feature columns ─────────────────────────────────────
RATING_COLS    = [c for c in feature_cols if c.startswith("rating_")]
LISTED_IN_COLS = [c for c in feature_cols if c.startswith("listed_in_")]

# Valid content ratings (strip prefix, exclude duration artefacts)
VALID_RATINGS = sorted(
    c[len("rating_"):] for c in RATING_COLS
    if not c[len("rating_"):].endswith(" min")
)

# Unique individual genre tokens
_genre_tokens: set[str] = set()
for col in LISTED_IN_COLS:
    for token in col[len("listed_in_"):].split(", "):
        _genre_tokens.add(token.strip())
GENRES = sorted(_genre_tokens)


# ── input form ─────────────────────────────────────────────────────────────────
with st.form("classifier_form"):
    col_left, col_right = st.columns(2)

    with col_left:
        release_year = st.number_input(
            "Release Year", min_value=1925, max_value=2026, value=2021, step=1
        )
        content_rating = st.selectbox("Content Rating", VALID_RATINGS, index=VALID_RATINGS.index("TV-MA") if "TV-MA" in VALID_RATINGS else 0)
        dur_col1, dur_col2 = st.columns([2, 1])
        with dur_col1:
            duration_val = st.number_input("Duration", min_value=1, max_value=500, value=90, step=1)
        with dur_col2:
            duration_type = st.selectbox("Unit", ["Minutes", "Seasons"])

    with col_right:
        selected_genres = st.multiselect(
            "Genres / Listed In",
            options=GENRES,
            default=["Dramas", "International Movies"],
            help="Select one or more genres. The classifier will match the closest training combination.",
        )
        description = st.text_area(
            "Description / Plot Summary",
            height=138,
            placeholder="e.g. A detective uncovers a web of corruption in a gritty crime drama set in Seoul...",
        )

    submitted = st.form_submit_button("🎯 Predict Content Type", type="primary", use_container_width=True)


# ── preprocessing & prediction ─────────────────────────────────────────────────
def build_listed_in_string(genres: list[str]) -> str:
    """Sort selected genres to match training column names (sorted alphabetically)."""
    return ", ".join(sorted(genres))


def find_best_listed_in_col(genres: list[str]) -> str | None:
    """
    Find the exact or closest matching listed_in column.
    Priority: exact match → most-overlap subset → first partial match.
    """
    if not genres:
        return None
    target = build_listed_in_string(genres)
    target_set = set(genres)

    # 1. exact match
    exact_key = f"listed_in_{target}"
    if exact_key in feature_cols:
        return exact_key

    # 2. best overlap: column whose genre set has most intersection with user selection
    best_col, best_score = None, -1
    for col in LISTED_IN_COLS:
        col_genres = set(col[len("listed_in_"):].split(", "))
        score = len(col_genres & target_set) / max(len(col_genres | target_set), 1)
        if score > best_score:
            best_score, best_col = score, col
    return best_col


def build_feature_vector(
    release_year: int,
    duration: int,
    rating: str,
    genres: list[str],
    description: str,
) -> pd.DataFrame:
    row: dict[str, float] = {c: 0.0 for c in feature_cols}

    # numeric
    row["release_year"] = float(release_year)
    row["duration"] = float(duration)

    # one-hot rating
    rating_col = f"rating_{rating}"
    if rating_col in row:
        row[rating_col] = 1.0

    # one-hot listed_in (best match)
    best_listed = find_best_listed_in_col(genres)
    if best_listed and best_listed in row:
        row[best_listed] = 1.0

    # target-encoded rating
    row["target_encoded_rating"] = rating_map.get(rating, global_mean)

    # target-encoded listed_in (use the matched combination key)
    if best_listed:
        combo_key = best_listed[len("listed_in_"):]
        row["target_encoded_listed_in"] = listed_map.get(combo_key, global_mean)
    else:
        row["target_encoded_listed_in"] = global_mean

    # TF-IDF on description
    tfidf_vals = tfidf_vec.transform([description or ""]).toarray()[0]
    for i, v in enumerate(tfidf_vals):
        row[f"tfidf_{i}"] = float(v)

    return pd.DataFrame([row])[feature_cols]


if submitted:
    if not description.strip():
        st.error("Please enter a description before predicting.")
    else:
        duration_numeric = duration_val  # already an integer from number_input

        with st.spinner("Running prediction..."):
            X = build_feature_vector(
                release_year=release_year,
                duration=duration_numeric,
                rating=content_rating,
                genres=selected_genres,
                description=description.strip(),
            )
            raw_pred = clf.predict(X)[0]          # 0 = Movie, 1 = TV Show
            probas   = clf.predict_proba(X)[0]

        label      = "Movie" if raw_pred == 0 else "TV Show"
        icon       = "🎬"   if raw_pred == 0 else "📺"
        color_cls  = "movie-color" if raw_pred == 0 else "tvshow-color"
        bar_color  = "#E50914"     if raw_pred == 0 else "#3b82f6"
        confidence = float(probas[raw_pred]) * 100

        conf_row = "".join(
            f"<span style='margin-right:1rem;color:#c0c0c0;'>"
            f"<strong style='color:#f5f5f5;'>{'Movie' if i == 0 else 'TV Show'}</strong> "
            f"{p*100:.1f}%</span>"
            for i, p in enumerate(probas)
        )

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label {color_cls}">{icon} {label}</div>
                <div style="color:#a0a0a0;font-size:0.9rem;">Predicted content type</div>
                <div class="hint">Confidence: <strong style="color:#f5f5f5;">{confidence:.1f}%</strong></div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{confidence:.1f}%;background:{bar_color};"></div>
                </div>
                <div class="hint" style="margin-top:0.7rem;">{conf_row}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
