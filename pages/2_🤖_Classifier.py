"""
Content-Type Classifier page — with XAI Inspector, Confidence Breakdown,
and What-If Analysis.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer

from ui_theme import inject_global_css
from xai_utils import (
    feature_impact_chart,
    proba_chart,
    whatif_comparison_chart,
    word_weights_from_rf,
)

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "cleaned_netflix_titles.csv"

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Classifier · Netflix AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

st.markdown(
    """
    <style>
    :root { --bg:#0c0c0c; --panel:#1e1e1e; --text:#f5f5f5;
            --muted:#a0a0a0; --accent:#E50914; --border:#2a2a2a; }
    .stApp { background:var(--bg); color:var(--text); }
    .block-container { padding-top:1rem; padding-bottom:2.5rem; }

    .result-box {
        border:1px solid var(--border); border-radius:18px;
        padding:1.4rem 1.6rem; margin-top:1rem; background:#141414;
    }
    .result-label { font-size:1.8rem; font-weight:900; margin-bottom:.25rem; }
    .movie-color  { color:#E50914; }
    .tvshow-color { color:#3b82f6; }
    .conf-track { height:10px; border-radius:999px; background:#2a2a2a; margin-top:.6rem; }
    .conf-fill  { height:10px; border-radius:999px; }
    .hint { color:var(--muted); font-size:.88rem; margin-top:.45rem; }

    /* XAI glass containers */
    .xai-panel {
        background:linear-gradient(160deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
        border:1px solid var(--border); border-radius:18px;
        padding:1.2rem 1.4rem; margin-top:1rem;
    }
    .xai-title {
        color:#f5f5f5; font-size:.93rem; font-weight:700;
        letter-spacing:.05em; text-transform:uppercase;
        margin-bottom:.7rem;
    }
    .xai-legend { display:flex; gap:1.2rem; margin-bottom:.6rem; }
    .legend-dot {
        width:10px; height:10px; border-radius:50%;
        display:inline-block; margin-right:5px; vertical-align:middle;
    }
    .legend-lbl { color:#a0a0a0; font-size:.8rem; vertical-align:middle; }

    /* what-if */
    .whatif-panel {
        background:linear-gradient(160deg,rgba(59,130,246,.06),rgba(12,12,12,.9));
        border:1px solid rgba(59,130,246,.25); border-radius:18px;
        padding:1.2rem 1.4rem; margin-top:1.2rem;
    }
    .whatif-title { color:#93c5fd; font-size:.93rem; font-weight:700;
                    letter-spacing:.05em; text-transform:uppercase; margin-bottom:.7rem; }

    label { color:#d0d0d0 !important; }
    textarea, div[data-baseweb="textarea"] textarea {
        color:#FFFFFF !important; background-color:#1F1F1F !important;
        border:1px solid #333 !important;
    }
    textarea::placeholder { color:#888 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:.12em;margin-bottom:.3rem;'>🤖 CONTENT TYPE CLASSIFIER</div>"
    "<div style='color:#a0a0a0;font-size:.93rem;margin-bottom:1.4rem;'>"
    "Predict whether a title is a <strong style='color:#E50914;'>Movie</strong> or "
    "<strong style='color:#3b82f6;'>TV Show</strong>, with full XAI breakdown.</div>",
    unsafe_allow_html=True,
)

# ── cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    clf_path = ROOT / "best_classifier.pkl"
    vec_path = ROOT / "vectorizer.pkl"
    if not clf_path.exists() or not vec_path.exists():
        return None, None
    return joblib.load(clf_path), joblib.load(vec_path)


@st.cache_data
def load_target_encoding_maps():
    df = pd.read_csv(DATASET_PATH)
    df["type_encoded"] = (df["type"] == "TV Show").astype(int)
    global_mean = float(df["type_encoded"].mean())
    rating_map  = df.groupby("rating")["type_encoded"].mean().to_dict()
    listed_map  = df.groupby("listed_in")["type_encoded"].mean().to_dict()
    return rating_map, listed_map, global_mean


@st.cache_resource
def load_tfidf() -> TfidfVectorizer:
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

RATING_COLS    = [c for c in feature_cols if c.startswith("rating_")]
LISTED_IN_COLS = [c for c in feature_cols if c.startswith("listed_in_")]
VALID_RATINGS  = sorted(
    c[len("rating_"):] for c in RATING_COLS
    if not c[len("rating_"):].endswith(" min")
)
_genre_tokens: set[str] = set()
for col in LISTED_IN_COLS:
    for token in col[len("listed_in_"):].split(", "):
        _genre_tokens.add(token.strip())
GENRES = sorted(_genre_tokens)

CLASS_COLORS = {"Movie": "#E50914", "TV Show": "#3b82f6"}
CLASS_LABELS = {0: "Movie", 1: "TV Show"}


# ── helpers ────────────────────────────────────────────────────────────────────
def build_listed_in_string(genres: list[str]) -> str:
    return ", ".join(sorted(genres))


def find_best_listed_in_col(genres: list[str]) -> str | None:
    if not genres:
        return None
    target_set = set(genres)
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
) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (DataFrame for clf.predict, raw numpy vector)."""
    row: dict[str, float] = {c: 0.0 for c in feature_cols}
    row["release_year"] = float(release_year)
    row["duration"]     = float(duration)
    rating_col = f"rating_{rating}"
    if rating_col in row:
        row[rating_col] = 1.0
    best_listed = find_best_listed_in_col(genres)
    if best_listed and best_listed in row:
        row[best_listed] = 1.0
    row["target_encoded_rating"] = rating_map.get(rating, global_mean)
    if best_listed:
        combo_key = best_listed[len("listed_in_"):]
        row["target_encoded_listed_in"] = listed_map.get(combo_key, global_mean)
    else:
        row["target_encoded_listed_in"] = global_mean
    tfidf_vals = tfidf_vec.transform([description or ""]).toarray()[0]
    for i, v in enumerate(tfidf_vals):
        row[f"tfidf_{i}"] = float(v)
    df_row = pd.DataFrame([row])[feature_cols]
    return df_row, df_row.values[0]


# ── input form ─────────────────────────────────────────────────────────────────
with st.form("classifier_form"):
    col_left, col_right = st.columns(2)
    with col_left:
        release_year   = st.number_input("Release Year", min_value=1925, max_value=2026, value=2021, step=1)
        content_rating = st.selectbox(
            "Content Rating", VALID_RATINGS,
            index=VALID_RATINGS.index("TV-MA") if "TV-MA" in VALID_RATINGS else 0,
        )
        dc1, dc2 = st.columns([2, 1])
        with dc1:
            duration_val  = st.number_input("Duration", min_value=1, max_value=500, value=90, step=1)
        with dc2:
            duration_type = st.selectbox("Unit", ["Minutes", "Seasons"])
    with col_right:
        selected_genres = st.multiselect(
            "Genres / Listed In", options=GENRES,
            default=["Dramas", "International Movies"],
        )
        description = st.text_area(
            "Description / Plot Summary", height=138,
            placeholder="e.g. A detective uncovers a web of corruption in a gritty crime drama set in Seoul...",
        )
    submitted = st.form_submit_button("🎯 Predict Content Type", type="primary", use_container_width=True)


# ── prediction + XAI ──────────────────────────────────────────────────────────
if submitted:
    if not description.strip():
        st.error("Please enter a description before predicting.")
    else:
        with st.spinner("Running prediction…"):
            X_df, X_vec = build_feature_vector(
                release_year=release_year,
                duration=int(duration_val),
                rating=content_rating,
                genres=selected_genres,
                description=description.strip(),
            )
            raw_pred = clf.predict(X_df)[0]
            probas   = clf.predict_proba(X_df)[0]

        label      = CLASS_LABELS[raw_pred]
        icon       = "🎬" if raw_pred == 0 else "📺"
        bar_color  = CLASS_COLORS[label]
        confidence = float(probas[raw_pred]) * 100

        # store in session state for what-if
        st.session_state["clf_orig_desc"]   = description.strip()
        st.session_state["clf_orig_year"]   = release_year
        st.session_state["clf_orig_dur"]    = int(duration_val)
        st.session_state["clf_orig_rating"] = content_rating
        st.session_state["clf_orig_genres"] = selected_genres
        st.session_state["clf_orig_probas"] = probas.tolist()
        st.session_state["clf_classes"]     = [CLASS_LABELS[c] for c in clf.classes_]

        # ── result card ────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="result-box">
              <div class="result-label" style="color:{bar_color};">{icon} {label}</div>
              <div style="color:#a0a0a0;font-size:.9rem;">Predicted content type</div>
              <div class="hint">Confidence: <strong style="color:#f5f5f5;">{confidence:.1f}%</strong></div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{confidence:.1f}%;background:{bar_color};"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── XAI section ────────────────────────────────────────────────────────
        xai_col, proba_col = st.columns([1, 1])

        # 1. Feature impact chart
        with xai_col:
            st.markdown("<div class='xai-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='xai-title'>🔍 Feature Impact Analysis</div>", unsafe_allow_html=True)
            st.markdown(
                "<div class='xai-legend'>"
                "<span><span class='legend-dot' style='background:#22c55e;'></span>"
                "<span class='legend-lbl'>Supporting prediction</span></span>"
                "</div>",
                unsafe_allow_html=True,
            )
            word_df, xai_ms = word_weights_from_rf(
                feature_importances=clf.feature_importances_,
                feature_names=list(feature_cols),
                input_vector=X_vec,
                tfidf_prefix="tfidf_",
                top_n=12,
            )
            # map tfidf_0..N back to actual words
            vocab_inv = {v: k for k, v in tfidf_vec.vocabulary_.items()}
            if not word_df.empty:
                word_df["word"] = word_df["word"].apply(
                    lambda w: vocab_inv.get(int(w), w) if w.isdigit() else w
                )
            fig_impact = feature_impact_chart(
                word_df,
                title=f"Top Words → {label}",
                pos_color="#22c55e",
            )
            st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # 2. Multi-class probability breakdown
        with proba_col:
            st.markdown("<div class='xai-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='xai-title'>📊 Confidence Breakdown</div>", unsafe_allow_html=True)
            class_names = [CLASS_LABELS[c] for c in clf.classes_]
            fig_proba = proba_chart(
                classes=class_names,
                probas=probas.tolist(),
                bar_colors=CLASS_COLORS,
                title="Probability Across All Classes",
            )
            st.plotly_chart(fig_proba, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── telemetry badge ────────────────────────────────────────────────────
        _clf_badge_c = "#22c55e" if xai_ms < 200 else ("#f59e0b" if xai_ms < 800 else "#E50914")
        st.markdown(
            f"<div style='margin-top:.5rem;'>"
            f"<span style='background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);"
            f"border-radius:999px;padding:3px 11px;font-size:.76rem;font-weight:700;"
            f"color:{_clf_badge_c};'>⚡ XAI computed in {xai_ms:.1f} ms</span>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── what-if analysis ───────────────────────────────────────────────────────────
if "clf_orig_probas" in st.session_state:
    st.markdown("<div class='whatif-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='whatif-title'>🧪 What-If Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#a0a0a0;font-size:.85rem;margin-bottom:.9rem;'>"
        "Tweak keywords or duration and compare how the prediction shifts.</div>",
        unsafe_allow_html=True,
    )

    wi_col1, wi_col2 = st.columns([2, 1])
    with wi_col1:
        wi_desc = st.text_area(
            "Modified Description",
            value=st.session_state["clf_orig_desc"],
            height=110,
            key="wi_clf_desc",
        )
    with wi_col2:
        wi_dur = st.number_input(
            "Duration (min/seasons)",
            min_value=1, max_value=500,
            value=st.session_state["clf_orig_dur"],
            key="wi_clf_dur",
        )
        wi_year = st.number_input(
            "Release Year",
            min_value=1925, max_value=2026,
            value=st.session_state["clf_orig_year"],
            key="wi_clf_year",
        )

    if st.button("🔄 Run What-If", key="wi_clf_btn", type="primary"):
        with st.spinner("Re-running…"):
            wi_X_df, _ = build_feature_vector(
                release_year=wi_year,
                duration=wi_dur,
                rating=st.session_state["clf_orig_rating"],
                genres=st.session_state["clf_orig_genres"],
                description=wi_desc.strip() or " ",
            )
            wi_probas = clf.predict_proba(wi_X_df)[0]

        fig_wi = whatif_comparison_chart(
            classes=st.session_state["clf_classes"],
            probas_orig=st.session_state["clf_orig_probas"],
            probas_new=wi_probas.tolist(),
        )
        st.plotly_chart(fig_wi, use_container_width=True, config={"displayModeBar": False})

        wi_pred = CLASS_LABELS[clf.predict(wi_X_df)[0]]
        delta_txt = ""
        orig_conf = max(st.session_state["clf_orig_probas"]) * 100
        new_conf  = max(wi_probas) * 100
        delta     = new_conf - orig_conf
        arrow     = "▲" if delta > 0 else "▼"
        color     = "#22c55e" if delta > 0 else "#E50914"
        st.markdown(
            f"<div style='color:#a0a0a0;font-size:.85rem;margin-top:.5rem;'>"
            f"What-If predicts <strong style='color:#f5f5f5;'>{wi_pred}</strong> · "
            f"Top confidence <span style='color:{color};font-weight:700;'>{arrow} {abs(delta):.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
