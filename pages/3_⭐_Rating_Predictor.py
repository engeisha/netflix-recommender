"""
Page 3 – Netflix Audience Rating Predictor
Loads rating_classifier.pkl + rating_preprocessor.pkl and predicts the
audience-rating group for a piece of content based on its metadata.
"""

from __future__ import annotations
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ui_theme import inject_global_css

ROOT = Path(__file__).resolve().parent.parent

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rating Predictor · Netflix AI",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

st.markdown(
    """
    <style>
    :root {
        --bg:#0c0c0c; --panel:#1e1e1e; --text:#f5f5f5;
        --muted:#a0a0a0; --accent:#E50914; --border:#2a2a2a;
    }
    .stApp  { background: var(--bg); color: var(--text); }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }

    /* result card */
    .result-box {
        border: 1px solid var(--border); border-radius: 18px;
        padding: 1.5rem 1.7rem; margin-top: 1rem; background: #131313;
    }
    .result-label { font-size: 1.9rem; font-weight: 900; margin-bottom: 0.2rem; }
    .conf-track { height: 10px; border-radius: 999px; background: #2a2a2a; margin-top: 0.55rem; }
    .conf-fill  { height: 10px; border-radius: 999px; }
    .hint { color: var(--muted); font-size: 0.88rem; margin-top: 0.45rem; }

    /* metrics comparison */
    .metric-grid { display: flex; gap: 0.9rem; flex-wrap: wrap; margin-top: 1.2rem; }
    .metric-card {
        flex: 1; min-width: 160px;
        background: #171717; border: 1px solid var(--border);
        border-radius: 14px; padding: 0.9rem 1.1rem;
    }
    .metric-model { color: var(--muted); font-size: 0.78rem; font-weight: 700;
                    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem; }
    .metric-val   { color: var(--accent); font-size: 1.5rem; font-weight: 900; }
    .metric-lbl   { color: var(--muted); font-size: 0.8rem; }
    .metric-best  { border-color: rgba(229,9,20,0.5); }

    /* rating badges */
    .badge-adult  { color: #E50914; }
    .badge-teen   { color: #f59e0b; }
    .badge-older  { color: #3b82f6; }
    .badge-kids   { color: #22c55e; }

    /* prob bar row */
    .prob-row { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.45rem; }
    .prob-label { color:#c0c0c0; font-size:0.85rem; width:110px; flex-shrink:0; }
    .prob-track { flex:1; height:8px; border-radius:999px; background:#2a2a2a; }
    .prob-fill  { height:8px; border-radius:999px; }
    .prob-pct   { color:#f5f5f5; font-size:0.82rem; width:44px; text-align:right; flex-shrink:0; }

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

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:0.12em;margin-bottom:0.3rem;'>⭐ AUDIENCE RATING PREDICTOR</div>"
    "<div style='color:#a0a0a0;font-size:0.93rem;margin-bottom:1.4rem;'>"
    "Predict whether content is rated "
    "<span style='color:#E50914;font-weight:700;'>Adult</span>, "
    "<span style='color:#f59e0b;font-weight:700;'>Teen</span>, "
    "<span style='color:#3b82f6;font-weight:700;'>Older Kids</span>, or "
    "<span style='color:#22c55e;font-weight:700;'>Kids / Family</span> "
    "from its metadata.</div>",
    unsafe_allow_html=True,
)

# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    clf_path = ROOT / "rating_classifier.pkl"
    pre_path = ROOT / "rating_preprocessor.pkl"
    if not clf_path.exists() or not pre_path.exists():
        return None, None
    return joblib.load(clf_path), joblib.load(pre_path)

clf, meta = load_model()

if clf is None:
    st.warning(
        "Model files not found. Run `python task3_rating_classifier.py` from "
        "the project root to train and save `rating_classifier.pkl` and "
        "`rating_preprocessor.pkl`.",
        icon="⚠️",
    )
    st.stop()

# ── UI helpers ─────────────────────────────────────────────────────────────────
RATING_COLORS = {
    "Adult":         ("#E50914", "badge-adult"),
    "Teen":          ("#f59e0b", "badge-teen"),
    "Older Kids":    ("#3b82f6", "badge-older"),
    "Kids / Family": ("#22c55e", "badge-kids"),
}
BAR_COLORS = {
    "Adult":         "#E50914",
    "Teen":          "#f59e0b",
    "Older Kids":    "#3b82f6",
    "Kids / Family": "#22c55e",
}
RATING_ICONS = {
    "Adult": "🔞", "Teen": "👦", "Older Kids": "🧒", "Kids / Family": "👨‍👩‍👧",
}
RATING_DESC = {
    "Adult":         "TV-MA · R · NC-17 — Mature audiences only",
    "Teen":          "TV-14 · PG-13 — Parents strongly cautioned",
    "Older Kids":    "TV-PG · TV-Y7 — Parental guidance suggested",
    "Kids / Family": "TV-Y · TV-G · G · PG — Suitable for all ages",
}

# extract unique listed_in values for the selectbox
listed_options: list[str] = meta.get("listed_in_options", [])
country_options: list[str] = meta.get("country_options", [])

# ── input form ─────────────────────────────────────────────────────────────────
with st.form("rating_form"):
    c1, c2 = st.columns(2)

    with c1:
        content_type = st.selectbox("Content Type", ["Movie", "TV Show"])
        release_year = st.number_input(
            "Release Year", min_value=1925, max_value=2026, value=2020, step=1
        )
        dur_c1, dur_c2 = st.columns([2, 1])
        with dur_c1:
            duration_val = st.number_input(
                "Duration", min_value=1, max_value=500, value=90, step=1
            )
        with dur_c2:
            dur_unit = st.selectbox("Unit", ["Minutes", "Seasons"])

    with c2:
        # listed_in as free text + selectbox for known values
        listed_in = st.selectbox(
            "Genres / Listed In",
            options=listed_options if listed_options else ["Dramas", "International Movies"],
            index=0,
        )
        country = st.selectbox(
            "Country of Origin",
            options=country_options if country_options else ["United States"],
            index=0,
        )
        description = st.text_area(
            "Plot Description",
            height=130,
            placeholder="e.g. A gritty crime drama following a detective who uncovers a city-wide conspiracy…",
        )

    predict_btn = st.form_submit_button(
        "⭐ Predict Audience Rating", type="primary", use_container_width=True
    )

# ── prediction ─────────────────────────────────────────────────────────────────
if predict_btn:
    if not description.strip():
        st.error("Please enter a plot description before predicting.")
    else:
        duration_numeric = int(duration_val)

        row = pd.DataFrame([{
            "type":         content_type,
            "release_year": int(release_year),
            "duration":     duration_numeric,
            "listed_in":    listed_in,
            "country":      country,
            "description":  description.strip(),
        }])

        with st.spinner("Running prediction…"):
            pred_label = clf.predict(row)[0]
            probas     = clf.predict_proba(row)[0]
            classes    = clf.classes_

        conf       = float(probas[list(classes).index(pred_label)]) * 100
        color, _   = RATING_COLORS.get(pred_label, ("#f5f5f5", ""))
        bar_color  = BAR_COLORS.get(pred_label, "#E50914")
        icon       = RATING_ICONS.get(pred_label, "⭐")
        desc_line  = RATING_DESC.get(pred_label, "")

        # ── result card ────────────────────────────────────────────────────────
        prob_bars_html = ""
        for cls, prob in sorted(zip(classes, probas), key=lambda x: -x[1]):
            pct    = prob * 100
            bcol   = BAR_COLORS.get(cls, "#888")
            prob_bars_html += (
                f"<div class='prob-row'>"
                f"<div class='prob-label'>{cls}</div>"
                f"<div class='prob-track'><div class='prob-fill' style='width:{pct:.1f}%;background:{bcol};'></div></div>"
                f"<div class='prob-pct'>{pct:.1f}%</div>"
                f"</div>"
            )

        st.markdown(
            f"""
            <div class="result-box">
                <div class="result-label" style="color:{color};">{icon} {pred_label}</div>
                <div style="color:#a0a0a0;font-size:0.88rem;margin-bottom:0.7rem;">{desc_line}</div>
                <div class="hint">Confidence: <strong style="color:#f5f5f5;">{conf:.1f}%</strong></div>
                <div class="conf-track">
                    <div class="conf-fill" style="width:{conf:.1f}%;background:{bar_color};"></div>
                </div>
                <div style="margin-top:1.1rem;color:#a0a0a0;font-size:0.82rem;font-weight:700;
                            text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem;">
                    All Class Probabilities
                </div>
                {prob_bars_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── model metrics comparison ────────────────────────────────────────────
        st.markdown(
            "<div style='color:#f5f5f5;font-size:1rem;font-weight:700;"
            "margin:1.6rem 0 0.8rem;'>Model Performance Comparison</div>",
            unsafe_allow_html=True,
        )

        metrics: dict = meta.get("metrics", {})
        best_name: str = meta.get("best_model_name", "")

        cards_html = "<div style='display:flex;gap:0.9rem;flex-wrap:wrap;margin-top:1.2rem;'>"
        for model_name, m in metrics.items():
            is_best = model_name == best_name
            border  = "rgba(229,9,20,0.5)" if is_best else "#2a2a2a"
            best_tag = (
                " <span style='color:#E50914;font-size:0.75rem;font-weight:700;'>★ BEST</span>"
                if is_best else ""
            )
            cards_html += (
                f"<div style='flex:1;min-width:160px;background:#171717;"
                f"border:1px solid {border};border-radius:14px;padding:0.9rem 1.1rem;'>"
                f"<div style='color:#a0a0a0;font-size:0.78rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.3rem;'>"
                f"{model_name}{best_tag}</div>"
                f"<div style='color:#E50914;font-size:1.5rem;font-weight:900;'>{m['accuracy']*100:.1f}%</div>"
                f"<div style='color:#a0a0a0;font-size:0.8rem;'>Accuracy</div>"
                f"<div style='margin-top:0.5rem;color:#a0a0a0;font-size:0.82rem;'>"
                f"F1 (weighted): <strong style='color:#f5f5f5;'>{m['f1']*100:.1f}%</strong></div>"
                f"</div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)
