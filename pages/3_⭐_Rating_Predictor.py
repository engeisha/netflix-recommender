"""
Page 3 – Netflix Audience Rating Predictor
With XAI Inspector, Multi-Class Confidence Breakdown, and What-If Analysis.
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ui_theme import inject_global_css
from xai_utils import (
    feature_impact_chart,
    proba_chart,
    whatif_comparison_chart,
    word_weights_from_rf_named,
)

ROOT = Path(__file__).resolve().parent.parent

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
    .stApp  { background:var(--bg); color:var(--text); }
    .block-container { padding-top:1rem; padding-bottom:3rem; }

    .result-box {
        border:1px solid var(--border); border-radius:18px;
        padding:1.5rem 1.7rem; margin-top:1rem; background:#131313;
    }
    .result-label { font-size:1.9rem; font-weight:900; margin-bottom:.2rem; }
    .conf-track { height:10px; border-radius:999px; background:#2a2a2a; margin-top:.55rem; }
    .conf-fill  { height:10px; border-radius:999px; }
    .hint { color:var(--muted); font-size:.88rem; margin-top:.45rem; }

    /* model metric cards */
    .metric-card {
        flex:1; min-width:160px; background:#171717;
        border:1px solid var(--border); border-radius:14px; padding:.9rem 1.1rem;
    }

    /* XAI glass */
    .xai-panel {
        background:linear-gradient(160deg,rgba(255,255,255,.04),rgba(255,255,255,.01));
        border:1px solid var(--border); border-radius:18px;
        padding:1.2rem 1.4rem; margin-top:1rem;
    }
    .xai-title {
        color:#f5f5f5; font-size:.93rem; font-weight:700;
        letter-spacing:.05em; text-transform:uppercase; margin-bottom:.7rem;
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
    "letter-spacing:.12em;margin-bottom:.3rem;'>⭐ AUDIENCE RATING PREDICTOR</div>"
    "<div style='color:#a0a0a0;font-size:.93rem;margin-bottom:1.4rem;'>"
    "Predict whether content is rated "
    "<span style='color:#E50914;font-weight:700;'>Adult</span>, "
    "<span style='color:#f59e0b;font-weight:700;'>Teen</span>, "
    "<span style='color:#3b82f6;font-weight:700;'>Older Kids</span>, or "
    "<span style='color:#22c55e;font-weight:700;'>Kids / Family</span>, "
    "with XAI breakdown.</div>",
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
        "Model files not found. Run `python task3_rating_classifier.py` to generate them.",
        icon="⚠️",
    )
    st.stop()

# ── constants ──────────────────────────────────────────────────────────────────
RATING_COLORS = {
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

listed_options: list[str] = meta.get("listed_in_options", [])
country_options: list[str] = meta.get("country_options", [])


# ── cached feature names ───────────────────────────────────────────────────────
@st.cache_resource
def get_feature_names() -> list[str]:
    return list(clf.named_steps["prep"].get_feature_names_out())

FEATURE_NAMES = get_feature_names()


# ── input form ─────────────────────────────────────────────────────────────────
with st.form("rating_form"):
    c1, c2 = st.columns(2)
    with c1:
        content_type = st.selectbox("Content Type", ["Movie", "TV Show"])
        release_year = st.number_input("Release Year", min_value=1925, max_value=2026, value=2020, step=1)
        dc1, dc2 = st.columns([2, 1])
        with dc1:
            duration_val = st.number_input("Duration", min_value=1, max_value=500, value=90, step=1)
        with dc2:
            dur_unit = st.selectbox("Unit", ["Minutes", "Seasons"])
    with c2:
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
            "Plot Description", height=130,
            placeholder="e.g. A gritty crime drama following a detective who uncovers a city-wide conspiracy…",
        )
    predict_btn = st.form_submit_button("⭐ Predict Audience Rating", type="primary", use_container_width=True)


# ── prediction ─────────────────────────────────────────────────────────────────
def make_input_row(
    ctype: str, year: int, dur: int,
    listed: str, ctry: str, desc: str,
) -> pd.DataFrame:
    return pd.DataFrame([{
        "type":         ctype,
        "release_year": int(year),
        "duration":     int(dur),
        "listed_in":    listed,
        "country":      ctry,
        "description":  desc,
    }])


if predict_btn:
    if not description.strip():
        st.error("Please enter a plot description before predicting.")
    else:
        row = make_input_row(
            content_type, int(release_year), int(duration_val),
            listed_in, country, description.strip(),
        )

        with st.spinner("Running prediction…"):
            pred_label = clf.predict(row)[0]
            probas     = clf.predict_proba(row)[0]
            classes    = list(clf.classes_)

            # get the transformed feature vector for XAI
            X_transformed = clf.named_steps["prep"].transform(row)
            if hasattr(X_transformed, "toarray"):
                X_transformed = X_transformed.toarray()
            x_vec = X_transformed[0]

        conf       = float(probas[classes.index(pred_label)]) * 100
        color      = RATING_COLORS.get(pred_label, "#f5f5f5")
        bar_color  = color
        icon       = RATING_ICONS.get(pred_label, "⭐")
        desc_line  = RATING_DESC.get(pred_label, "")

        # persist for what-if
        st.session_state["rp_orig_row"]    = row
        st.session_state["rp_orig_probas"] = probas.tolist()
        st.session_state["rp_classes"]     = classes
        st.session_state["rp_desc"]        = description.strip()
        st.session_state["rp_dur"]         = int(duration_val)
        st.session_state["rp_year"]        = int(release_year)
        st.session_state["rp_type"]        = content_type
        st.session_state["rp_listed"]      = listed_in
        st.session_state["rp_country"]     = country

        # ── result card ────────────────────────────────────────────────────────
        st.markdown(
            f"""
            <div class="result-box">
              <div class="result-label" style="color:{color};">{icon} {pred_label}</div>
              <div style="color:#a0a0a0;font-size:.88rem;margin-bottom:.7rem;">{desc_line}</div>
              <div class="hint">Confidence: <strong style="color:#f5f5f5;">{conf:.1f}%</strong></div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{conf:.1f}%;background:{bar_color};"></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── XAI panels ────────────────────────────────────────────────────────
        xai_col, proba_col = st.columns([1, 1])

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
            word_df, xai_ms = word_weights_from_rf_named(
                feature_importances=clf.named_steps["clf"].feature_importances_,
                feature_names=FEATURE_NAMES,
                input_vector=x_vec,
                word_to_idx={},
                top_n=12,
            )
            # only keep text__ features for word chart
            if not word_df.empty:
                word_df = word_df[
                    word_df["word"].apply(
                        lambda w: not any(w.startswith(p) for p in ["type_", "listed_in_", "country_", "release", "duration"])
                    )
                ]
            fig_impact = feature_impact_chart(
                word_df,
                title=f"Top Words → {pred_label}",
                pos_color="#22c55e",
            )
            st.plotly_chart(fig_impact, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        with proba_col:
            st.markdown("<div class='xai-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='xai-title'>📊 Confidence Breakdown</div>", unsafe_allow_html=True)
            fig_proba = proba_chart(
                classes=classes,
                probas=probas.tolist(),
                bar_colors=RATING_COLORS,
                title="Probability Across All Rating Groups",
            )
            st.plotly_chart(fig_proba, use_container_width=True, config={"displayModeBar": False})
            st.markdown("</div>", unsafe_allow_html=True)

        # ── telemetry badge ────────────────────────────────────────────────────
        _rp_badge_c = "#22c55e" if xai_ms < 200 else ("#f59e0b" if xai_ms < 800 else "#E50914")
        st.markdown(
            f"<div style='margin-top:.5rem;'>"
            f"<span style='background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.25);"
            f"border-radius:999px;padding:3px 11px;font-size:.76rem;font-weight:700;"
            f"color:{_rp_badge_c};'>⚡ XAI computed in {xai_ms:.1f} ms</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── model performance cards ────────────────────────────────────────────
        metrics   = meta.get("metrics", {})
        best_name = meta.get("best_model_name", "")
        st.markdown(
            "<div style='color:#f5f5f5;font-size:1rem;font-weight:700;"
            "margin:1.6rem 0 .8rem;'>Model Performance Comparison</div>",
            unsafe_allow_html=True,
        )
        cards_html = "<div style='display:flex;gap:.9rem;flex-wrap:wrap;'>"
        for mname, m in metrics.items():
            is_best = mname == best_name
            border  = "rgba(229,9,20,.5)" if is_best else "#2a2a2a"
            best_tag = " <span style='color:#E50914;font-size:.75rem;font-weight:700;'>★ BEST</span>" if is_best else ""
            cards_html += (
                f"<div class='metric-card' style='border-color:{border};'>"
                f"<div style='color:#a0a0a0;font-size:.78rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;'>{mname}{best_tag}</div>"
                f"<div style='color:#E50914;font-size:1.5rem;font-weight:900;'>{m['accuracy']*100:.1f}%</div>"
                f"<div style='color:#a0a0a0;font-size:.8rem;'>Accuracy</div>"
                f"<div style='margin-top:.5rem;color:#a0a0a0;font-size:.82rem;'>"
                f"F1 (weighted): <strong style='color:#f5f5f5;'>{m['f1']*100:.1f}%</strong></div>"
                f"</div>"
            )
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)


# ── what-if analysis ───────────────────────────────────────────────────────────
if "rp_orig_probas" in st.session_state:
    st.markdown("<div class='whatif-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='whatif-title'>🧪 What-If Analysis</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='color:#a0a0a0;font-size:.85rem;margin-bottom:.9rem;'>"
        "Modify keywords or duration and see how the rating prediction shifts.</div>",
        unsafe_allow_html=True,
    )

    wi_c1, wi_c2 = st.columns([2, 1])
    with wi_c1:
        wi_desc = st.text_area(
            "Modified Description",
            value=st.session_state["rp_desc"],
            height=110,
            key="wi_rp_desc",
        )
    with wi_c2:
        wi_dur = st.number_input(
            "Duration", min_value=1, max_value=500,
            value=st.session_state["rp_dur"],
            key="wi_rp_dur",
        )
        wi_year = st.number_input(
            "Release Year", min_value=1925, max_value=2026,
            value=st.session_state["rp_year"],
            key="wi_rp_year",
        )

    if st.button("🔄 Run What-If", key="wi_rp_btn", type="primary"):
        with st.spinner("Re-running…"):
            wi_row = make_input_row(
                st.session_state["rp_type"],
                wi_year, wi_dur,
                st.session_state["rp_listed"],
                st.session_state["rp_country"],
                wi_desc.strip() or " ",
            )
            wi_probas  = clf.predict_proba(wi_row)[0]
            wi_pred    = clf.predict(wi_row)[0]

        fig_wi = whatif_comparison_chart(
            classes=st.session_state["rp_classes"],
            probas_orig=st.session_state["rp_orig_probas"],
            probas_new=wi_probas.tolist(),
            bar_colors=RATING_COLORS,
        )
        st.plotly_chart(fig_wi, use_container_width=True, config={"displayModeBar": False})

        orig_conf = max(st.session_state["rp_orig_probas"]) * 100
        new_conf  = max(wi_probas) * 100
        delta     = new_conf - orig_conf
        arrow     = "▲" if delta > 0 else "▼"
        c         = "#22c55e" if delta > 0 else "#E50914"
        wi_icon   = RATING_ICONS.get(wi_pred, "⭐")
        wi_color  = RATING_COLORS.get(wi_pred, "#f5f5f5")
        st.markdown(
            f"<div style='color:#a0a0a0;font-size:.85rem;margin-top:.5rem;'>"
            f"What-If predicts "
            f"<strong style='color:{wi_color};'>{wi_icon} {wi_pred}</strong> · "
            f"Top confidence <span style='color:{c};font-weight:700;'>{arrow} {abs(delta):.1f}%</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
