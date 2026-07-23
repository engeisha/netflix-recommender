"""
Page 4 – Netflix Content Segmentation
Interactive cluster explorer: scatter plot, cluster deep-dive, and
"assign new content to a cluster" tool.
"""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.sparse import csr_matrix, hstack

from ui_theme import inject_global_css

ROOT = Path(__file__).resolve().parent.parent

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Content Clusters · Netflix AI",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --bg:#0c0c0c; --panel:#1e1e1e; --text:#f5f5f5;
        --muted:#a0a0a0; --accent:#E50914; --border:#2a2a2a;
    }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { padding-top: 1rem; padding-bottom: 3rem; }

    /* section headers */
    .sec-head {
        color: var(--text); font-size: 1.05rem; font-weight: 700;
        margin: 1.6rem 0 0.8rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }

    /* cluster cards */
    .cl-card {
        background: #141414; border: 1px solid var(--border);
        border-radius: 16px; padding: 1.1rem 1.2rem; height: 100%;
    }
    .cl-title { font-size: 1rem; font-weight: 800; margin-bottom: 0.35rem; }
    .cl-meta  { color: var(--muted); font-size: 0.82rem; line-height: 1.5; }
    .cl-tag {
        display: inline-block; padding: 0.2rem 0.6rem;
        border-radius: 999px; font-size: 0.76rem; font-weight: 600;
        background: rgba(229,9,20,0.12); color: #E50914;
        border: 1px solid rgba(229,9,20,0.25); margin: 0.15rem 0.15rem 0 0;
    }

    /* assign result box */
    .assign-box {
        background: #131313; border: 1px solid var(--border);
        border-radius: 18px; padding: 1.4rem 1.6rem; margin-top: 0.8rem;
    }
    .assign-title { font-size: 1.6rem; font-weight: 900; margin-bottom: 0.2rem; }

    /* confidence bars */
    .bar-row { display:flex; align-items:center; gap:0.55rem; margin-bottom:0.4rem; }
    .bar-label { color:#c0c0c0; font-size:0.82rem; flex:0 0 220px; overflow:hidden;
                 text-overflow:ellipsis; white-space:nowrap; }
    .bar-track { flex:1; height:8px; border-radius:999px; background:#252525; }
    .bar-fill  { height:8px; border-radius:999px; background:#E50914; }
    .bar-pct   { color:#f5f5f5; font-size:0.8rem; flex:0 0 40px; text-align:right; }

    /* textarea / input contrast */
    textarea, div[data-baseweb="textarea"] textarea {
        color: #FFFFFF !important;
        background-color: #1F1F1F !important;
        border: 1px solid #333333 !important;
    }
    textarea::placeholder { color: #888888 !important; }
    input[type="number"], div[data-baseweb="input"] input {
        color: #FFFFFF !important;
        background-color: #1F1F1F !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:0.12em;margin-bottom:0.3rem;'>🧩 CONTENT SEGMENTATION</div>"
    "<div style='color:#a0a0a0;font-size:0.93rem;margin-bottom:1.4rem;'>"
    "K-Means clustering of 8 800+ Netflix titles visualised in 2-D PCA space.</div>",
    unsafe_allow_html=True,
)

# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    paths = {
        "kmeans":   ROOT / "kmeans_model.pkl",
        "pca":      ROOT / "pca_model.pkl",
        "pipeline": ROOT / "clustering_pipeline.pkl",
        "df":       ROOT / "clustered_df.pkl",
    }
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        return None
    return {k: joblib.load(p) for k, p in paths.items()}

arts = load_artifacts()

if arts is None:
    st.warning(
        "Clustering artifacts not found. Run `python task4_content_segmentation.py` "
        "from the project root first.",
        icon="⚠️",
    )
    st.stop()

kmeans:   object = arts["kmeans"]
pca:      object = arts["pca"]
pipeline: dict   = arts["pipeline"]
df:  pd.DataFrame = arts["df"].copy()

summary: dict = pipeline.get("cluster_summary", {})
k:        int = pipeline.get("k", len(df["cluster"].unique()))

# ── palette ────────────────────────────────────────────────────────────────────
PALETTE = [
    "#E50914", "#3b82f6", "#22c55e", "#f59e0b",
    "#a855f7", "#ec4899", "#14b8a6", "#f97316",
]

def cluster_color(cid: int) -> str:
    return PALETTE[int(cid) % len(PALETTE)]

# ── Section 1: scatter plot ────────────────────────────────────────────────────
st.markdown("<div class='sec-head'>① Interactive PCA Cluster Map</div>", unsafe_allow_html=True)

# sample for performance if very large
plot_df = df.sample(min(4000, len(df)), random_state=42) if len(df) > 4000 else df.copy()
plot_df["cluster_label"] = plot_df["cluster"].astype(str) + " · " + plot_df["persona"]
plot_df["duration_display"] = plot_df["duration"].fillna("N/A")

fig = px.scatter(
    plot_df,
    x="pca_x", y="pca_y",
    color="cluster_label",
    color_discrete_sequence=PALETTE,
    hover_data={
        "title":        True,
        "type":         True,
        "listed_in":    True,
        "rating":       True,
        "release_year": True,
        "pca_x":        False,
        "pca_y":        False,
        "cluster_label": False,
    },
    labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2",
            "cluster_label": "Cluster"},
    opacity=0.72,
)
fig.update_traces(marker=dict(size=5))
fig.update_layout(
    paper_bgcolor="#0c0c0c",
    plot_bgcolor="#111111",
    font=dict(color="#c0c0c0", size=12),
    legend=dict(
        bgcolor="#161616", bordercolor="#2a2a2a", borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
    yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# cluster size bar chart
st.markdown("<div class='sec-head'>Cluster Size Distribution</div>", unsafe_allow_html=True)
size_data = (
    df.groupby(["cluster", "persona"])
    .size()
    .reset_index(name="count")
    .sort_values("cluster")
)
size_data["label"] = size_data["cluster"].astype(str) + " · " + size_data["persona"]

fig2 = px.bar(
    size_data, x="label", y="count",
    color="label",
    color_discrete_sequence=PALETTE,
    labels={"label": "Cluster", "count": "Title Count"},
)
fig2.update_layout(
    paper_bgcolor="#0c0c0c", plot_bgcolor="#111111",
    font=dict(color="#c0c0c0", size=12),
    showlegend=False,
    margin=dict(l=20, r=20, t=10, b=80),
    xaxis=dict(gridcolor="#1e1e1e", tickangle=-25),
    yaxis=dict(gridcolor="#1e1e1e"),
    height=320,
)
st.plotly_chart(fig2, use_container_width=True)

# ── Section 2: cluster deep-dive tabs ─────────────────────────────────────────
st.markdown("<div class='sec-head'>② Cluster Deep-Dive</div>", unsafe_allow_html=True)

tab_labels = [
    f"{summary[cid]['persona']}" for cid in sorted(summary.keys())
]
tabs = st.tabs(tab_labels)

for tab, cid in zip(tabs, sorted(summary.keys())):
    with tab:
        s = summary[cid]
        col_info, col_titles = st.columns([3, 2])

        with col_info:
            color = cluster_color(cid)
            type_split = s.get("type_split", {})
            movie_cnt  = type_split.get("Movie", 0)
            tv_cnt     = type_split.get("TV Show", 0)
            total      = s["count"]

            genre_tags = "".join(
                f"<span class='cl-tag'>{g}</span>" for g in s["top_genres"]
            )
            country_tags = "".join(
                f"<span style='display:inline-block;padding:0.2rem 0.6rem;"
                f"border-radius:999px;font-size:0.76rem;font-weight:600;"
                f"background:rgba(59,130,246,0.1);color:#3b82f6;"
                f"border:1px solid rgba(59,130,246,0.25);"
                f"margin:0.15rem 0.15rem 0 0;'>{c}</span>"
                for c in s["top_countries"]
            )

            st.markdown(
                f"""
                <div class='cl-card'>
                    <div class='cl-title' style='color:{color};'>{s['persona']}</div>
                    <div style='display:flex;gap:1.2rem;margin:0.5rem 0 0.8rem;
                                flex-wrap:wrap;color:#c0c0c0;font-size:0.86rem;'>
                        <span>🎬 {movie_cnt} Movies</span>
                        <span>📺 {tv_cnt} TV Shows</span>
                        <span>📅 Avg year: {s['avg_year']}</span>
                        <span>🏅 {s['common_rating']}</span>
                        <span>Total: {total}</span>
                    </div>
                    <div style='margin-bottom:0.4rem;color:#a0a0a0;font-size:0.78rem;
                                font-weight:700;text-transform:uppercase;letter-spacing:0.08em;'>
                        Top Genres
                    </div>
                    <div style='margin-bottom:0.8rem;'>{genre_tags}</div>
                    <div style='margin-bottom:0.4rem;color:#a0a0a0;font-size:0.78rem;
                                font-weight:700;text-transform:uppercase;letter-spacing:0.08em;'>
                        Top Countries
                    </div>
                    <div>{country_tags}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col_titles:
            st.markdown(
                "<div style='color:#a0a0a0;font-size:0.78rem;font-weight:700;"
                "text-transform:uppercase;letter-spacing:0.08em;"
                "margin-bottom:0.5rem;'>Sample Titles</div>",
                unsafe_allow_html=True,
            )
            for title in s["sample_titles"]:
                st.markdown(
                    f"<div style='color:#e0e0e0;font-size:0.9rem;padding:0.35rem 0;"
                    f"border-bottom:1px solid #1e1e1e;'>▸ {title}</div>",
                    unsafe_allow_html=True,
                )

            # mini type pie
            if movie_cnt + tv_cnt > 0:
                pie = go.Figure(go.Pie(
                    labels=["Movies", "TV Shows"],
                    values=[movie_cnt, tv_cnt],
                    hole=0.55,
                    marker_colors=[cluster_color(cid), "#2a2a2a"],
                    textinfo="percent",
                    textfont=dict(color="#f5f5f5", size=11),
                ))
                pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    height=160,
                    showlegend=True,
                    legend=dict(font=dict(color="#a0a0a0", size=10),
                                bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(pie, use_container_width=True)


# ── Section 3: assign new content to cluster ──────────────────────────────────
st.markdown("<div class='sec-head'>③ Assign Content to a Cluster</div>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#a0a0a0;font-size:0.9rem;margin-bottom:1rem;'>"
    "Enter content attributes and find which cluster it belongs to.</div>",
    unsafe_allow_html=True,
)

# derive UI option lists from the training data
type_options    = ["Movie", "TV Show"]
rating_options  = sorted(df["rating"][df["rating"] != "Unknown"].unique().tolist())
country_options = sorted(df["country_primary"][df["country_primary"] != "Other"].unique().tolist())
genre_options   = sorted(
    {g.strip() for genres in df["listed_in"].dropna() for g in genres.split(",")}
)

with st.form("assign_form"):
    a1, a2 = st.columns(2)

    with a1:
        a_type    = st.selectbox("Content Type", type_options)
        a_year    = st.number_input("Release Year", min_value=1925, max_value=2026, value=2021, step=1)
        ad1, ad2  = st.columns([2, 1])
        with ad1:
            a_dur  = st.number_input("Duration", min_value=1, max_value=500, value=90, step=1)
        with ad2:
            a_unit = st.selectbox("Unit", ["Minutes", "Seasons"])
        a_rating   = st.selectbox("Content Rating", rating_options,
                                   index=rating_options.index("TV-MA") if "TV-MA" in rating_options else 0)

    with a2:
        a_genres  = st.multiselect(
            "Genres", options=genre_options,
            default=["Dramas", "International Movies"],
        )
        a_country = st.selectbox(
            "Country", options=country_options,
            index=country_options.index("United States") if "United States" in country_options else 0,
        )
        a_desc    = st.text_area(
            "Plot Description",
            height=130,
            placeholder="e.g. A detective investigates a series of mysterious disappearances…",
        )

    assign_btn = st.form_submit_button(
        "🔍 Find My Cluster", type="primary", use_container_width=True
    )

if assign_btn:
    listed_str = ", ".join(sorted(a_genres)) if a_genres else "Unknown"

    row_dict = {
        "type":            a_type,
        "release_year":    float(a_year),
        "duration_num":    float(a_dur),
        "rating":          a_rating,
        "listed_in":       listed_str,
        "country_primary": a_country,
        "description":     a_desc.strip() if a_desc else "",
    }

    with st.spinner("Computing cluster assignment…"):
        X_new   = pipeline["tfidf"].transform([row_dict["description"]])
        X_cat   = pipeline["ohe"].transform(
            pd.DataFrame([[a_type, a_rating, a_country]],
                         columns=["type", "rating", "country_primary"])
        )
        X_genre = pipeline["genre_tfidf"].transform([listed_str])
        X_num   = csr_matrix(
            pipeline["scaler"].transform(
                pd.DataFrame([[float(a_year), float(a_dur)]],
                             columns=["release_year", "duration_num"])
            )
        )
        X_row   = hstack([X_new, X_cat, X_genre, X_num])

        pred_cluster = int(kmeans.predict(X_row)[0])

        # distances to all centroids → softmax for pseudo-probabilities
        dists = kmeans.transform(X_row)[0]
        inv   = 1.0 / (dists + 1e-9)
        probs = inv / inv.sum()

    s_pred  = summary.get(pred_cluster, {})
    persona = s_pred.get("persona", f"Cluster {pred_cluster}")
    color   = cluster_color(pred_cluster)

    # top-3 clusters by probability
    top3 = sorted(enumerate(probs), key=lambda x: -x[1])[:3]

    bar_html = ""
    for cid_b, prob_b in top3:
        s_b    = summary.get(cid_b, {})
        p_b    = s_b.get("persona", f"Cluster {cid_b}")
        pct_b  = prob_b * 100
        col_b  = cluster_color(cid_b)
        bar_html += (
            f"<div class='bar-row'>"
            f"<div class='bar-label'>{p_b}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{pct_b:.1f}%;background:{col_b};'></div></div>"
            f"<div class='bar-pct'>{pct_b:.1f}%</div>"
            f"</div>"
        )

    genre_tags = "".join(
        f"<span class='cl-tag'>{g}</span>" for g in s_pred.get("top_genres", [])
    )

    st.markdown(
        f"""
        <div class='assign-box'>
            <div class='assign-title' style='color:{color};'>{persona}</div>
            <div style='color:#a0a0a0;font-size:0.88rem;margin-bottom:0.9rem;'>
                Best matching cluster · {s_pred.get('count','?')} titles in this group
            </div>
            <div style='margin-bottom:0.35rem;color:#a0a0a0;font-size:0.78rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.08em;'>Top Genres in Cluster</div>
            <div style='margin-bottom:1rem;'>{genre_tags}</div>
            <div style='margin-bottom:0.45rem;color:#a0a0a0;font-size:0.78rem;font-weight:700;
                        text-transform:uppercase;letter-spacing:0.08em;'>Cluster Similarity</div>
            {bar_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # highlight the predicted cluster on the scatter
    st.markdown(
        "<div style='color:#f5f5f5;font-size:0.95rem;font-weight:700;"
        "margin:1.4rem 0 0.6rem;'>Cluster Location on PCA Map</div>",
        unsafe_allow_html=True,
    )
    highlight_df = plot_df[plot_df["cluster"] == pred_cluster]
    rest_df      = plot_df[plot_df["cluster"] != pred_cluster]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=rest_df["pca_x"], y=rest_df["pca_y"],
        mode="markers",
        marker=dict(size=4, color="#2a2a2a"),
        name="Other clusters",
        hoverinfo="skip",
    ))
    fig3.add_trace(go.Scatter(
        x=highlight_df["pca_x"], y=highlight_df["pca_y"],
        mode="markers",
        marker=dict(size=6, color=color, opacity=0.85),
        name=persona,
        text=highlight_df["title"],
        hovertemplate="<b>%{text}</b><extra></extra>",
    ))
    fig3.update_layout(
        paper_bgcolor="#0c0c0c", plot_bgcolor="#111111",
        font=dict(color="#c0c0c0", size=12),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
        xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a",
                   title="PCA Component 1"),
        yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a",
                   title="PCA Component 2"),
        legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", borderwidth=1),
    )
    st.plotly_chart(fig3, use_container_width=True)
