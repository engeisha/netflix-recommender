"""
Page 7 – System Diagnostics & Model Performance
================================================
• Confusion matrices and Precision / Recall / F1 for both classifiers
• Interactive 2-D PCA scatter of the full content catalogue coloured by cluster
• Optional t-SNE projection (computed on demand from PCA coords)
• Cluster topology statistics
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)
from sklearn.model_selection import train_test_split

from ui_theme import inject_global_css

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "netflix_titles.csv"
CLEANED_PATH = ROOT / "cleaned_netflix_titles.csv"

st.set_page_config(
    page_title="Diagnostics · Netflix AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_global_css()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{--bg:#0c0c0c;--text:#f5f5f5;--muted:#a0a0a0;
      --accent:#E50914;--border:#2a2a2a;}
.stApp{background:var(--bg);color:var(--text);}
.block-container{padding-top:1rem;padding-bottom:3rem;}
.sec-head{
    color:var(--text);font-size:1.05rem;font-weight:700;
    margin:1.6rem 0 .8rem;padding-bottom:.35rem;
    border-bottom:1px solid var(--border);
}
.diag-card{
    background:#141414;border:1px solid var(--border);
    border-radius:16px;padding:1rem 1.2rem;
}
.metric-pill{
    display:inline-block;padding:4px 12px;border-radius:999px;
    font-size:.83rem;font-weight:700;margin:3px 4px 3px 0;
    border:1px solid;
}
.note-box{
    background:rgba(229,9,20,.07);border:1px solid rgba(229,9,20,.25);
    border-radius:12px;padding:.7rem 1rem;
    color:#a0a0a0;font-size:.82rem;margin-top:.5rem;
}
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:.12em;margin-bottom:.3rem;'>📊 SYSTEM DIAGNOSTICS</div>"
    "<div style='color:#a0a0a0;font-size:.93rem;margin-bottom:1.4rem;'>"
    "Model performance cards · Confusion matrices · "
    "Content catalogue topology (PCA / t-SNE)</div>",
    unsafe_allow_html=True,
)

# ── dark layout helper ────────────────────────────────────────────────────────
def _dl(height: int = 380, title: str = "") -> dict:
    return dict(
        paper_bgcolor="#0a0a0a", plot_bgcolor="#111",
        font=dict(color="#c0c0c0", size=12),
        margin=dict(l=20, r=20, t=44 if title else 20, b=20),
        xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
        yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
        height=height,
        title=dict(text=title, font=dict(color="#f5f5f5", size=13)) if title else {},
        legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", borderwidth=1,
                    font=dict(size=11)),
    )

PALETTE = ["#E50914","#3b82f6","#22c55e","#f59e0b",
           "#a855f7","#ec4899","#14b8a6","#f97316"]


# ══════════════════════════════════════════════════════════════════════════════
# ① Content-Type Classifier diagnostics
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>① Content-Type Classifier (Movie vs TV Show)</div>",
            unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def clf_metrics():
    from data_pipeline import prepare_data
    clf = joblib.load(ROOT / "best_classifier.pkl")
    X_train, X_test, y_train, y_test = prepare_data(str(DATASET_PATH))
    y_pred = clf.predict(X_test)
    cm   = confusion_matrix(y_test, y_pred, labels=[0, 1])
    acc  = accuracy_score(y_test, y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1]
    )
    return cm, acc, p, r, f, s

try:
    cm1, acc1, p1, r1, f1_s, sup1 = clf_metrics()
    classes1 = ["Movie", "TV Show"]

    # performance cards
    mc_html = "<div style='display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;'>"
    for i, cls in enumerate(classes1):
        mc_html += (
            f"<div class='diag-card' style='flex:1;min-width:160px;'>"
            f"<div style='color:#a0a0a0;font-size:.76rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;'>{cls}</div>"
            f"<div style='display:flex;gap:1rem;flex-wrap:wrap;'>"
            f"<div><div style='color:#3b82f6;font-size:1.3rem;font-weight:900;'>{p1[i]*100:.1f}%</div>"
            f"<div style='color:#a0a0a0;font-size:.75rem;'>Precision</div></div>"
            f"<div><div style='color:#22c55e;font-size:1.3rem;font-weight:900;'>{r1[i]*100:.1f}%</div>"
            f"<div style='color:#a0a0a0;font-size:.75rem;'>Recall</div></div>"
            f"<div><div style='color:#f59e0b;font-size:1.3rem;font-weight:900;'>{f1_s[i]*100:.1f}%</div>"
            f"<div style='color:#a0a0a0;font-size:.75rem;'>F1</div></div>"
            f"<div><div style='color:#f5f5f5;font-size:1.3rem;font-weight:900;'>{int(sup1[i])}</div>"
            f"<div style='color:#a0a0a0;font-size:.75rem;'>Support</div></div>"
            f"</div></div>"
        )
    mc_html += f"<div class='diag-card' style='flex:1;min-width:140px;'>"
    mc_html += (f"<div style='color:#a0a0a0;font-size:.76rem;font-weight:700;"
                f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;'>Overall</div>"
                f"<div style='color:#E50914;font-size:1.5rem;font-weight:900;'>{acc1*100:.1f}%</div>"
                f"<div style='color:#a0a0a0;font-size:.75rem;'>Accuracy</div></div>")
    mc_html += "</div>"
    st.markdown(mc_html, unsafe_allow_html=True)

    if acc1 >= 0.999:
        st.markdown(
            "<div class='note-box'>⚠ Perfect test-set accuracy suggests the test split "
            "overlaps with training data (same cleaned CSV). Results reflect in-distribution "
            "performance; treat as an upper bound.</div>",
            unsafe_allow_html=True,
        )

    # confusion matrix heatmap
    fig_cm1 = ff.create_annotated_heatmap(
        z=cm1.tolist(),
        x=["Pred: Movie", "Pred: TV Show"],
        y=["True: Movie", "True: TV Show"],
        colorscale=[[0, "#111111"], [1, "#E50914"]],
        showscale=True,
        font_colors=["#f5f5f5"],
    )
    fig_cm1.update_layout(
        **_dl(300, "Confusion Matrix — Content-Type Classifier"),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig_cm1, use_container_width=True, config={"displayModeBar": False})

except Exception as e:
    st.warning(f"Could not load classifier: {e}", icon="⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# ② Rating Predictor diagnostics
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>② Audience Rating Predictor (4-Class)</div>",
            unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def rating_metrics():
    from task3_rating_classifier import load_and_prepare
    rc  = joblib.load(ROOT / "rating_classifier.pkl")
    meta = joblib.load(ROOT / "rating_preprocessor.pkl")
    df3  = load_and_prepare(str(DATASET_PATH))
    feat = ["type", "release_year", "duration", "listed_in", "country", "description"]
    X3 = df3[feat]
    y3 = df3["rating_group"]
    _, X3te, _, y3te = train_test_split(
        X3, y3, test_size=0.2, random_state=42, stratify=y3
    )
    y3pred = rc.predict(X3te)
    classes = list(rc.classes_)
    cm3  = confusion_matrix(y3te, y3pred, labels=classes)
    acc3 = accuracy_score(y3te, y3pred)
    p3, r3, f3, s3 = precision_recall_fscore_support(
        y3te, y3pred, average=None, labels=classes
    )
    return cm3, acc3, p3, r3, f3, s3, classes, meta

try:
    cm3, acc3, p3, r3, f3_s, sup3, classes3, meta3 = rating_metrics()
    CLASS_COLORS3 = {"Adult":"#E50914","Teen":"#f59e0b",
                     "Older Kids":"#3b82f6","Kids / Family":"#22c55e"}

    # stored metrics vs live metrics comparison
    stored = meta3.get("metrics", {})
    best3  = meta3.get("best_model_name", "")

    mcards = "<div style='display:flex;gap:.8rem;flex-wrap:wrap;margin-bottom:1rem;'>"
    for name, m in stored.items():
        is_best = name == best3
        border  = "rgba(229,9,20,.5)" if is_best else "#2a2a2a"
        tag     = " <span style='color:#E50914;font-size:.73rem;font-weight:700;'>★ BEST</span>" if is_best else ""
        mcards += (
            f"<div class='diag-card' style='flex:1;min-width:170px;border-color:{border};'>"
            f"<div style='color:#a0a0a0;font-size:.76rem;font-weight:700;"
            f"text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;'>{name}{tag}</div>"
            f"<div style='color:#E50914;font-size:1.4rem;font-weight:900;'>{m['accuracy']*100:.1f}%</div>"
            f"<div style='color:#a0a0a0;font-size:.75rem;'>Accuracy</div>"
            f"<div style='margin-top:.4rem;color:#a0a0a0;font-size:.82rem;'>"
            f"F1: <strong style='color:#f5f5f5;'>{m['f1']*100:.1f}%</strong></div>"
            f"</div>"
        )
    mcards += "</div>"
    st.markdown(mcards, unsafe_allow_html=True)

    # per-class precision / recall / F1 table
    prf_df = pd.DataFrame({
        "Class":     classes3,
        "Precision": (p3 * 100).round(1),
        "Recall":    (r3 * 100).round(1),
        "F1":        (f3_s * 100).round(1),
        "Support":   sup3.astype(int),
    })
    st.dataframe(prf_df.set_index("Class"), use_container_width=True)

    # confusion matrix heatmap
    fig_cm3 = ff.create_annotated_heatmap(
        z=cm3.tolist(),
        x=[f"Pred: {c}" for c in classes3],
        y=[f"True: {c}" for c in classes3],
        colorscale=[[0, "#111111"], [0.5, "#1e3a5a"], [1, "#3b82f6"]],
        showscale=True,
        font_colors=["#f5f5f5"],
    )
    fig_cm3.update_layout(
        **_dl(380, "Confusion Matrix — Audience Rating Predictor"),
        xaxis=dict(side="bottom", tickangle=-20),
    )
    st.plotly_chart(fig_cm3, use_container_width=True, config={"displayModeBar": False})

    # per-class bar chart
    fig_prf = go.Figure()
    bar_colors = {"Precision": "#3b82f6", "Recall": "#22c55e", "F1": "#f59e0b"}
    for metric, color in bar_colors.items():
        fig_prf.add_trace(go.Bar(
            name=metric, x=prf_df["Class"],
            y=prf_df[metric],
            marker_color=color,
            hovertemplate=f"<b>{metric}</b> · %{{x}}: %{{y:.1f}}%<extra></extra>",
        ))
    fig_prf.update_layout(
        **_dl(320, "Per-Class Precision / Recall / F1"),
        barmode="group",
        yaxis=dict(title="Score (%)", gridcolor="#1e1e1e"),
        xaxis=dict(title=None),
    )
    st.plotly_chart(fig_prf, use_container_width=True, config={"displayModeBar": False})

except Exception as e:
    st.warning(f"Could not load rating classifier: {e}", icon="⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# ③ Catalogue Topology — PCA / t-SNE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>③ Content Catalogue Topology</div>",
            unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_cluster_df():
    p = ROOT / "clustered_df.pkl"
    if not p.exists():
        return None
    return joblib.load(p)

cdf = load_cluster_df()

if cdf is None:
    st.warning("clustered_df.pkl not found. Run task4_content_segmentation.py first.", icon="⚠️")
else:
    # sidebar controls scoped to this section
    topo_col, ctrl_col = st.columns([4, 1])
    with ctrl_col:
        proj_mode = st.radio("Projection", ["2-D PCA", "3-D PCA", "t-SNE (2-D)"],
                             key="proj_mode")
        color_by  = st.radio("Color by", ["Cluster", "Type"], key="color_by")
        max_pts   = st.slider("Max points", 1000, len(cdf), min(4000, len(cdf)),
                              500, key="max_pts")
        show_centroids = st.toggle("Show centroids", value=True, key="show_centroids")

    sample_df = cdf.sample(min(max_pts, len(cdf)), random_state=42).copy()
    sample_df["cluster_label"] = (
        sample_df["cluster"].astype(str) + " · " + sample_df["persona"]
    )

    with topo_col:
        if proj_mode == "t-SNE (2-D)":
            @st.cache_data(show_spinner=False, max_entries=2)
            def compute_tsne(n: int):
                sub = cdf.sample(min(n, 3000), random_state=42).copy()
                coords = TSNE(
                    n_components=2, perplexity=30, random_state=42,
                    n_iter=500, learning_rate="auto", init="pca",
                ).fit_transform(sub[["pca_x", "pca_y"]].values)
                sub["tsne_x"] = coords[:, 0]
                sub["tsne_y"] = coords[:, 1]
                return sub

            with st.spinner("Computing t-SNE (this may take ~10 s)…"):
                tsne_df = compute_tsne(max_pts)
            tsne_df["cluster_label"] = (
                tsne_df["cluster"].astype(str) + " · " + tsne_df["persona"]
            )

            color_col = "cluster_label" if color_by == "Cluster" else "type"
            fig_topo = px.scatter(
                tsne_df, x="tsne_x", y="tsne_y",
                color=color_col,
                color_discrete_sequence=PALETTE,
                hover_data={"title": True, "type": True,
                            "rating": True, "release_year": True,
                            "tsne_x": False, "tsne_y": False,
                            "cluster_label": color_by == "Cluster"},
                labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2",
                        "cluster_label": "Cluster"},
                opacity=0.72, title="t-SNE Projection of Netflix Catalogue",
            )
            fig_topo.update_traces(marker=dict(size=4))
            fig_topo.update_layout(**_dl(520))
            st.plotly_chart(fig_topo, use_container_width=True)

        elif proj_mode == "3-D PCA":
            color_col = "cluster_label" if color_by == "Cluster" else "type"
            # add a pseudo 3rd axis from cluster id (no full 3-D PCA stored)
            sample_df["pca_z"] = (
                sample_df["cluster"].astype(float)
                + np.random.default_rng(42).normal(0, 0.15, len(sample_df))
            )
            fig_topo = px.scatter_3d(
                sample_df, x="pca_x", y="pca_y", z="pca_z",
                color=color_col,
                color_discrete_sequence=PALETTE,
                hover_data={"title": True, "type": True,
                            "rating": True, "release_year": True,
                            "cluster_label": color_by == "Cluster",
                            "pca_x": False, "pca_y": False, "pca_z": False},
                labels={"pca_x": "PC1", "pca_y": "PC2", "pca_z": "Cluster axis",
                        "cluster_label": "Cluster"},
                opacity=0.75, title="3-D PCA View",
            )
            fig_topo.update_traces(marker=dict(size=3))
            fig_topo.update_layout(
                paper_bgcolor="#0a0a0a",
                font=dict(color="#c0c0c0", size=11),
                height=560,
                scene=dict(
                    bgcolor="#111",
                    xaxis=dict(gridcolor="#222", zerolinecolor="#333",
                               title="PC1"),
                    yaxis=dict(gridcolor="#222", zerolinecolor="#333",
                               title="PC2"),
                    zaxis=dict(gridcolor="#222", zerolinecolor="#333",
                               title="Cluster axis"),
                ),
                legend=dict(bgcolor="#161616", bordercolor="#2a2a2a",
                            borderwidth=1, font=dict(size=10)),
            )
            st.plotly_chart(fig_topo, use_container_width=True)

        else:  # 2-D PCA (default)
            color_col = "cluster_label" if color_by == "Cluster" else "type"
            fig_topo = px.scatter(
                sample_df, x="pca_x", y="pca_y",
                color=color_col,
                color_discrete_sequence=PALETTE,
                hover_data={"title": True, "type": True,
                            "rating": True, "release_year": True,
                            "pca_x": False, "pca_y": False,
                            "cluster_label": color_by == "Cluster"},
                labels={"pca_x": "PCA Component 1", "pca_y": "PCA Component 2",
                        "cluster_label": "Cluster"},
                opacity=0.72, title="2-D PCA Projection — Netflix Catalogue",
            )
            fig_topo.update_traces(marker=dict(size=5))
            fig_topo.update_layout(**_dl(520))

            # centroid overlays
            if show_centroids:
                centroids = (
                    cdf.groupby(["cluster", "persona"])[["pca_x", "pca_y"]]
                    .mean().reset_index()
                )
                fig_topo.add_trace(go.Scatter(
                    x=centroids["pca_x"], y=centroids["pca_y"],
                    mode="markers+text",
                    marker=dict(size=16, symbol="diamond",
                                color="rgba(255,255,255,0.9)",
                                line=dict(color="#E50914", width=2)),
                    text=centroids["cluster"].astype(str),
                    textfont=dict(color="#0a0a0a", size=9, family="Segoe UI"),
                    textposition="middle center",
                    name="Centroids",
                    hovertext=centroids["persona"],
                    hovertemplate="<b>%{hovertext}</b><extra></extra>",
                    showlegend=True,
                ))

            st.plotly_chart(fig_topo, use_container_width=True)

    # cluster statistics table
    st.markdown(
        "<div style='color:#f5f5f5;font-size:.93rem;font-weight:700;"
        "margin:1.2rem 0 .6rem;'>Cluster Statistics</div>",
        unsafe_allow_html=True,
    )
    stats = (
        cdf.groupby(["cluster", "persona"])
        .agg(
            Count=("title", "count"),
            Movies=("type", lambda x: (x == "Movie").sum()),
            TV_Shows=("type", lambda x: (x == "TV Show").sum()),
            Avg_Year=("release_year", "mean"),
        )
        .reset_index()
        .rename(columns={"persona": "Persona", "cluster": "ID"})
    )
    stats["Avg_Year"] = stats["Avg_Year"].round(0).astype(int)
    stats["Movie %"] = (stats["Movies"] / stats["Count"] * 100).round(1)
    st.dataframe(
        stats[["ID", "Persona", "Count", "Movies", "TV_Shows", "Movie %", "Avg_Year"]]
        .set_index("ID"),
        use_container_width=True,
        hide_index=False,
    )
