"""
Page 6 – Netflix Content Success Analytics Engine
Step 1: Feature Engineering overview
Step 2: Model performance comparison
Step 3: Feature importance
Step 4: Automated business insights (genre, country, trend, rating)
Step 5: Visual report + live prediction tool
"""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui_theme import inject_global_css

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Success Analytics · Netflix AI",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{--bg:#0c0c0c;--panel:#1e1e1e;--text:#f5f5f5;--muted:#a0a0a0;
      --accent:#E50914;--border:#2a2a2a;}
.stApp{background:var(--bg);color:var(--text);}
.block-container{padding-top:1rem;padding-bottom:3rem;}
.sec-head{color:var(--text);font-size:1.05rem;font-weight:700;
          margin:1.6rem 0 0.8rem;padding-bottom:0.35rem;
          border-bottom:1px solid var(--border);}
.kpi-grid{display:flex;gap:0.8rem;flex-wrap:wrap;margin-bottom:1.2rem;}
.kpi-card{flex:1;min-width:120px;background:#141414;
          border:1px solid var(--border);border-radius:14px;padding:0.85rem 1rem;}
.kpi-val{color:var(--accent);font-size:1.5rem;font-weight:900;}
.kpi-lbl{color:var(--muted);font-size:0.78rem;margin-top:0.1rem;}
.kpi-sub{color:#c0c0c0;font-size:0.76rem;margin-top:0.15rem;}
.model-grid{display:flex;gap:0.8rem;flex-wrap:wrap;margin-top:0.5rem;}
.model-card{flex:1;min-width:170px;background:#141414;
            border:1px solid var(--border);border-radius:14px;padding:0.9rem 1.1rem;}
.model-card.best{border-color:rgba(229,9,20,.55);}
.mc-name{color:var(--muted);font-size:0.76rem;font-weight:700;
         text-transform:uppercase;letter-spacing:.1em;margin-bottom:.3rem;}
.mc-val{color:var(--accent);font-size:1.4rem;font-weight:900;}
.mc-lbl{color:var(--muted);font-size:0.76rem;}
.mc-sub{margin-top:.4rem;font-size:.82rem;color:#c0c0c0;}
.insight-box{background:#131313;border:1px solid var(--border);
             border-radius:16px;padding:1.1rem 1.3rem;margin-bottom:.8rem;}
.pred-box{background:#131313;border:1px solid var(--border);
          border-radius:18px;padding:1.5rem 1.7rem;margin-top:.8rem;}
.pred-title{font-size:1.8rem;font-weight:900;margin-bottom:.2rem;}
.conf-track{height:10px;border-radius:999px;background:#252525;margin-top:.5rem;}
.conf-fill{height:10px;border-radius:999px;}
textarea,div[data-baseweb="textarea"] textarea{
    color:#fff!important;background-color:#1f1f1f!important;
    border:1px solid #333!important;}
textarea::placeholder{color:#888!important;}
input[type="number"],div[data-baseweb="input"] input{
    color:#fff!important;background-color:#1f1f1f!important;}
</style>
""", unsafe_allow_html=True)

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:.12em;margin-bottom:.3rem;'>🏆 CONTENT SUCCESS ANALYTICS ENGINE</div>"
    "<div style='color:#a0a0a0;font-size:.93rem;margin-bottom:1.4rem;'>"
    "End-to-end ML pipeline · Feature engineering · Model comparison · "
    "Automated business insights · Live prediction</div>",
    unsafe_allow_html=True,
)

# ── load ───────────────────────────────────────────────────────────────────────
@st.cache_resource
def load():
    mp = ROOT / "success_models.pkl"
    dp = ROOT / "success_data.pkl"
    if not mp.exists() or not dp.exists():
        return None, None
    return joblib.load(mp), joblib.load(dp)

models, payload = load()

if models is None:
    st.warning(
        "Run `python task6_success_analytics.py` from the project root first.",
        icon="⚠️",
    )
    st.stop()

metrics:    dict       = payload["metrics"]
best_name:  str        = payload["best_model"]
fi_dict:    dict       = payload["feature_importance"]
insights:   dict       = payload["insights"]
eng_df:     pd.DataFrame = payload["engineered_df"]

# ── helpers ────────────────────────────────────────────────────────────────────
PALETTE = ["#E50914","#3b82f6","#22c55e","#f59e0b",
           "#a855f7","#ec4899","#14b8a6","#f97316"]

def dark_layout(height=360, title="", **kw) -> dict:
    return dict(
        paper_bgcolor="#0c0c0c", plot_bgcolor="#111111",
        font=dict(color="#c0c0c0", size=12),
        margin=dict(l=20, r=20, t=40 if title else 20, b=30),
        xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
        yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
        legend=dict(bgcolor="#161616", bordercolor="#2a2a2a",
                    borderwidth=1, font=dict(size=11)),
        height=height,
        title=dict(text=title, font=dict(color="#f5f5f5", size=13)) if title else {},
        **kw,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Step 1 – KPIs & Feature Engineering Overview
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Step 1 · Dataset & Feature Engineering</div>",
            unsafe_allow_html=True)

kpis = insights["kpis"]
kpi_html = (
    "<div class='kpi-grid'>"
    f"<div class='kpi-card'><div class='kpi-val'>{kpis['total_titles']:,}</div>"
    f"<div class='kpi-lbl'>Total Titles</div>"
    f"<div class='kpi-sub'>{kpis['year_range']}</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{kpis['high_engagement']:,}</div>"
    f"<div class='kpi-lbl'>High Engagement</div>"
    f"<div class='kpi-sub'>{kpis['success_rate']}% of catalogue</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{kpis['total_movies']:,}</div>"
    f"<div class='kpi-lbl'>Movies</div>"
    f"<div class='kpi-sub'>{kpis['total_tv']:,} TV Shows</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{kpis['unique_countries']}</div>"
    f"<div class='kpi-lbl'>Countries</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{kpis['unique_genres']}</div>"
    f"<div class='kpi-lbl'>Unique Genres</div></div>"
    "</div>"
)
st.markdown(kpi_html, unsafe_allow_html=True)

# feature overview table
feat_overview = pd.DataFrame([
    {"Feature Group": "Numeric",     "Features": "duration, release_year, desc_length, desc_word_count, genre_count, cast_count, title_length …", "Count": 13},
    {"Feature Group": "Categorical", "Features": "rating (OHE), country_primary (OHE), type (OHE)",  "Count": "~40"},
    {"Feature Group": "Text TF-IDF", "Features": "description (top 80 terms)",                        "Count": 80},
    {"Feature Group": "Genre TF-IDF","Features": "listed_in (binary, top 40 genres)",                 "Count": 40},
    {"Feature Group": "Engineered",  "Features": "is_recent, is_movie, month_added, year_added, has_known_director", "Count": 5},
])
st.dataframe(feat_overview.set_index("Feature Group"), use_container_width=True)

# success label distribution donut
type_split = insights["type_split"]
labels_eng = ["Low Engagement", "High Engagement"]
vals_eng   = [int(eng_df["success"].eq(0).sum()), int(eng_df["success"].eq(1).sum())]

fig_donut = go.Figure(go.Pie(
    labels=labels_eng, values=vals_eng, hole=0.6,
    marker_colors=["#2a2a2a", "#E50914"],
    textinfo="percent+label",
    textfont=dict(color="#f5f5f5", size=12),
))
fig_donut.update_layout(**dark_layout(260, "Engagement Label Distribution"))
st.plotly_chart(fig_donut, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 – Model Performance Comparison
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Step 2 · Model Performance Comparison</div>",
            unsafe_allow_html=True)

cards_html = "<div class='model-grid'>"
for name, m in metrics.items():
    is_best  = name == best_name
    cls      = "model-card best" if is_best else "model-card"
    tag      = " <span style='color:#E50914;font-size:.73rem;font-weight:700;'>★ BEST</span>" if is_best else ""
    cards_html += (
        f"<div class='{cls}'>"
        f"<div class='mc-name'>{name}{tag}</div>"
        f"<div class='mc-val'>{m['roc_auc']*100:.1f}%</div>"
        f"<div class='mc-lbl'>ROC-AUC</div>"
        f"<div class='mc-sub'>"
        f"Acc <strong style='color:#f5f5f5'>{m['accuracy']*100:.1f}%</strong> &nbsp;"
        f"F1 <strong style='color:#f5f5f5'>{m['f1']*100:.1f}%</strong> &nbsp;"
        f"Prec <strong style='color:#f5f5f5'>{m['precision']*100:.1f}%</strong>"
        f"</div></div>"
    )
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# radar chart comparing models
categories = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
fig_radar = go.Figure()
for i, (name, m) in enumerate(metrics.items()):
    vals = [m["accuracy"], m["f1"], m["precision"], m["recall"], m["roc_auc"]]
    vals += vals[:1]
    c = PALETTE[i % len(PALETTE)]
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    fig_radar.add_trace(go.Scatterpolar(
        r=vals,
        theta=categories + [categories[0]],
        fill="toself",
        name=name,
        line=dict(color=c, width=2),
        fillcolor=f"rgba({r},{g},{b}, 0.10)",
    ))

fig_radar.update_layout(
    **dark_layout(380, "Model Comparison – All Metrics"),
    polar=dict(
        bgcolor="#111111",
        radialaxis=dict(visible=True, range=[0.85, 1.0],
                        gridcolor="#2a2a2a", tickcolor="#a0a0a0",
                        tickfont=dict(color="#a0a0a0", size=10)),
        angularaxis=dict(gridcolor="#2a2a2a", tickcolor="#a0a0a0",
                         tickfont=dict(color="#c0c0c0")),
    ),
)
st.plotly_chart(fig_radar, use_container_width=True)

# metric bar chart
met_long = []
for name, m in metrics.items():
    for k, v in m.items():
        met_long.append({"Model": name, "Metric": k.upper(), "Score": v})
met_df_long = pd.DataFrame(met_long)

fig_bars = px.bar(
    met_df_long, x="Metric", y="Score", color="Model",
    barmode="group", color_discrete_sequence=PALETTE,
    range_y=[0.85, 1.01],
)
fig_bars.update_layout(**dark_layout(340, "Metric-by-Metric Comparison"))
st.plotly_chart(fig_bars, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 – Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Step 3 · Feature Importance</div>",
            unsafe_allow_html=True)

fi_models = [n for n in ["Gradient Boosting", "Random Forest", "Extra Trees"]
             if n in fi_dict]
if fi_models:
    fi_tabs = st.tabs(fi_models)
    for tab, name in zip(fi_tabs, fi_models):
        with tab:
            fi_df = fi_dict[name].head(15)
            fig_fi = px.bar(
                fi_df, x="importance", y="feature",
                orientation="h", color="importance",
                color_continuous_scale=["#2a2a2a", "#E50914"],
            )
            fig_fi.update_layout(
                **dark_layout(400, f"Top 15 Features – {name}"),
                yaxis_autorange="reversed",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 – Automated Business Insights
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Step 4 · Automated Business Insights</div>",
            unsafe_allow_html=True)

ins_tabs = st.tabs(["🎭 Genre", "🌍 Country", "📅 Annual Trend",
                    "📊 Rating", "🏅 Top Titles"])

# ── Genre success ──────────────────────────────────────────────────────────────
with ins_tabs[0]:
    gs = insights["genre_success"]
    fig_g = px.bar(
        gs, x="success_rate", y="genre", orientation="h",
        color="success_rate", color_continuous_scale=["#1a1a2a", "#E50914"],
        hover_data={"count": True, "success_rate": ":.1%"},
        labels={"success_rate": "Engagement Rate", "genre": "Genre", "count": "Titles"},
    )
    fig_g.update_layout(
        **dark_layout(420, "Engagement Rate by Genre"),
        yaxis_autorange="reversed",
        coloraxis_showscale=False,
        xaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_g, use_container_width=True)

    # bubble chart: count vs success_rate
    fig_bubble = px.scatter(
        gs, x="count", y="success_rate", size="count",
        text="genre", color="success_rate",
        color_continuous_scale=["#1a1a2a","#E50914"],
        labels={"count":"Titles","success_rate":"Engagement Rate"},
    )
    fig_bubble.update_traces(textposition="top center",
                             textfont=dict(color="#c0c0c0", size=10))
    fig_bubble.update_layout(
        **dark_layout(380, "Genre Volume vs Engagement Rate"),
        coloraxis_showscale=False,
    )
    fig_bubble.update_xaxes(gridcolor="#1e1e1e")
    fig_bubble.update_yaxes(gridcolor="#1e1e1e", tickformat=".0%")
    st.plotly_chart(fig_bubble, use_container_width=True)

# ── Country success ────────────────────────────────────────────────────────────
with ins_tabs[1]:
    cs = insights["country_success"]
    fig_c = px.bar(
        cs, x="country", y="success_rate",
        color="success_rate", color_continuous_scale=["#1a1a2a","#3b82f6"],
        hover_data={"count": True},
        labels={"success_rate": "Engagement Rate", "country": "Country"},
    )
    fig_c.update_layout(
        **dark_layout(360, "Engagement Rate by Country"),
        coloraxis_showscale=False,
    )
    fig_c.update_xaxes(tickangle=-30, gridcolor="#1e1e1e")
    fig_c.update_yaxes(tickformat=".0%", gridcolor="#1e1e1e")
    st.plotly_chart(fig_c, use_container_width=True)

    # treemap of title counts
    fig_tree = px.treemap(
        cs, path=["country"], values="count",
        color="success_rate",
        color_continuous_scale=["#1a1a2a", "#3b82f6"],
        hover_data={"success_rate": ":.1%"},
    )
    fig_tree.update_layout(
        paper_bgcolor="#0c0c0c", margin=dict(l=10,r=10,t=40,b=10),
        height=320,
        title=dict(text="Content Volume by Country",
                   font=dict(color="#f5f5f5", size=13)),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# ── Annual trend ───────────────────────────────────────────────────────────────
with ins_tabs[2]:
    at = insights["annual_trend"]
    fig_at = go.Figure()
    fig_at.add_trace(go.Bar(
        x=at["release_year"], y=at["total"],
        name="Total Titles", marker_color="#2a2a2a",
    ))
    fig_at.add_trace(go.Bar(
        x=at["release_year"], y=at["success_count"],
        name="High Engagement", marker_color="#E50914",
    ))
    fig_at.add_trace(go.Scatter(
        x=at["release_year"], y=at["success_rate"],
        name="Engagement Rate", yaxis="y2",
        line=dict(color="#f59e0b", width=2.5),
        mode="lines+markers", marker=dict(size=5),
    ))
    fig_at.update_layout(
        **dark_layout(380, "Annual Content Production & Engagement"),
        barmode="overlay",
        yaxis2=dict(
            overlaying="y", side="right",
            tickformat=".0%", gridcolor="rgba(0,0,0,0)",
            title="Engagement Rate", titlefont=dict(color="#f59e0b"),
            tickfont=dict(color="#f59e0b"),
        ),
    )
    fig_at.update_xaxes(gridcolor="#1e1e1e")
    fig_at.update_yaxes(title_text="Title Count", gridcolor="#1e1e1e")
    st.plotly_chart(fig_at, use_container_width=True)

    # monthly heatmap
    mt = insights["monthly_trend"].copy()
    mt["month"] = mt["ym_ts"].dt.month
    mt["year"]  = mt["ym_ts"].dt.year
    pivot = mt.pivot_table(index="year", columns="month",
                           values="total", fill_value=0)
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_heat = px.imshow(
        pivot, labels=dict(x="Month", y="Year", color="Additions"),
        x=[month_names[i-1] for i in pivot.columns],
        color_continuous_scale=["#111111","#E50914"],
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="#0c0c0c", plot_bgcolor="#111111",
        font=dict(color="#c0c0c0", size=11),
        height=240, margin=dict(l=20,r=20,t=40,b=20),
        title=dict(text="Monthly Additions Heatmap",
                   font=dict(color="#f5f5f5", size=13)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ── Rating breakdown ───────────────────────────────────────────────────────────
with ins_tabs[3]:
    rb = insights["rating_breakdown"]
    fig_rb = px.bar(
        rb, x="rating", y="success_rate",
        color="success_rate", color_continuous_scale=["#1a1a2a","#22c55e"],
        hover_data={"total": True},
        labels={"success_rate": "Engagement Rate", "rating": "Rating", "total": "Titles"},
    )
    fig_rb.update_layout(
        **dark_layout(320, "Engagement Rate by Content Rating"),
        coloraxis_showscale=False,
    )
    fig_rb.update_xaxes(gridcolor="#1e1e1e")
    fig_rb.update_yaxes(tickformat=".0%", gridcolor="#1e1e1e")
    st.plotly_chart(fig_rb, use_container_width=True)

    # pie by total count
    fig_pie = px.pie(
        rb, names="rating", values="total",
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    fig_pie.update_layout(
        paper_bgcolor="#0c0c0c",
        font=dict(color="#c0c0c0", size=11),
        height=300, margin=dict(l=0,r=0,t=30,b=0),
        title=dict(text="Titles by Rating",
                   font=dict(color="#f5f5f5", size=13)),
        legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", borderwidth=1),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ── Top titles ─────────────────────────────────────────────────────────────────
with ins_tabs[4]:
    tt = insights["top_titles"][["title","type","rating","release_year","listed_in"]]
    tt.columns = ["Title","Type","Rating","Year","Genres"]
    st.dataframe(tt, use_container_width=True, hide_index=True)

    # top genres by count bar
    genre_counts = (
        eng_df["listed_in"].str.split(",").explode().str.strip()
        .value_counts().head(12).reset_index()
    )
    genre_counts.columns = ["Genre", "Count"]
    fig_gc = px.bar(
        genre_counts, x="Count", y="Genre", orientation="h",
        color="Count", color_continuous_scale=["#1a1a2a","#f59e0b"],
    )
    fig_gc.update_layout(
        **dark_layout(360, "Top 12 Genres by Title Count"),
        yaxis_autorange="reversed",
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_gc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 – Live Prediction Tool
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>Step 5 · Live Engagement Prediction</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div style='color:#a0a0a0;font-size:.9rem;margin-bottom:1rem;'>"
    f"Using <strong style='color:#f5f5f5;'>{best_name}</strong> "
    f"(ROC-AUC {metrics[best_name]['roc_auc']*100:.1f}%) "
    "to predict whether new content will have high audience engagement.</div>",
    unsafe_allow_html=True,
)

# derive option lists from training data
rating_opts  = sorted(eng_df["rating"][~eng_df["rating"].isin(
    {"Unknown","NR","UR","74 min","84 min","66 min"})].unique().tolist())
country_opts = sorted(eng_df["country_primary"][
    eng_df["country_primary"] != "Unknown"].unique().tolist())
genre_tokens = sorted({
    g.strip() for gs in eng_df["listed_in"].dropna()
    for g in gs.split(",")
})

best_model = models[best_name]

with st.form("pred_form"):
    p1, p2 = st.columns(2)
    with p1:
        p_type   = st.selectbox("Content Type", ["Movie", "TV Show"])
        p_year   = st.number_input("Release Year", 1950, 2026, 2022, step=1)
        pd1, pd2 = st.columns([2, 1])
        with pd1:
            p_dur  = st.number_input("Duration", 1, 500, 90, step=1)
        with pd2:
            p_unit = st.selectbox("Unit", ["Minutes", "Seasons"])
        p_rating = st.selectbox(
            "Content Rating", rating_opts,
            index=rating_opts.index("TV-MA") if "TV-MA" in rating_opts else 0,
        )
        p_country = st.selectbox(
            "Country", country_opts,
            index=country_opts.index("United States") if "United States" in country_opts else 0,
        )
    with p2:
        p_genres = st.multiselect(
            "Genres", genre_tokens,
            default=["Dramas", "International Movies"],
        )
        p_director = st.text_input("Director (optional)", placeholder="e.g. Christopher Nolan")
        p_cast_count = st.slider("Approx. Cast Size", 1, 30, 8)
        p_desc = st.text_area(
            "Plot Description",
            height=150,
            placeholder="e.g. A detective uncovers a conspiracy stretching across continents…",
        )

    pred_btn = st.form_submit_button(
        "🔮 Predict Engagement", type="primary", use_container_width=True
    )

if pred_btn:
    listed_str = ", ".join(sorted(p_genres)) if p_genres else "Unknown"

    NUM_COLS_LOCAL = [
        "duration_num","release_year","desc_length","desc_word_count",
        "is_movie","is_recent","month_added","year_added",
        "genre_count","has_known_director","cast_count",
        "title_length","title_word_count",
    ]

    desc_text = p_desc.strip() if p_desc else ""
    row = {
        "type":             p_type,
        "release_year":     float(p_year),
        "duration_num":     float(p_dur),
        "rating":           p_rating,
        "listed_in":        listed_str,
        "country_primary":  p_country,
        "description":      desc_text,
        "desc_length":      float(len(desc_text)),
        "desc_word_count":  float(len(desc_text.split())),
        "is_movie":         1.0 if p_type == "Movie" else 0.0,
        "is_recent":        1.0 if p_year >= 2018 else 0.0,
        "month_added":      0.0,
        "year_added":       0.0,
        "genre_count":      float(len(p_genres)) if p_genres else 1.0,
        "has_known_director": 1.0 if p_director.strip() else 0.0,
        "cast_count":       float(p_cast_count),
        "title_length":     0.0,
        "title_word_count": 0.0,
    }
    X_new = pd.DataFrame([row])

    with st.spinner("Running prediction…"):
        pred   = int(best_model.predict(X_new)[0])
        probas = best_model.predict_proba(X_new)[0]

    label      = "High Engagement" if pred == 1 else "Low Engagement"
    color      = "#22c55e" if pred == 1 else "#E50914"
    icon       = "🚀" if pred == 1 else "📉"
    conf       = float(probas[pred]) * 100
    low_p      = probas[0] * 100
    high_p     = probas[1] * 100

    st.markdown(
        f"""
        <div class='pred-box'>
            <div class='pred-title' style='color:{color};'>{icon} {label}</div>
            <div style='color:#a0a0a0;font-size:.88rem;margin-bottom:.7rem;'>
                Predicted by <strong style='color:#f5f5f5;'>{best_name}</strong>
            </div>
            <div style='color:#a0a0a0;font-size:.83rem;'>
                Confidence: <strong style='color:#f5f5f5;'>{conf:.1f}%</strong>
            </div>
            <div class='conf-track'>
                <div class='conf-fill' style='width:{conf:.1f}%;background:{color};'></div>
            </div>
            <div style='display:flex;gap:2rem;margin-top:.9rem;'>
                <div><div style='color:#E50914;font-size:1.2rem;font-weight:900;'>{low_p:.1f}%</div>
                     <div style='color:#a0a0a0;font-size:.78rem;'>Low Engagement</div></div>
                <div><div style='color:#22c55e;font-size:1.2rem;font-weight:900;'>{high_p:.1f}%</div>
                     <div style='color:#a0a0a0;font-size:.78rem;'>High Engagement</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # mini gauge chart
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=high_p,
        number=dict(suffix="%", font=dict(color="#22c55e", size=28)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#a0a0a0",
                      tickfont=dict(color="#a0a0a0")),
            bar=dict(color="#22c55e"),
            bgcolor="#1e1e1e",
            borderwidth=1, bordercolor="#2a2a2a",
            steps=[
                dict(range=[0,   40], color="#1a1a1a"),
                dict(range=[40,  70], color="#1e1e1e"),
                dict(range=[70, 100], color="#1e3a1e"),
            ],
            threshold=dict(
                line=dict(color="#f5f5f5", width=2),
                thickness=0.75, value=50,
            ),
        ),
        title=dict(text="High Engagement Probability",
                   font=dict(color="#a0a0a0", size=13)),
    ))
    fig_gauge.update_layout(
        paper_bgcolor="#0c0c0c",
        font=dict(color="#c0c0c0"),
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
