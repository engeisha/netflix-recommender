"""
Page 5 – Netflix Trend Forecasting
Four sections:
  1. Historical Additions Overview
  2. Future Trend Forecasts (slider 6-24 months)
  3. Genre & Category Growth
  4. Model Evaluation Metrics
"""
from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui_theme import inject_global_css

ROOT = Path(__file__).resolve().parent.parent

st.set_page_config(
    page_title="Trend Forecasting · Netflix AI",
    page_icon="📈",
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

    .sec-head {
        color: var(--text); font-size: 1.05rem; font-weight: 700;
        margin: 1.6rem 0 0.8rem; padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }
    .kpi-grid { display:flex; gap:0.9rem; flex-wrap:wrap; margin-bottom:1.2rem; }
    .kpi-card {
        flex:1; min-width:130px; background:#141414;
        border:1px solid var(--border); border-radius:14px; padding:0.85rem 1rem;
    }
    .kpi-val { color:var(--accent); font-size:1.55rem; font-weight:900; }
    .kpi-lbl { color:var(--muted); font-size:0.79rem; margin-top:0.1rem; }
    .kpi-sub { color:#c0c0c0; font-size:0.78rem; margin-top:0.2rem; }

    .metric-grid { display:flex; gap:0.9rem; flex-wrap:wrap; margin-top:0.6rem; }
    .metric-card {
        flex:1; min-width:160px; background:#141414;
        border:1px solid var(--border); border-radius:14px; padding:0.9rem 1.1rem;
    }
    .metric-model { color:var(--muted); font-size:0.78rem; font-weight:700;
                    text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.3rem; }
    .metric-val   { color:var(--accent); font-size:1.4rem; font-weight:900; }
    .metric-lbl   { color:var(--muted); font-size:0.78rem; }

    .growth-bar-row { display:flex; align-items:center; gap:0.6rem; margin-bottom:0.5rem; }
    .growth-label { color:#c0c0c0; font-size:0.86rem; width:210px; flex-shrink:0; }
    .growth-track { flex:1; height:9px; border-radius:999px; background:#252525; }
    .growth-fill  { height:9px; border-radius:999px; }
    .growth-pct   { font-size:0.82rem; font-weight:700; width:60px; text-align:right; flex-shrink:0; }

    textarea, div[data-baseweb="textarea"] textarea {
        color:#FFFFFF !important; background-color:#1F1F1F !important;
        border:1px solid #333333 !important;
    }
    textarea::placeholder { color:#888888 !important; }
    input[type="number"], div[data-baseweb="input"] input {
        color:#FFFFFF !important; background-color:#1F1F1F !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#E50914;font-size:1.1rem;font-weight:800;"
    "letter-spacing:0.12em;margin-bottom:0.3rem;'>📈 TREND FORECASTING</div>"
    "<div style='color:#a0a0a0;font-size:0.93rem;margin-bottom:1.4rem;'>"
    "Monthly content additions history, future forecasts, and genre growth trends.</div>",
    unsafe_allow_html=True,
)

# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    m = ROOT / "forecasting_model.pkl"
    d = ROOT / "forecasting_data.pkl"
    if not m.exists() or not d.exists():
        return None, None
    return joblib.load(m), joblib.load(d)

models, data = load_artifacts()

if models is None:
    st.warning(
        "Forecasting artifacts not found. Run `python task5_trend_forecaster.py` "
        "from the project root first.",
        icon="⚠️",
    )
    st.stop()

history:      dict = data["history"]
forecasts:    dict = data["forecasts"]
metrics:      dict = data["metrics"]
genre_growth: dict = data["genre_growth"]
TOP_GENRES:   list = data["top_genres"]

# ── helpers ────────────────────────────────────────────────────────────────────
SERIES_COLORS = {
    "Overall":  "#E50914",
    "Movie":    "#3b82f6",
    "TV Show":  "#22c55e",
}
GENRE_COLORS = [
    "#f59e0b", "#a855f7", "#ec4899", "#14b8a6",
    "#f97316", "#06b6d4", "#84cc16", "#e879f9",
]

def plot_bgcolor() -> dict:
    return dict(
        paper_bgcolor="#0c0c0c",
        plot_bgcolor="#111111",
        font=dict(color="#c0c0c0", size=12),
        margin=dict(l=20, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
        yaxis=dict(gridcolor="#1e1e1e", zerolinecolor="#2a2a2a"),
        legend=dict(bgcolor="#161616", bordercolor="#2a2a2a", borderwidth=1,
                    font=dict(size=11)),
        hovermode="x unified",
    )

def series_to_ts(name: str) -> pd.Series:
    """Return full series as a DatetimeIndex pd.Series."""
    s = history[name]["full_series"].copy()
    if hasattr(s.index, "to_timestamp"):
        s.index = s.index.to_timestamp()
    return s

# ══════════════════════════════════════════════════════════════════════════════
# Section 1 – Historical Additions Overview
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>① Historical Additions Overview</div>",
            unsafe_allow_html=True)

# KPI strip
overall_s = series_to_ts("Overall")
movie_s   = series_to_ts("Movie")
tv_s      = series_to_ts("TV Show")

kpi_html = (
    "<div class='kpi-grid'>"
    f"<div class='kpi-card'><div class='kpi-val'>{int(overall_s.sum()):,}</div>"
    f"<div class='kpi-lbl'>Total Titles (2016–2021)</div>"
    f"<div class='kpi-sub'>Peak: {int(overall_s.max())} / month</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{int(movie_s.sum()):,}</div>"
    f"<div class='kpi-lbl'>Movies Added</div>"
    f"<div class='kpi-sub'>{movie_s.sum()/overall_s.sum()*100:.0f}% of total</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{int(tv_s.sum()):,}</div>"
    f"<div class='kpi-lbl'>TV Shows Added</div>"
    f"<div class='kpi-sub'>{tv_s.sum()/overall_s.sum()*100:.0f}% of total</div></div>"

    f"<div class='kpi-card'><div class='kpi-val'>{int(overall_s.mean())}</div>"
    f"<div class='kpi-lbl'>Avg / Month</div>"
    f"<div class='kpi-sub'>Median: {int(overall_s.median())}</div></div>"
    "</div>"
)
st.markdown(kpi_html, unsafe_allow_html=True)

# filter toggles
h_col1, h_col2 = st.columns([3, 1])
with h_col2:
    show_overall = st.checkbox("Overall",  value=True)
    show_movies  = st.checkbox("Movies",   value=True)
    show_tv      = st.checkbox("TV Shows", value=True)
    show_3m_avg  = st.checkbox("3-month avg", value=True)

fig_hist = go.Figure()

series_map = [
    ("Overall", overall_s, "#E50914", show_overall),
    ("Movie",   movie_s,   "#3b82f6", show_movies),
    ("TV Show", tv_s,      "#22c55e", show_tv),
]
for name, s, color, visible in series_map:
    if not visible:
        continue
    fig_hist.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines",
        name=name, line=dict(color=color, width=2),
        hovertemplate=f"<b>{name}</b>: %{{y}}<extra></extra>",
    ))
    if show_3m_avg:
        roll = s.rolling(3, center=True).mean()
        fig_hist.add_trace(go.Scatter(
            x=roll.index, y=roll.values, mode="lines",
            name=f"{name} 3m avg",
            line=dict(color=color, width=1.5, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))

fig_hist.update_layout(
    **plot_bgcolor(), height=380,
    title=dict(text="Monthly Netflix Additions (2016–2021)",
               font=dict(color="#f5f5f5", size=14)),
    xaxis_title="Date", yaxis_title="Titles Added",
)
st.plotly_chart(fig_hist, use_container_width=True)

# year-over-year bar chart
yoy_data = pd.DataFrame({
    "Overall": overall_s, "Movie": movie_s, "TV Show": tv_s,
})
yoy_data["year"] = yoy_data.index.year
yoy_annual = yoy_data.groupby("year")[["Overall", "Movie", "TV Show"]].sum().reset_index()
yoy_annual = yoy_annual[yoy_annual["year"] >= 2016]

fig_yoy = go.Figure()
for col, color in [("Movie", "#3b82f6"), ("TV Show", "#22c55e")]:
    fig_yoy.add_trace(go.Bar(
        x=yoy_annual["year"], y=yoy_annual[col],
        name=col, marker_color=color,
    ))
fig_yoy.update_layout(
    **plot_bgcolor(), height=280, barmode="stack",
    title=dict(text="Annual Additions by Type",
               font=dict(color="#f5f5f5", size=13)),
    xaxis_title="Year", yaxis_title="Count",
)
st.plotly_chart(fig_yoy, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 2 – Future Trend Forecasts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>② Future Trend Forecasts</div>",
            unsafe_allow_html=True)

fc_col1, fc_col2 = st.columns([3, 1])
with fc_col2:
    horizon = st.select_slider(
        "Forecast horizon (months)",
        options=[6, 12, 18, 24],
        value=12,
    )
    fc_series = st.selectbox(
        "Series", options=[s for s in ["Overall", "Movie", "TV Show"] if s in forecasts],
        index=0,
    )

with fc_col1:
    hist_s    = series_to_ts(fc_series)
    fc_df     = forecasts[fc_series].iloc[:horizon]

    # build a simple ±1.5 std confidence band from test residuals
    te        = history[fc_series]["test"]
    resid_std = float(np.std(te["actual"] - te["predicted"])) if te is not None else 10.0
    fc_upper  = fc_df["forecast"] + 1.5 * resid_std
    fc_lower  = (fc_df["forecast"] - 1.5 * resid_std).clip(lower=0)

    fig_fc = go.Figure()

    # historical
    fig_fc.add_trace(go.Scatter(
        x=hist_s.index, y=hist_s.values,
        mode="lines", name="Historical",
        line=dict(color="#E50914", width=2),
    ))
    # test fitted
    if te is not None:
        fig_fc.add_trace(go.Scatter(
            x=te.index, y=te["predicted"],
            mode="lines", name="Model fit (test)",
            line=dict(color="#f59e0b", width=1.5, dash="dot"),
        ))
    # confidence band
    fig_fc.add_trace(go.Scatter(
        x=list(fc_df.index) + list(fc_df.index[::-1]),
        y=list(fc_upper) + list(fc_lower[::-1]),
        fill="toself",
        fillcolor="rgba(229,9,20,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        name="±1.5σ band",
        hoverinfo="skip",
    ))
    # forecast line
    fig_fc.add_trace(go.Scatter(
        x=fc_df.index, y=fc_df["forecast"],
        mode="lines+markers", name="Forecast",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=5, color="#f97316"),
        hovertemplate="<b>Forecast</b>: %{y:.0f}<extra></extra>",
    ))
    # vertical divider
    last_hist = hist_s.index[-1]
    fig_fc.add_shape(
        type="line",
        x0=str(last_hist.date()), x1=str(last_hist.date()),
        y0=0, y1=1, xref="x", yref="paper",
        line=dict(color="#444444", width=1.5, dash="dash"),
    )
    fig_fc.add_annotation(
        x=str(last_hist.date()), y=1, xref="x", yref="paper",
        text="Forecast start", showarrow=False,
        font=dict(color="#a0a0a0", size=11),
        xanchor="left", yanchor="bottom",
    )

    fig_fc.update_layout(
        **plot_bgcolor(), height=400,
        title=dict(
            text=f"{fc_series} — {horizon}-month Forecast",
            font=dict(color="#f5f5f5", size=14),
        ),
        xaxis_title="Date", yaxis_title="Titles / Month",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

# forecast table
fc_display = fc_df.copy()
fc_display.index = fc_display.index.strftime("%b %Y")
fc_display["forecast"] = fc_display["forecast"].round(0).astype(int)
fc_display.columns = ["Predicted Additions"]

st.markdown(
    "<div style='color:#a0a0a0;font-size:0.82rem;margin-bottom:0.4rem;'>"
    "Monthly forecast breakdown</div>",
    unsafe_allow_html=True,
)
st.dataframe(
    fc_display.T,
    use_container_width=True,
)

# ══════════════════════════════════════════════════════════════════════════════
# Section 3 – Genre & Category Growth
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>③ Genre & Category Growth</div>",
            unsafe_allow_html=True)

g_col1, g_col2 = st.columns([3, 2])

with g_col1:
    # genre historical lines
    fig_genre = go.Figure()
    for i, genre in enumerate(TOP_GENRES):
        if genre not in history:
            continue
        s = series_to_ts(genre)
        color = GENRE_COLORS[i % len(GENRE_COLORS)]
        fig_genre.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines",
            name=genre, line=dict(color=color, width=1.8),
            hovertemplate=f"<b>{genre}</b>: %{{y}}<extra></extra>",
        ))
    fig_genre.update_layout(
        **plot_bgcolor(), height=360,
        title=dict(text="Monthly Additions by Genre",
                   font=dict(color="#f5f5f5", size=13)),
        xaxis_title="Date", yaxis_title="Titles / Month",
    )
    st.plotly_chart(fig_genre, use_container_width=True)

with g_col2:
    st.markdown(
        "<div style='color:#a0a0a0;font-size:0.82rem;font-weight:700;"
        "text-transform:uppercase;letter-spacing:0.1em;"
        "margin-bottom:0.65rem;'>Growth vs Prior Period</div>",
        unsafe_allow_html=True,
    )
    sorted_growth = sorted(genre_growth.items(), key=lambda x: -x[1])
    max_pct = max(abs(v) for _, v in sorted_growth) if sorted_growth else 1

    bar_rows = ""
    for genre, pct in sorted_growth:
        fill_w   = min(abs(pct) / max_pct * 100, 100)
        clr      = "#22c55e" if pct >= 0 else "#E50914"
        sign     = "+" if pct >= 0 else ""
        bar_rows += (
            f"<div class='growth-bar-row'>"
            f"<div class='growth-label'>{genre}</div>"
            f"<div class='growth-track'>"
            f"<div class='growth-fill' style='width:{fill_w:.1f}%;background:{clr};'></div>"
            f"</div>"
            f"<div class='growth-pct' style='color:{clr};'>{sign}{pct:.0f}%</div>"
            f"</div>"
        )
    st.markdown(bar_rows, unsafe_allow_html=True)

# genre forecast chart (12-month ahead)
st.markdown(
    "<div style='color:#f5f5f5;font-size:0.95rem;font-weight:700;"
    "margin:1.2rem 0 0.7rem;'>12-Month Genre Forecasts</div>",
    unsafe_allow_html=True,
)
fig_gfc = go.Figure()
for i, genre in enumerate(TOP_GENRES):
    if genre not in forecasts:
        continue
    fc = forecasts[genre].iloc[:12]
    color = GENRE_COLORS[i % len(GENRE_COLORS)]
    fig_gfc.add_trace(go.Bar(
        x=fc.index.strftime("%b %Y"),
        y=fc["forecast"].round(0),
        name=genre, marker_color=color,
        hovertemplate=f"<b>{genre}</b>: %{{y:.0f}}<extra></extra>",
    ))
fig_gfc.update_layout(
    **plot_bgcolor(), height=340, barmode="group",
    title=dict(text="Forecasted Monthly Additions by Genre (next 12 months)",
               font=dict(color="#f5f5f5", size=13)),
    xaxis_title="Month", yaxis_title="Predicted Titles",
    xaxis_tickangle=-30,
)
st.plotly_chart(fig_gfc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Section 4 – Model Evaluation Metrics
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-head'>④ Model Evaluation Metrics</div>",
            unsafe_allow_html=True)
st.markdown(
    "<div style='color:#a0a0a0;font-size:0.88rem;margin-bottom:1rem;'>"
    "Evaluated on a 12-month held-out test split. "
    "Model: Gradient Boosting Regressor with lag + rolling features.</div>",
    unsafe_allow_html=True,
)

# primary series metrics cards
primary = ["Overall", "Movie", "TV Show"]
cards_html = "<div class='metric-grid'>"
for name in primary:
    if name not in metrics:
        continue
    m = metrics[name]
    color = SERIES_COLORS.get(name, "#E50914")
    cards_html += (
        f"<div class='metric-card' style='border-color:{color}33;'>"
        f"<div class='metric-model' style='color:{color};'>{name}</div>"
        f"<div class='metric-val'>{m['mae']:.1f}</div>"
        f"<div class='metric-lbl'>MAE (titles/month)</div>"
        f"<div style='margin-top:0.45rem;display:flex;gap:1rem;'>"
        f"<div><div style='color:#f5f5f5;font-size:0.95rem;font-weight:700;'>{m['rmse']:.1f}</div>"
        f"<div style='color:#a0a0a0;font-size:0.75rem;'>RMSE</div></div>"
        f"<div><div style='color:#f5f5f5;font-size:0.95rem;font-weight:700;'>{m['mape']:.1f}%</div>"
        f"<div style='color:#a0a0a0;font-size:0.75rem;'>MAPE</div></div>"
        f"</div></div>"
    )
cards_html += "</div>"
st.markdown(cards_html, unsafe_allow_html=True)

# genre metrics table
st.markdown(
    "<div style='color:#f5f5f5;font-size:0.92rem;font-weight:700;"
    "margin:1.4rem 0 0.6rem;'>Genre Model Metrics</div>",
    unsafe_allow_html=True,
)
genre_met_rows = [
    {"Genre": g, "MAE": metrics[g]["mae"], "RMSE": metrics[g]["rmse"], "MAPE (%)": metrics[g]["mape"]}
    for g in TOP_GENRES if g in metrics
]
if genre_met_rows:
    gm_df = pd.DataFrame(genre_met_rows).set_index("Genre")
    st.dataframe(
        gm_df.style.format({"MAE": "{:.1f}", "RMSE": "{:.1f}", "MAPE (%)": "{:.1f}"}),
        use_container_width=True,
    )

# accuracy context note
st.markdown(
    "<div style='color:#a0a0a0;font-size:0.82rem;margin-top:0.8rem;border-left:"
    "3px solid #2a2a2a;padding-left:0.75rem;'>"
    "Note: Netflix data ends Sep 2021, so forecasts extrapolate beyond the observed trend. "
    "Lower MAPE on Overall / Movie / TV Show series reflects denser signals. "
    "Genre-level models are noisier due to sparser monthly counts.</div>",
    unsafe_allow_html=True,
)
