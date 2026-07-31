from pathlib import Path

import pandas as pd
import streamlit as st

from ui_theme import inject_global_css
from recommender import get_recommendations, get_matrices
from graph_viz import build_network_figure

DATASET_PATH = Path(__file__).resolve().parent.parent / "cleaned_netflix_titles.csv"


# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recommender · Netflix AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


# ── extra page-level styles ──────────────────────────────────────────────────
st.markdown(
    """
    <style>
    :root {
        --bg:#0c0c0c; --panel:#1a1a1a; --text:#f5f5f5;
        --muted:#a0a0a0; --accent:#E50914; --border:#2a2a2a;
        --green:#22c55e; --amber:#f59e0b;
    }
    .stApp { background:var(--bg); color:var(--text); }
    .block-container { padding-top:1rem; padding-bottom:2.5rem; }

    /* hero */
    .hero {
        background: linear-gradient(135deg, rgba(229,9,20,.16), rgba(12,12,12,.97));
        border:1px solid var(--border); border-radius:20px;
        padding:1.4rem 1.6rem; margin-bottom:1.2rem;
        box-shadow:0 10px 28px rgba(0,0,0,.35);
    }
    .hero-title  { color:var(--text); font-size:1.9rem; font-weight:700; margin:0 0 .3rem; }
    .hero-sub    { color:#ddd; font-size:.92rem; margin-bottom:.5rem; }
    .hero-desc   { color:var(--muted); font-size:.93rem; line-height:1.55;
                   max-width:680px; margin-bottom:.9rem; }
    .hero-tags   { display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.7rem; }
    .tag {
        display:inline-block; padding:5px 11px;
        border:1px solid var(--border); border-radius:999px;
        background:#161616; color:#cfcfcf; font-size:.82rem;
    }
    .tag strong  { color:#f5f5f5; }

    /* section header */
    .sec-hdr {
        color:var(--text); font-size:1.05rem; font-weight:700;
        margin:.4rem 0 .8rem; letter-spacing:.01em;
    }

    /* glass card */
    .g-card {
        background: linear-gradient(160deg, rgba(255,255,255,.045), rgba(255,255,255,.012));
        border:1px solid var(--border); border-radius:18px;
        padding:1rem 1rem 1.1rem; height:100%;
        transition:transform 180ms ease, border-color 180ms ease;
    }
    .g-card:hover { transform:translateY(-4px); border-color:rgba(229,9,20,.5); }
    .g-card-title { color:var(--text); font-size:.97rem; font-weight:700; margin-bottom:.5rem; line-height:1.3; }
    .g-card-meta  { color:var(--muted); font-size:.82rem; line-height:1.45; margin-bottom:.7rem; }

    /* match badge */
    .badge-wrap   { margin-bottom:.65rem; }
    .match-badge  {
        display:inline-block; padding:5px 13px; border-radius:999px;
        font-size:.88rem; font-weight:800; letter-spacing:.02em;
    }
    .badge-high   { background:rgba(34,197,94,.18); color:#22c55e; border:1px solid rgba(34,197,94,.35); }
    .badge-mid    { background:rgba(245,158,11,.18); color:#f59e0b; border:1px solid rgba(245,158,11,.35); }
    .badge-low    { background:rgba(229,9,20,.18);   color:#E50914; border:1px solid rgba(229,9,20,.35); }
    .serendip-pill{
        display:inline-block; padding:2px 8px; border-radius:999px;
        font-size:.72rem; font-weight:700; letter-spacing:.04em;
        background:rgba(139,92,246,.2); color:#a78bfa;
        border:1px solid rgba(139,92,246,.35); margin-left:6px; vertical-align:middle;
    }

    /* breakdown bar */
    .breakdown { margin:.55rem 0 .75rem; }
    .bd-row    { display:flex; align-items:center; gap:.5rem; margin-bottom:4px; font-size:.78rem; }
    .bd-label  { color:var(--muted); width:36px; flex-shrink:0; }
    .bd-bar-bg { flex:1; height:5px; background:#2a2a2a; border-radius:3px; overflow:hidden; }
    .bd-bar-fill-plot  { height:5px; background:#3b82f6; border-radius:3px; }
    .bd-bar-fill-genre { height:5px; background:#22c55e; border-radius:3px; }
    .bd-bar-fill-cast  { height:5px; background:#f59e0b; border-radius:3px; }
    .bd-val    { color:#888; width:32px; text-align:right; flex-shrink:0; }

    /* meta tags inside card */
    .card-tags  { display:flex; flex-wrap:wrap; gap:4px; margin-top:.55rem; }
    .card-tag   {
        font-size:.74rem; padding:3px 8px; border-radius:6px;
        background:#1f1f1f; border:1px solid #333; color:#bbb;
    }

    /* graph glass container */
    .graph-panel {
        background:linear-gradient(160deg,rgba(255,255,255,.035),rgba(10,10,10,.97));
        border:1px solid var(--border); border-radius:22px;
        padding:0; overflow:hidden; margin-top:.4rem;
        box-shadow:0 12px 32px rgba(0,0,0,.45);
    }
    .graph-legend-hint {
        color:var(--muted); font-size:.78rem; text-align:center;
        padding:.55rem 0 .2rem;
    }

    /* view toggle pill */
    .view-toggle-wrap { display:flex; gap:.5rem; margin-bottom:.9rem; align-items:center; }

    /* sidebar sliders */
    section[data-testid="stSidebar"] .stSlider label { color:#f5f5f5 !important; font-size:.9rem !important; }

    /* footer */
    .footer-banner {
        background:linear-gradient(90deg,rgba(229,9,20,.18),rgba(229,9,20,.04));
        border:1px solid var(--border); border-radius:18px;
        padding:1rem 1.3rem; margin-top:1.2rem; color:var(--text);
    }
    .foot { color:var(--muted); text-align:center; padding-top:.9rem; font-size:.84rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── data loader ───────────────────────────────────────────────────────────────
@st.cache_data
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


df = load_dataset()
titles = sorted(df["title"].dropna().astype(str).unique().tolist())

if not titles:
    st.error("No titles available.")
    st.stop()


# ── sidebar controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='color:#f5f5f5;font-size:1rem;font-weight:700;margin-bottom:.9rem;'>⚙️ Engine Controls</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='color:#a0a0a0;font-size:.82rem;margin-bottom:.4rem;'>Similarity Weights (must sum to 1.0)</div>",
        unsafe_allow_html=True,
    )

    w_plot  = st.slider("🖊 Plot Summary",     0.0, 1.0, 0.50, 0.05, key="w_plot")
    w_genre = st.slider("🎭 Genre Similarity", 0.0, 1.0, 0.30, 0.05, key="w_genre")
    w_cast  = st.slider("🎬 Cast & Director",  0.0, 1.0, 0.20, 0.05, key="w_cast")

    total_w = w_plot + w_genre + w_cast
    if abs(total_w - 1.0) < 0.001:
        st.success(f"✓ Weights sum to {total_w:.2f}")
    else:
        st.warning(f"⚠ Weights sum to {total_w:.2f} — will be auto-normalised")

    st.divider()

    st.markdown(
        "<div style='color:#f5f5f5;font-size:.93rem;font-weight:700;margin-bottom:.5rem;'>🎲 Discovery Mode</div>",
        unsafe_allow_html=True,
    )
    serendipity = st.toggle(
        "Explore (Serendipity)",
        value=False,
        help="Exploit = pure similarity ranking. Explore = mixes in titles from adjacent K-Means clusters for novelty.",
    )
    st.caption("🎯 Exploit: strict similarity matches\n\n✨ Explore: injects adjacent-cluster novelty")

    st.divider()

    top_n = st.slider("Number of Recommendations", 3, 10, 5, 1, key="top_n")

    st.divider()

    # graph-specific controls, only meaningful in graph view
    st.markdown(
        "<div style='color:#f5f5f5;font-size:.93rem;font-weight:700;margin-bottom:.5rem;'>🕸️ Graph Options</div>",
        unsafe_allow_html=True,
    )
    secondary_threshold = st.slider(
        "Secondary Edge Threshold",
        min_value=0.05, max_value=0.60, value=0.25, step=0.05,
        help="Minimum mutual similarity score to draw edges between recommendation nodes.",
        key="sec_threshold",
    )
    graph_height = st.slider(
        "Graph Height (px)", min_value=400, max_value=800, value=580, step=20, key="graph_h"
    )


# ── page header ───────────────────────────────────────────────────────────────
st.markdown(
    "<div style='color:#E50914;font-size:1.05rem;font-weight:800;letter-spacing:.12em;margin-bottom:.4rem;'>"
    "🎬 HYBRID RECOMMENDER</div>",
    unsafe_allow_html=True,
)

selected_title = st.selectbox(
    "Search a title to get recommendations", titles, key="title_selector"
)
selected_row = df[df["title"].astype(str) == selected_title].iloc[0]


# ── hero card ─────────────────────────────────────────────────────────────────
def tag(label: str, value: str) -> str:
    return f"<span class='tag'>{label} <strong>{value}</strong></span>"


st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">{selected_row['title']}</div>
      <div class="hero-sub">Featured Spotlight · {selected_row['type']}</div>
      <div class="hero-desc">{selected_row.get('description', '')}</div>
      <div class="hero-tags">
        {tag('Type',     str(selected_row['type']))}
        {tag('Rating',   str(selected_row['rating']))}
        {tag('Year',     str(selected_row.get('release_year', 'N/A')))}
        {tag('Duration', str(selected_row.get('duration', 'N/A')))}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# active weight / mode display
mode_label = "✨ Explore (Serendipity)" if serendipity else "🎯 Exploit (Strict Matches)"
st.markdown(
    f"<div style='color:#a0a0a0;font-size:.83rem;margin-bottom:.8rem;'>"
    f"Mode: <strong style='color:#f5f5f5;'>{mode_label}</strong> &nbsp;·&nbsp; "
    f"Weights — Plot <strong style='color:#3b82f6;'>{w_plot:.0%}</strong> &nbsp;"
    f"Genre <strong style='color:#22c55e;'>{w_genre:.0%}</strong> &nbsp;"
    f"Cast <strong style='color:#f59e0b;'>{w_cast:.0%}</strong>"
    f"</div>",
    unsafe_allow_html=True,
)


# ── fetch recommendations ─────────────────────────────────────────────────────
with st.spinner("Computing hybrid similarities…"):
    try:
        recs, query_ms = get_recommendations(
            selected_title,
            top_n=top_n,
            w_plot=w_plot,
            w_genre=w_genre,
            w_cast=w_cast,
            serendipity=serendipity,
        )
    except Exception as exc:
        st.error(f"Could not load recommendations: {exc}")
        st.stop()


# ── telemetry badge ───────────────────────────────────────────────────────────
_badge_color = "#22c55e" if query_ms < 500 else ("#f59e0b" if query_ms < 2000 else "#E50914")
st.markdown(
    f"<div style='display:flex;align-items:center;gap:.5rem;margin-bottom:.7rem;'>"
    f"<span style='background:rgba(34,197,94,.1);border:1px solid rgba(34,197,94,.3);"
    f"border-radius:999px;padding:3px 12px;font-size:.78rem;font-weight:700;"
    f"color:{_badge_color};letter-spacing:.04em;'>⚡ Query executed in {query_ms:.1f} ms</span>"
    f"<span style='color:#444;font-size:.75rem;'>{top_n} results · "
    f"{'Serendipity' if serendipity else 'Exploit'} mode</span>"
    f"</div>",
    unsafe_allow_html=True,
)

# ── view mode toggle ──────────────────────────────────────────────────────────
view_mode = st.radio(
    "View mode",
    options=["🎴 Grid Cards", "🕸️ Knowledge Graph"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    key="view_mode",
)

st.markdown(
    f"<div class='sec-hdr'>{'Recommended For You' if view_mode == '🎴 Grid Cards' else 'Similarity Network'}</div>",
    unsafe_allow_html=True,
)


# ── helper functions ──────────────────────────────────────────────────────────
def badge_class(score: float) -> str:
    if score >= 60:
        return "badge-high"
    if score >= 35:
        return "badge-mid"
    return "badge-low"


def breakdown_bar(label: str, pct: float, fill_class: str) -> str:
    w = max(0, min(100, pct))
    return (
        f"<div class='bd-row'>"
        f"  <span class='bd-label'>{label}</span>"
        f"  <div class='bd-bar-bg'><div class='{fill_class}' style='width:{w}%;'></div></div>"
        f"  <span class='bd-val'>{pct:.0f}%</span>"
        f"</div>"
    )


def cast_preview(raw: str, max_names: int = 3) -> str:
    if raw in ("N/A", "nan", ""):
        return "N/A"
    names = [n.strip() for n in raw.replace("…", "").split(",") if n.strip()]
    return ", ".join(names[:max_names]) + ("…" if len(names) > max_names else "")


# ══════════════════════════════════════════════════════════════════════════════
# VIEW A: Grid Cards
# ══════════════════════════════════════════════════════════════════════════════
if view_mode == "🎴 Grid Cards":
    cols = st.columns(top_n)
    for idx, row in enumerate(recs.itertuples(index=False)):
        with cols[idx]:
            score      = row.similarity_score_percent
            bc         = badge_class(score)
            is_novel   = getattr(row, "is_serendipity", False)
            novel_pill = "<span class='serendip-pill'>✨ Novel</span>" if is_novel else ""
            genres_short = ", ".join([g.strip() for g in str(row.genres).split(",")][:2])
            cast_str     = cast_preview(str(row.cast))

            st.markdown(
                f"""
                <div class="g-card">
                  <div class="g-card-title">{row.title}{novel_pill}</div>
                  <div class="badge-wrap">
                    <span class="match-badge {bc}">{score:.1f}% Match</span>
                  </div>
                  <div class="breakdown">
                    {breakdown_bar("Plot",  row.plot_pct,  "bd-bar-fill-plot")}
                    {breakdown_bar("Genre", row.genre_pct, "bd-bar-fill-genre")}
                    {breakdown_bar("Cast",  row.cast_pct,  "bd-bar-fill-cast")}
                  </div>
                  <div class="card-tags">
                    <span class="card-tag">📅 {row.release_year}</span>
                    <span class="card-tag">🔞 {row.rating}</span>
                    <span class="card-tag">⏱ {row.duration}</span>
                  </div>
                  <div class="g-card-meta" style="margin-top:.6rem;">
                    🎭 {genres_short}<br>
                    🎬 {cast_str}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# VIEW B: Knowledge Graph
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("<div class='graph-panel'>", unsafe_allow_html=True)

    with st.spinner("Building similarity network…"):
        try:
            # reuse cached matrices — no re-computation
            df_full, mat_plot_m, mat_genre_m, mat_cast_m = get_matrices()

            # normalise weights (mirror recommender logic)
            total_w = w_plot + w_genre + w_cast or 1.0
            nw_plot  = w_plot  / total_w
            nw_genre = w_genre / total_w
            nw_cast  = w_cast  / total_w

            fig = build_network_figure(
                selected_title=selected_title,
                recs=recs,
                df=df_full,
                mat_plot=mat_plot_m,
                mat_genre=mat_genre_m,
                mat_cast=mat_cast_m,
                w_plot=nw_plot,
                w_genre=nw_genre,
                w_cast=nw_cast,
                secondary_threshold=secondary_threshold,
                height=graph_height,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={
                    "displayModeBar": True,
                    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
                    "displaylogo": False,
                    "scrollZoom": True,
                },
            )
        except Exception as exc:
            st.error(f"Could not build network graph: {exc}")

    st.markdown(
        "<div class='graph-legend-hint'>"
        "🔴 Red node = selected title &nbsp;·&nbsp; "
        "Node size ∝ match score &nbsp;·&nbsp; "
        "Colour = K-Means cluster &nbsp;·&nbsp; "
        "Dotted edges = sibling similarity"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # compact table below graph for reference
    with st.expander("📋 Recommendations table", expanded=False):
        display_df = recs[["title", "genres", "release_year", "rating", "similarity_score_percent"]].copy()
        display_df.columns = ["Title", "Genres", "Year", "Rating", "Match %"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── export section ────────────────────────────────────────────────────────────
with st.expander("📥 Download Results & Executive Export", expanded=False):
    import io
    from datetime import datetime

    st.markdown(
        "<div style='color:#a0a0a0;font-size:.83rem;margin-bottom:.8rem;'>"
        "Export the current recommendation results as CSV or a formatted Executive Summary.</div>",
        unsafe_allow_html=True,
    )

    export_col1, export_col2 = st.columns(2)

    # ── CSV export ────────────────────────────────────────────────────────────
    with export_col1:
        st.markdown(
            "<div style='color:#f5f5f5;font-size:.88rem;font-weight:700;"
            "margin-bottom:.5rem;'>📄 CSV Export</div>",
            unsafe_allow_html=True,
        )
        export_df = recs[[
            "title", "genres", "release_year", "rating", "duration",
            "similarity_score_percent", "plot_pct", "genre_pct", "cast_pct",
            "is_serendipity",
        ]].copy()
        export_df.columns = [
            "Title", "Genres", "Release Year", "Rating", "Duration",
            "Match %", "Plot Weight %", "Genre Weight %", "Cast Weight %",
            "Serendipity Pick",
        ]
        # append query metadata as a header comment via StringIO
        csv_buf = io.StringIO()
        csv_buf.write(
            f"# Netflix AI · Hybrid Recommender Export\n"
            f"# Query: {selected_title}\n"
            f"# Weights: Plot={w_plot:.0%}  Genre={w_genre:.0%}  Cast={w_cast:.0%}\n"
            f"# Mode: {'Serendipity (Explore)' if serendipity else 'Exploit (Strict)'}\n"
            f"# Executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  "
            f"[{query_ms:.1f} ms]\n"
            f"#\n"
        )
        export_df.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode("utf-8")

        safe_title = "".join(c if c.isalnum() or c in " _-" else "_" for c in selected_title)[:40]
        st.download_button(
            label="⬇ Download CSV",
            data=csv_bytes,
            file_name=f"netflix_recs_{safe_title}.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.dataframe(export_df, use_container_width=True, hide_index=True, height=220)

    # ── Markdown executive summary ────────────────────────────────────────────
    with export_col2:
        st.markdown(
            "<div style='color:#f5f5f5;font-size:.88rem;font-weight:700;"
            "margin-bottom:.5rem;'>📝 Executive Summary</div>",
            unsafe_allow_html=True,
        )

        # build the markdown report
        def _rec_row(r) -> str:
            novel = " ✨" if r.is_serendipity else ""
            return (
                f"| {r.Title} | {r.Genres[:40]}… | "
                f"{r['Release Year']} | {r.Rating} | "
                f"**{r['Match %']:.1f}%** | "
                f"{r['Plot Weight %']:.0f}% / {r['Genre Weight %']:.0f}% / {r['Cast Weight %']:.0f}%{novel} |"
            )

        rec_table_rows = "\n".join(_rec_row(r) for r in export_df.itertuples())
        ts = datetime.now().strftime("%B %d, %Y at %H:%M")

        md_report = f"""# Netflix AI — Executive Recommendation Report

**Generated:** {ts}
**Query executed in:** {query_ms:.1f} ms

---

## Selected Title

**{selected_title}**
- Type: {selected_row['type']}
- Rating: {selected_row['rating']}
- Year: {selected_row.get('release_year', 'N/A')}
- Duration: {selected_row.get('duration', 'N/A')}

> {str(selected_row.get('description', ''))[:300]}{'…' if len(str(selected_row.get('description', ''))) > 300 else ''}

---

## Engine Configuration

| Parameter | Value |
|-----------|-------|
| Mode | {'✨ Serendipity (Explore)' if serendipity else '🎯 Exploit (Strict Matches)'} |
| Plot Weight | {w_plot:.0%} |
| Genre Weight | {w_genre:.0%} |
| Cast & Director Weight | {w_cast:.0%} |
| Results Requested | {top_n} |

---

## Top {len(export_df)} Recommendations

| Title | Genres | Year | Rating | Match | Plot / Genre / Cast |
|-------|--------|------|--------|-------|---------------------|
{rec_table_rows}

> ✨ = Serendipity pick from adjacent cluster

---

*Generated by Netflix AI Platform · Hybrid Weighted Engine v2*
"""

        md_bytes = md_report.encode("utf-8")
        st.download_button(
            label="⬇ Download Markdown Report",
            data=md_bytes,
            file_name=f"exec_summary_{safe_title}.md",
            mime="text/markdown",
            use_container_width=True,
        )

        # preview in an expander
        with st.expander("Preview report", expanded=False):
            st.markdown(md_report)


# ── footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="footer-banner">
        <strong>Trending Categories</strong><br>
        Crime · International · Science &amp; Nature · Thrillers · Documentaries
    </div>
    <div class="foot">Curated content discovery · Hybrid Weighted Engine v2</div>
    """,
    unsafe_allow_html=True,
)
