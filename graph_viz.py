"""
graph_viz.py – Network Graph Generator for the Hybrid Recommender
=================================================================
Builds a node-edge topology using NetworkX spring layout and renders
it as an interactive Plotly figure.

Node encoding
  • Central node  : selected title  (larger, red accent ring)
  • Satellite nodes: top-N recs     (size ∝ similarity score)
  • Node colour   : K-Means cluster ID  (8-colour palette)
  • Edge width    : ∝ blended similarity score

Secondary edges between recommendation siblings are drawn when their
mutual similarity (via the cached blended matrix) exceeds `secondary_threshold`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

CLUSTERED_PKL = Path(__file__).resolve().parent / "clustered_df.pkl"

# ── cluster colour palette (8 clusters) ──────────────────────────────────────
_CLUSTER_COLORS = {
    0: "#e05b6a",   # warm red  – Global Drama & Thrillers
    1: "#22c55e",   # green     – Kids & Family
    2: "#f59e0b",   # amber     – Hollywood Blockbusters
    3: "#3b82f6",   # blue      – Binge-Worthy TV
    4: "#a78bfa",   # violet    – Comedies & Stand-Up
    5: "#fb923c",   # orange    – Independent & World Cinema
    6: "#ec4899",   # pink      – Crime, Mystery & Horror
    7: "#06b6d4",   # cyan      – International & World TV
}
_DEFAULT_COLOR  = "#888888"
_CENTER_COLOR   = "#E50914"   # Netflix red for the query node
_BG_COLOR       = "#0a0a0a"
_PAPER_COLOR    = "#0a0a0a"
_EDGE_COLOR_MAIN = "rgba(229,9,20,0.55)"
_EDGE_COLOR_SEC  = "rgba(120,120,140,0.30)"
_TEXT_COLOR      = "#f5f5f5"
_MUTED_COLOR     = "#888888"


# ── cluster metadata loader ───────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_cluster_data() -> dict[str, dict]:
    """Returns {title_lower: {cluster, persona, type}} from clustered_df.pkl."""
    if not CLUSTERED_PKL.exists():
        return {}
    try:
        cdf = joblib.load(CLUSTERED_PKL)
        out: dict[str, dict] = {}
        for _, row in cdf.iterrows():
            out[str(row["title"]).lower()] = {
                "cluster": int(row["cluster"]),
                "persona": str(row.get("persona", f"Cluster {row['cluster']}")),
                "type":    str(row.get("type", "Unknown")),
            }
        return out
    except Exception:
        return {}


# ── mutual similarity between two recommendation titles ──────────────────────

def _mutual_score(
    title_a: str,
    title_b: str,
    df: pd.DataFrame,
    mat_plot,
    mat_genre,
    mat_cast,
    w_plot: float,
    w_genre: float,
    w_cast: float,
) -> float:
    """Blended cosine similarity between two arbitrary titles."""
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    mask_a = df["title"].astype(str).str.lower() == title_a.lower()
    mask_b = df["title"].astype(str).str.lower() == title_b.lower()
    if not mask_a.any() or not mask_b.any():
        return 0.0
    ia = df.index[mask_a][0]
    ib = df.index[mask_b][0]

    sp = float(cos_sim(mat_plot[ia],  mat_plot[ib])[0, 0])
    sg = float(cos_sim(mat_genre[ia], mat_genre[ib])[0, 0])
    sc = float(cos_sim(mat_cast[ia],  mat_cast[ib])[0, 0])
    return w_plot * sp + w_genre * sg + w_cast * sc


# ── main graph builder ────────────────────────────────────────────────────────

def build_network_figure(
    selected_title: str,
    recs: pd.DataFrame,
    df: pd.DataFrame,
    mat_plot,
    mat_genre,
    mat_cast,
    w_plot: float = 0.50,
    w_genre: float = 0.30,
    w_cast: float = 0.20,
    secondary_threshold: float = 0.25,
    height: int = 580,
) -> go.Figure:
    """
    Build and return a Plotly network graph figure.

    Parameters
    ----------
    selected_title     : query title (centre node)
    recs               : DataFrame from get_recommendations()
    df                 : full cleaned dataset DataFrame
    mat_plot/genre/cast: cached sparse similarity matrices
    w_*                : blending weights (pre-normalised)
    secondary_threshold: min mutual similarity to draw sibling edges
    height             : figure height in pixels
    """
    cluster_data = _load_cluster_data()

    # ── 1. Build NetworkX graph ───────────────────────────────────────────────
    G = nx.Graph()

    # centre node
    centre_lower = selected_title.lower()
    centre_info  = cluster_data.get(centre_lower, {})
    centre_cluster = centre_info.get("cluster", -1)
    centre_persona  = centre_info.get("persona", "Unknown")
    centre_type     = centre_info.get("type", "Unknown")

    sel_row = df[df["title"].astype(str).str.lower() == centre_lower]
    centre_year   = int(sel_row["release_year"].iloc[0]) if not sel_row.empty and pd.notna(sel_row["release_year"].iloc[0]) else "N/A"
    centre_rating = str(sel_row["rating"].iloc[0]) if not sel_row.empty else "N/A"

    G.add_node(
        selected_title,
        is_center=True,
        score=100.0,
        cluster=centre_cluster,
        persona=centre_persona,
        content_type=centre_type,
        year=centre_year,
        rating=centre_rating,
    )

    # satellite nodes + primary edges
    rec_titles: list[str] = []
    for row in recs.itertuples(index=False):
        t = str(row.title)
        rec_titles.append(t)
        info = cluster_data.get(t.lower(), {})

        G.add_node(
            t,
            is_center=False,
            score=float(row.similarity_score_percent),
            cluster=info.get("cluster", -1),
            persona=info.get("persona", "Unknown"),
            content_type=info.get("type", str(getattr(row, "type", "Unknown"))),
            year=getattr(row, "release_year", "N/A"),
            rating=str(getattr(row, "rating", "N/A")),
            genres=str(getattr(row, "genres", "")),
            plot_pct=float(getattr(row, "plot_pct", 0)),
            genre_pct=float(getattr(row, "genre_pct", 0)),
            cast_pct=float(getattr(row, "cast_pct", 0)),
            is_serendipity=bool(getattr(row, "is_serendipity", False)),
        )
        G.add_edge(
            selected_title, t,
            weight=float(row.similarity_score_percent) / 100.0,
            edge_type="primary",
        )

    # secondary edges between sibling recommendations
    for i in range(len(rec_titles)):
        for j in range(i + 1, len(rec_titles)):
            ta, tb = rec_titles[i], rec_titles[j]
            ms = _mutual_score(ta, tb, df, mat_plot, mat_genre, mat_cast,
                               w_plot, w_genre, w_cast)
            if ms >= secondary_threshold:
                G.add_edge(ta, tb, weight=ms, edge_type="secondary")

    # ── 2. Layout (Fruchterman-Reingold / spring) ─────────────────────────────
    pos = nx.spring_layout(
        G,
        seed=42,
        k=2.2 / max(1, np.sqrt(G.number_of_nodes())),
        iterations=120,
        weight="weight",
    )

    # ── 3. Build Plotly traces ────────────────────────────────────────────────
    edge_traces_primary  = []
    edge_traces_secondary = []

    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = float(data.get("weight", 0.3))

        if data.get("edge_type") == "primary":
            # draw gradient line by splitting into two segments meeting at midpoint
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            width = max(1.0, w * 8)
            edge_traces_primary.append(
                go.Scatter(
                    x=[x0, mx, x1, None],
                    y=[y0, my, y1, None],
                    mode="lines",
                    line=dict(width=width, color=_EDGE_COLOR_MAIN),
                    hoverinfo="none",
                    showlegend=False,
                )
            )
        else:
            width = max(0.5, w * 3)
            edge_traces_secondary.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=width, color=_EDGE_COLOR_SEC, dash="dot"),
                    hoverinfo="none",
                    showlegend=False,
                )
            )

    # ── 4. Node traces (one per cluster for legend) ───────────────────────────
    # Group nodes by cluster so legend entries are meaningful
    cluster_node_groups: dict[int, list[str]] = {}
    for node in G.nodes():
        c = G.nodes[node].get("cluster", -1)
        cluster_node_groups.setdefault(c, []).append(node)

    node_traces: list[go.Scatter] = []
    legend_seen: set[int] = set()

    for cluster_id, nodes_in_cluster in sorted(cluster_node_groups.items()):
        cluster_color = _CLUSTER_COLORS.get(cluster_id, _DEFAULT_COLOR)

        xs, ys, sizes, colors, texts, hovers = [], [], [], [], [], []

        for node in nodes_in_cluster:
            nd   = G.nodes[node]
            x, y = pos[node]
            xs.append(x)
            ys.append(y)

            is_center = nd.get("is_center", False)
            score     = nd.get("score", 50.0)

            # node size: centre is fixed large; satellites scale with score
            size = 38 if is_center else max(16, score * 0.38)
            sizes.append(size)

            # centre gets accent colour; others get cluster colour
            colors.append(_CENTER_COLOR if is_center else cluster_color)

            # short label shown on graph
            label = node if len(node) <= 18 else node[:16] + "…"
            texts.append(label)

            # rich hover
            persona  = nd.get("persona",      "Unknown")
            ctype    = nd.get("content_type", "Unknown")
            year     = nd.get("year",          "N/A")
            rating   = nd.get("rating",        "N/A")
            score_s  = f"{score:.1f}%" if not is_center else "Selected"
            genres   = nd.get("genres", "")
            plot_p   = nd.get("plot_pct",  0)
            genre_p  = nd.get("genre_pct", 0)
            cast_p   = nd.get("cast_pct",  0)
            novel    = "✨ Novel Pick  |  " if nd.get("is_serendipity") else ""

            hover_parts = [
                f"<b>{node}</b>",
                f"{novel}{ctype}  ·  {year}  ·  {rating}",
                f"Cluster: {persona}",
                f"Match: <b>{score_s}</b>",
            ]
            if not is_center:
                hover_parts.append(
                    f"Plot {plot_p:.0f}%  ·  Genre {genre_p:.0f}%  ·  Cast {cast_p:.0f}%"
                )
            if genres:
                genres_short = ", ".join(g.strip() for g in genres.split(",")[:2])
                hover_parts.append(f"🎭 {genres_short}")

            hovers.append("<br>".join(hover_parts))

        persona_label = nd.get("persona", f"Cluster {cluster_id}")  # last nd in loop is fine for label

        show_in_legend = cluster_id not in legend_seen and cluster_id != -1
        if show_in_legend:
            legend_seen.add(cluster_id)

        node_traces.append(
            go.Scatter(
                x=xs, y=ys,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=colors,
                    opacity=0.92,
                    line=dict(
                        width=[3 if G.nodes[n].get("is_center") else 1.2 for n in nodes_in_cluster],
                        color=["#ffffff" if G.nodes[n].get("is_center") else "rgba(255,255,255,0.2)"
                               for n in nodes_in_cluster],
                    ),
                ),
                text=texts,
                textposition="top center",
                textfont=dict(
                    size=[11 if G.nodes[n].get("is_center") else 9 for n in nodes_in_cluster],
                    color=_TEXT_COLOR,
                    family="Segoe UI, Roboto, sans-serif",
                ),
                hovertext=hovers,
                hovertemplate="%{hovertext}<extra></extra>",
                name=persona_label if show_in_legend else "",
                showlegend=show_in_legend,
                legendgroup=str(cluster_id),
            )
        )

    # ── 5. Assemble figure ────────────────────────────────────────────────────
    all_traces = edge_traces_secondary + edge_traces_primary + node_traces

    fig = go.Figure(data=all_traces)

    fig.update_layout(
        paper_bgcolor=_PAPER_COLOR,
        plot_bgcolor=_BG_COLOR,
        height=height,
        margin=dict(l=10, r=10, t=44, b=10),
        hovermode="closest",
        font=dict(color=_TEXT_COLOR, family="Segoe UI, Roboto, sans-serif"),
        title=dict(
            text=f"<b>Content Similarity Network</b>  ·  <span style='color:#888;font-size:12px;'>center: {selected_title}</span>",
            font=dict(size=14, color=_TEXT_COLOR),
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showline=False,
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            showline=False,
        ),
        legend=dict(
            bgcolor="rgba(20,20,20,0.85)",
            bordercolor="#333",
            borderwidth=1,
            font=dict(size=11, color=_TEXT_COLOR),
            title=dict(text="Cluster", font=dict(size=11, color="#888")),
            itemsizing="constant",
            x=1.01,
            y=1.0,
            xanchor="left",
        ),
        dragmode="pan",
    )

    return fig
