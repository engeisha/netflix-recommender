"""
xai_utils.py – Shared XAI helpers for Classifier and Rating Predictor pages.

Provides:
  • feature_impact_chart()   – Plotly horizontal bar of top driving words
  • proba_chart()            – Plotly full class probability breakdown
  • word_weights_from_rf()   – extract per-word contribution for a RF + TF-IDF model

Both classifiers use RandomForest, so we approximate local word importance as:
    word_contribution = global_feature_importance[i] * tfidf_value[i]
normalised to a [-1, +1]-style signed score using the class index.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── palette ───────────────────────────────────────────────────────────────────
_BG    = "#0e0e0e"
_PANEL = "#161616"
_TEXT  = "#f5f5f5"
_MUTED = "#888888"
_RED   = "#E50914"
_GREEN = "#22c55e"
_BLUE  = "#3b82f6"
_AMBER = "#f59e0b"

_PLOTLY_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_PANEL,
    font=dict(color=_TEXT, family="Segoe UI, Roboto, sans-serif", size=12),
    margin=dict(l=10, r=20, t=36, b=10),
    xaxis=dict(
        gridcolor="#222",
        zerolinecolor="#333",
        tickfont=dict(color=_MUTED),
    ),
    yaxis=dict(
        gridcolor="#222",
        tickfont=dict(color=_TEXT),
    ),
)


# ── word-level attribution ────────────────────────────────────────────────────

def word_weights_from_rf(
    feature_importances: np.ndarray,
    feature_names: list[str],
    input_vector: np.ndarray,
    tfidf_prefix: str = "tfidf_",
    top_n: int = 12,
) -> tuple[pd.DataFrame, float]:
    """
    Returns (word_df, elapsed_ms).

    word_df has columns: word, score, direction.
    Score = global_importance[i] * tfidf_input_value[i].
    """
    t0 = time.perf_counter()
    records = []
    for i, (name, imp) in enumerate(zip(feature_names, feature_importances)):
        if not name.startswith(tfidf_prefix):
            continue
        word = name[len(tfidf_prefix):]
        if not word:
            continue
        val = float(input_vector[i]) if i < len(input_vector) else 0.0
        score = float(imp) * val
        if score > 0:
            records.append({"word": word, "score": score, "direction": "pos"})

    df = pd.DataFrame(records)
    if df.empty:
        return df, round((time.perf_counter() - t0) * 1000, 2)

    df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return df, round((time.perf_counter() - t0) * 1000, 2)


def word_weights_from_rf_named(
    feature_importances: np.ndarray,
    feature_names: list[str],
    input_vector: np.ndarray,
    word_to_idx: dict[str, int],
    top_n: int = 12,
) -> tuple[pd.DataFrame, float]:
    """
    Variant for models where feature names are the raw vocabulary words
    (e.g. 'text__american' from ColumnTransformer).
    Prefix stripped automatically.

    Returns (word_df, elapsed_ms).
    """
    t0 = time.perf_counter()
    records = []
    for i, name in enumerate(feature_names):
        # strip transformer prefix like 'text__'
        word = name.split("__", 1)[1] if "__" in name else name
        val = float(input_vector[i]) if i < len(input_vector) else 0.0
        score = float(feature_importances[i]) * val
        if score > 0:
            records.append({"word": word, "score": score, "direction": "pos"})

    df = pd.DataFrame(records)
    if df.empty:
        return df, round((time.perf_counter() - t0) * 1000, 2)

    df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return df, round((time.perf_counter() - t0) * 1000, 2)


# ── charts ────────────────────────────────────────────────────────────────────

def feature_impact_chart(
    word_df: pd.DataFrame,
    title: str = "Top Words Driving This Prediction",
    pos_color: str = _GREEN,
    neg_color: str = _RED,
) -> go.Figure:
    """
    Horizontal bar chart of word contributions.
    word_df must have columns: word, score, direction.
    """
    if word_df.empty:
        fig = go.Figure()
        fig.update_layout(
            **_PLOTLY_LAYOUT,
            title=dict(text="No text features activated for this input.", font=dict(color=_MUTED)),
            height=180,
        )
        return fig

    df = word_df.sort_values("score")
    colors = [pos_color if d == "pos" else neg_color for d in df["direction"]]

    fig = go.Figure(
        go.Bar(
            x=df["score"],
            y=df["word"],
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>",
        )
    )
    base = {k: v for k, v in _PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **base,
        title=dict(
            text=title,
            font=dict(size=13, color=_TEXT),
            x=0,
            pad=dict(l=4),
        ),
        height=max(240, len(df) * 28 + 60),
        xaxis=dict(**_PLOTLY_LAYOUT["xaxis"], title="Contribution Score"),
        yaxis=dict(**_PLOTLY_LAYOUT["yaxis"], title=None),
        bargap=0.28,
    )
    return fig


def proba_chart(
    classes: list[str],
    probas: list[float],
    bar_colors: dict[str, str] | None = None,
    title: str = "Confidence Across All Classes",
) -> go.Figure:
    """
    Horizontal bar chart of predict_proba() across every class.
    Sorted descending. Bars are labelled with % values.
    """
    pairs = sorted(zip(classes, probas), key=lambda x: -x[1])
    labels = [p[0] for p in pairs]
    values = [p[1] * 100 for p in pairs]

    default_palette = [_RED, _AMBER, _BLUE, _GREEN, "#a78bfa", "#fb923c"]
    colors = []
    for i, lbl in enumerate(labels):
        if bar_colors and lbl in bar_colors:
            colors.append(bar_colors[lbl])
        else:
            colors.append(default_palette[i % len(default_palette)])

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(color=_TEXT, size=11),
            hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>",
            cliponaxis=False,
        )
    )
    base = {k: v for k, v in _PLOTLY_LAYOUT.items() if k != "xaxis"}
    fig.update_layout(
        **base,
        title=dict(
            text=title,
            font=dict(size=13, color=_TEXT),
            x=0,
            pad=dict(l=4),
        ),
        height=max(200, len(labels) * 44 + 70),
        xaxis=dict(
            **_PLOTLY_LAYOUT["xaxis"],
            range=[0, min(110, max(values) * 1.22)],
            ticksuffix="%",
        ),
        yaxis_title=None,
        bargap=0.35,
    )
    return fig


def whatif_comparison_chart(
    classes: list[str],
    probas_orig: list[float],
    probas_new: list[float],
    bar_colors: dict[str, str] | None = None,
) -> go.Figure:
    """
    Side-by-side grouped bar comparing original vs what-if probabilities.
    """
    default_palette = [_RED, _AMBER, _BLUE, _GREEN]
    orig_color = "#444"
    new_color  = _BLUE

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Original",
        x=classes,
        y=[p * 100 for p in probas_orig],
        marker_color=orig_color,
        hovertemplate="Original · <b>%{x}</b>: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        name="What-If",
        x=classes,
        y=[p * 100 for p in probas_new],
        marker_color=new_color,
        hovertemplate="What-If · <b>%{x}</b>: %{y:.1f}%<extra></extra>",
    ))

    base = {k: v for k, v in _PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")}
    fig.update_layout(
        **base,
        title=dict(
            text="What-If vs Original Prediction",
            font=dict(size=13, color=_TEXT),
            x=0,
        ),
        barmode="group",
        height=320,
        xaxis=dict(**_PLOTLY_LAYOUT["xaxis"], title=None),
        yaxis=dict(
            **_PLOTLY_LAYOUT["yaxis"],
            ticksuffix="%",
            title="Probability (%)",
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=_TEXT, size=11),
            orientation="h",
            y=1.08,
        ),
        bargap=0.22,
        bargroupgap=0.08,
    )
    return fig
