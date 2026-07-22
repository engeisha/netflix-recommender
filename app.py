from pathlib import Path

import pandas as pd
import streamlit as st

from recommender import get_recommendations


DATASET_PATH = Path(__file__).resolve().parent / "cleaned_netflix_titles.csv"


@st.cache_data
def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def render_tag(label: str, value: str) -> str:
    return f"<span style='display:inline-block; padding:6px 10px; margin:0 8px 8px 0; border:1px solid #2d2d2d; border-radius:999px; background:#161616; color:#cfcfcf; font-size:0.83rem;'>{label} <strong style='color:#f5f5f5;'>{value}</strong></span>"


def main() -> None:
    st.set_page_config(page_title="Netflix Recommender", page_icon="🎬", layout="wide")

    st.markdown(
        """
        <style>
        :root {
            --bg: #0c0c0c;
            --panel: #1e1e1e;
            --panel-2: #171717;
            --text: #f5f5f5;
            --muted: #a0a0a0;
            --accent: #E50914;
            --border: #2a2a2a;
        }
        .stApp {
            background: var(--bg);
            color: var(--text);
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 2.5rem;
        }
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.6rem 0 1rem;
        }
        .brand {
            color: var(--accent);
            font-size: 1.25rem;
            font-weight: 800;
            letter-spacing: 0.14em;
        }
        .nav {
            color: var(--muted);
            font-size: 0.92rem;
            display: flex;
            gap: 1rem;
        }
        .hero {
            background: linear-gradient(135deg, rgba(229,9,20,0.16), rgba(12,12,12,0.95));
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 1.4rem 1.5rem;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.3);
        }
        .hero-title {
            color: var(--text);
            font-size: 2.05rem;
            font-weight: 700;
            margin: 0 0 0.35rem;
        }
        .hero-subtitle {
            color: #e7e7e7;
            font-size: 0.98rem;
            margin-bottom: 0.6rem;
        }
        .hero-desc {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.55;
            max-width: 700px;
            margin-bottom: 0.9rem;
        }
        .hero-actions {
            display: flex;
            gap: 0.7rem;
            flex-wrap: wrap;
        }
        .btn {
            background: var(--accent);
            color: #FFFFFF !important;
            border: none;
            border-radius: 999px;
            padding: 0.6rem 1rem;
            font-weight: 700;
            text-decoration: none;
            display: inline-block;
        }
        .btn.secondary {
            background: transparent;
            border: 1px solid var(--border);
            color: var(--text);
        }
        .search-panel {
            display: flex;
            justify-content: center;
            margin: 0.8rem 0 1.2rem;
        }
        .search-box {
            width: min(560px, 100%);
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.25rem 0.6rem;
        }
        .card {
            background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem;
            transition: transform 180ms ease, border-color 180ms ease;
            height: 100%;
        }
        .card:hover {
            transform: translateY(-3px);
            border-color: rgba(229,9,20,0.45);
        }
        .card-title {
            color: var(--text);
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.4rem;
        }
        .card-meta {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin-bottom: 0.7rem;
        }
        .match-badge {
            display: inline-block;
            color: #22c55e;
            font-size: 0.88rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }
        .match-badge.red {
            color: var(--accent);
        }
        .footer-banner {
            background: linear-gradient(90deg, rgba(229,9,20,0.18), rgba(229,9,20,0.05));
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-top: 1.2rem;
            color: var(--text);
        }
        .foot {
            color: var(--muted);
            text-align: center;
            padding-top: 0.9rem;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="topbar">
            <div class="brand">NETFLIX</div>
            <div class="nav">
                <span>Home</span>
                <span>Series</span>
                <span>Films</span>
                <span>My List</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = load_dataset()
    titles = sorted(df["title"].dropna().astype(str).unique().tolist())

    if not titles:
        st.error("No titles available")
        return

    selected_title = st.selectbox("", titles, key="title_selector")

    selected_row = df[df["title"].astype(str) == selected_title].iloc[0]

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{selected_row['title']}</div>
            <div class="hero-subtitle">Featured Spotlight · {selected_row['type']}</div>
            <div class="hero-desc">{selected_row['description']}</div>
            <div class="hero-actions">
                <a class="btn" href="#">▶ Watch Trailer</a>
                <a class="btn secondary" href="#">ℹ Details</a>
            </div>
            <div style="margin-top: 0.9rem;">{render_tag('Type', str(selected_row['type']))}{render_tag('Rating', str(selected_row['rating']))}{render_tag('Year', str(selected_row.get('release_year', 'N/A')))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="search-panel">
            <div class="search-box">
                <div style="text-align:center; color:#a0a0a0; padding:0.4rem 0;">Featured Title Search</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='margin: 0.4rem 0 0.8rem; color:#f5f5f5; font-size:1.1rem; font-weight:700;'>Recommended For You</div>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading recommendations..."):
        recommendations = get_recommendations(selected_title, top_n=5)

    cols = st.columns(5)
    for idx, row in enumerate(recommendations.itertuples(index=False)):
        with cols[idx]:
            match_class = "match-badge red" if row.similarity_score_percent >= 8 else "match-badge"
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-title">{row.title}</div>
                    <div class="card-meta">{row.genres}</div>
                    <div class="{match_class}">{row.similarity_score_percent:.2f}% Match</div>
                    <a class="btn" href="#" style="font-size:0.85rem;">View Content</a>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer-banner">
            <strong>Trending Categories</strong><br>
            Crime • International • Science & Nature • Thrillers • Documentaries
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='foot'>Curated content discovery for the next binge.</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
