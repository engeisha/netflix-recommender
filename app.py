import streamlit as st

from ui_theme import inject_global_css

st.set_page_config(
    page_title="Netflix AI Platform",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()

st.markdown(
    """
    <style>
    :root {
        --bg: #0c0c0c; --panel: #1e1e1e; --text: #f5f5f5;
        --muted: #a0a0a0; --accent: #E50914; --border: #2a2a2a;
    }
    .stApp { background: var(--bg); color: var(--text); }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; }

    .brand-bar {
        display: flex; align-items: center; gap: 0.75rem;
        padding-bottom: 1.2rem; border-bottom: 1px solid var(--border);
        margin-bottom: 1.6rem;
    }
    .brand-logo { color: var(--accent); font-size: 2rem; font-weight: 900; letter-spacing: 0.12em; }
    .brand-tag { color: var(--muted); font-size: 1rem; letter-spacing: 0.06em; padding-left: 0.5rem; border-left: 2px solid var(--border); }

    .hero-section {
        background: linear-gradient(135deg, rgba(229,9,20,0.22) 0%, rgba(12,12,12,0.97) 60%);
        border: 1px solid var(--border); border-radius: 28px;
        padding: 2.4rem 2.6rem; margin-bottom: 2rem;
        box-shadow: 0 14px 40px rgba(0,0,0,0.45);
    }
    .hero-eyebrow { color: var(--accent); font-size: 0.85rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; margin-bottom: 0.5rem; }
    .hero-title { color: var(--text); font-size: 2.6rem; font-weight: 900; line-height: 1.15; margin-bottom: 0.7rem; }
    .hero-body { color: #c8c8c8; font-size: 1.02rem; line-height: 1.65; max-width: 680px; margin-bottom: 1.4rem; }
    .badge-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
    .badge {
        display: inline-block; padding: 0.35rem 0.9rem;
        border: 1px solid var(--border); border-radius: 999px;
        background: rgba(255,255,255,0.04); color: #e0e0e0; font-size: 0.82rem;
    }

    .section-title { color: var(--text); font-size: 1.15rem; font-weight: 700; margin: 0.2rem 0 1rem; }

    .module-card {
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        border: 1px solid var(--border); border-radius: 22px;
        padding: 1.4rem 1.3rem; height: 100%;
        transition: transform 200ms ease, border-color 200ms ease, box-shadow 200ms ease;
    }
    .module-card:hover { transform: translateY(-4px); border-color: rgba(229,9,20,0.5); box-shadow: 0 8px 24px rgba(229,9,20,0.12); }
    .module-icon { font-size: 2rem; margin-bottom: 0.65rem; }
    .module-title { color: var(--text); font-size: 1.05rem; font-weight: 800; margin-bottom: 0.4rem; }
    .module-desc { color: var(--muted); font-size: 0.92rem; line-height: 1.5; margin-bottom: 0.9rem; }
    .module-tag { display: inline-block; padding: 0.25rem 0.7rem; border-radius: 999px; font-size: 0.78rem; font-weight: 700; }
    .tag-ready { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); }
    .tag-model { background: rgba(229,9,20,0.12); color: #E50914; border: 1px solid rgba(229,9,20,0.3); }

    .stats-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }
    .stat-card {
        flex: 1; min-width: 140px;
        background: var(--panel); border: 1px solid var(--border);
        border-radius: 16px; padding: 1rem 1.2rem;
    }
    .stat-value { color: var(--accent); font-size: 1.7rem; font-weight: 900; }
    .stat-label { color: var(--muted); font-size: 0.83rem; margin-top: 0.1rem; }

    .feature-list { list-style: none; padding: 0; margin: 0; }
    .feature-list li { color: #c0c0c0; font-size: 0.93rem; padding: 0.35rem 0; border-bottom: 1px solid #1e1e1e; }
    .feature-list li::before { content: "✦ "; color: var(--accent); }

    .footer { color: var(--muted); text-align: center; font-size: 0.84rem; padding-top: 1.4rem; border-top: 1px solid var(--border); margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Brand bar ──────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="brand-bar">
        <div class="brand-logo">NETFLIX</div>
        <div class="brand-tag">AI Platform</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-section">
        <div class="hero-eyebrow">Welcome to the Platform</div>
        <div class="hero-title">Netflix AI Platform</div>
        <div class="hero-body">
            A unified machine-learning workspace for Netflix content intelligence.
            Explore AI-powered tools for personalised recommendations and automatic
            content-type classification — all backed by the Netflix titles dataset.
        </div>
        <div class="badge-row">
            <span class="badge">🧠 TF-IDF Similarity</span>
            <span class="badge">🎯 ML Classification</span>
            <span class="badge">📊 8 000+ Titles</span>
            <span class="badge">⚡ Real-time Inference</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Platform stats ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="stats-row">
        <div class="stat-card"><div class="stat-value">8 807</div><div class="stat-label">Titles in Dataset</div></div>
        <div class="stat-card"><div class="stat-value">2</div><div class="stat-label">AI Modules</div></div>
        <div class="stat-card"><div class="stat-value">TF-IDF</div><div class="stat-label">Vectorisation</div></div>
        <div class="stat-card"><div class="stat-value">Top-5</div><div class="stat-label">Recommendations</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Module cards ───────────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Available AI Modules</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-icon">🎬</div>
            <div class="module-title">Content Recommender</div>
            <div class="module-desc">
                Select any Netflix title and instantly receive five personalised
                recommendations. Powered by cosine similarity over TF-IDF vectors
                built from title metadata, cast, genres, and descriptions.
            </div>
            <span class="module-tag tag-ready">✓ Ready</span>
            &nbsp;
            <span class="module-tag tag-model">TF-IDF · Cosine Similarity</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-icon">🤖</div>
            <div class="module-title">Content Type Classifier</div>
            <div class="module-desc">
                Enter any content description and the classifier predicts whether
                it is a <strong style="color:#E50914;">Movie</strong> or a
                <strong style="color:#3b82f6;">TV Show</strong>, complete with
                confidence scores. Requires a trained
                <code>best_classifier.pkl</code> and <code>vectorizer.pkl</code>.
            </div>
            <span class="module-tag tag-model">Sklearn Classifier · Pickle</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Platform features ──────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Platform Features</div>", unsafe_allow_html=True)

f1, f2 = st.columns(2)
with f1:
    st.markdown(
        """
        <ul class="feature-list">
            <li>Dark Netflix-themed UI across all pages</li>
            <li>Cached data loading for fast repeat visits</li>
            <li>TF-IDF vectorisation over combined metadata fields</li>
            <li>Cosine similarity ranking for recommendations</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )
with f2:
    st.markdown(
        """
        <ul class="feature-list">
            <li>Plug-in classifier — swap any sklearn model</li>
            <li>Probability confidence bar for classifications</li>
            <li>Multi-page Streamlit navigation</li>
            <li>Modular, extensible codebase</li>
        </ul>
        """,
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<div class='footer'>Netflix AI Platform · Built with Streamlit &amp; scikit-learn</div>",
    unsafe_allow_html=True,
)
