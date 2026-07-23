"""
ui_theme.py – shared Netflix AI Platform UI helpers.
Import and call inject_global_css() at the top of every page
(after st.set_page_config) to apply the glassmorphism sidebar theme
and global dark-mode styles consistently across all pages.
"""
import streamlit as st


_GLOBAL_CSS = """
<style>
/* ═══════════════════════════════════════════════════════════════
   SIDEBAR – High-Visibility Netflix Layout
   ═══════════════════════════════════════════════════════════════ */

/* 1. Main sidebar container */
section[data-testid="stSidebar"] {
    background-color: #0F0F12 !important;
    border-right: 1px solid #2B2B32 !important;
}

/* 2. Force all text inside sidebar to white */
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Nav container */
section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
    background-color: transparent !important;
    font-family: "Segoe UI", Roboto, sans-serif !important;
}

/* List wrapper */
section[data-testid="stSidebar"] ul {
    border-bottom: none !important;
    margin-top: 1.5rem !important;
    padding: 0 0.5rem !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 0 !important;
}

/* 3. Nav links as distinct visible cards */
section[data-testid="stSidebar"] ul li a {
    background-color: #1A1A22 !important;
    border: 1px solid #33333F !important;
    border-radius: 10px !important;
    margin-bottom: 8px !important;
    padding: 10px 14px !important;
    transition: all 0.25s ease !important;
    text-decoration: none !important;
    display: block !important;
}

/* Inner text inside link buttons */
section[data-testid="stSidebar"] ul li a span,
section[data-testid="stSidebar"] ul li a p,
section[data-testid="stSidebar"] ul li a div {
    color: #F0F0F5 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* 4. Hover: red border glow + slide */
section[data-testid="stSidebar"] ul li a:hover {
    background-color: #262633 !important;
    border-color: #E50914 !important;
    transform: translateX(4px) !important;
    text-decoration: none !important;
}

/* 5. Active / selected page – glowing Netflix red pill */
section[data-testid="stSidebar"] ul li[aria-selected="true"] a,
section[data-testid="stSidebar"] ul li[data-selected="true"] a,
section[data-testid="stSidebar"] ul li a[aria-current="page"] {
    background: linear-gradient(135deg, #E50914 0%, #900C3F 100%) !important;
    border: 1px solid #FF3333 !important;
    box-shadow: 0px 4px 15px rgba(229, 9, 20, 0.5) !important;
}

section[data-testid="stSidebar"] ul li[aria-selected="true"] a span,
section[data-testid="stSidebar"] ul li[aria-selected="true"] a p,
section[data-testid="stSidebar"] ul li[aria-selected="true"] a div {
    color: #FFFFFF !important;
    font-weight: 700 !important;
}

/* 6. Brand pseudo-header */
section[data-testid="stSidebar"]::before {
    content: "NETFLIX  AI";
    display: block;
    color: #E50914 !important;
    font-family: "Segoe UI", Roboto, sans-serif;
    font-size: 0.82rem;
    font-weight: 900;
    letter-spacing: 0.24em;
    padding: 1.2rem 1.1rem 0;
}

/* 7. Red divider under brand */
section[data-testid="stSidebar"]::after {
    content: "";
    display: block;
    height: 1px;
    background: linear-gradient(90deg, #E50914 0%, rgba(229,9,20,0) 100%);
    margin: 0.45rem 1.1rem 0.2rem;
}

/* 8. Scrollbar */
section[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
section[data-testid="stSidebar"]::-webkit-scrollbar-track { background: transparent; }
section[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: rgba(229, 9, 20, 0.4);
    border-radius: 2px;
}
/* ═══════════════════════════════════════════════════════════════
   GLOBAL DARK BASE
   ═══════════════════════════════════════════════════════════════ */

:root {
    --bg: #0c0c0c;
    --text: #f5f5f5;
    --muted: #a0a0a0;
    --accent: #E50914;
    --border: #2a2a2a;
}

.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
}

/* Streamlit main content area */
.main .block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 3rem !important;
}

/* All text inputs / textareas */
textarea,
div[data-baseweb="textarea"] textarea {
    color: #FFFFFF !important;
    background-color: #1F1F1F !important;
    border: 1px solid #333333 !important;
}
textarea::placeholder { color: #888888 !important; }

input[type="number"],
div[data-baseweb="input"] input {
    color: #FFFFFF !important;
    background-color: #1F1F1F !important;
}

/* Selectbox / dropdown */
div[data-baseweb="select"] > div {
    background-color: #1a1a1a !important;
    border-color: #333 !important;
    color: #f5f5f5 !important;
}

/* Dataframe / table */
.stDataFrame { background: #141414 !important; }

/* Tabs */
button[data-baseweb="tab"] {
    color: #a0a0a0 !important;
    background: transparent !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #f5f5f5 !important;
    border-bottom: 2px solid #E50914 !important;
}
</style>
"""


def inject_global_css() -> None:
    """Inject glassmorphism sidebar + global dark-mode CSS into the current page."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
