import streamlit as st

CHART = {
    "bg": "#faf6f0", "paper": "#faf6f0",
    "grid": "rgba(163,139,92,0.10)", "text": "#1a1a1a",
    "muted": "#6b6560", "gold": "#a38b5c",
    "up": "#27ae60", "down": "#e74c3c",
    "blue": "#2471a3", "purple": "#7d3c98",
    "orange": "#d35400", "teal": "#1a9c8a",
    "line1": "#a38b5c", "line2": "#2471a3",
    "line3": "#7d3c98", "line4": "#1a9c8a",
}

CSS_BASE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800&family=DM+Serif+Display&display=swap');
html, body, .stApp, .main { font-family: 'DM Sans', sans-serif !important; background-color: #faf6f0 !important; color: #1a1a1a; }
.block-container { padding-top:1.25rem !important; padding-bottom:3rem !important; padding-left:2.75rem !important; padding-right:2.75rem !important; max-width:1480px !important; }
[data-testid="stSidebarNav"], section[data-testid="stSidebar"], [data-testid="collapsedControl"] { display:none !important; visibility:hidden !important; width:0 !important; height:0 !important; }
</style>
"""

CSS_NAV = """
<style>
div[role="radiogroup"] { display:flex; flex-wrap:wrap; gap:0.2rem; padding:0.6rem 0 0.85rem 0; border-bottom:2px solid rgba(163,139,92,0.18); margin-bottom:0; }
div[role="radiogroup"] > label { color:#1a1a1a !important; font-weight:600 !important; font-size:0.84rem !important; padding:0.42rem 1.05rem !important; border-radius:8px !important; border:1px solid transparent !important; white-space:nowrap; transition:all 0.15s ease; background:#1a1a1a; color:white !important; }
div[role="radiogroup"] > label:hover { background:rgba(163,139,92,0.12) !important; color:#1a1a1a !important; }
div[role="radiogroup"] > label[data-selected="true"] { background:#a38b5c !important; color:white !important; }
</style>
"""

CSS_HERO = """
<style>
.hero-wrap { padding:1.75rem 0 1.5rem 0; margin-bottom:1.75rem; border-bottom:1px solid rgba(163,139,92,0.18); }
.hero-eyebrow { font-size:0.7rem; font-weight:700; text-transform:uppercase; letter-spacing:0.12em; color:#a38b5c; margin-bottom:0.55rem; }
.hero-title { font-family:'DM Serif Display',serif; font-size:2.7rem; font-weight:400; color:#1a1a1a; line-height:1.1; margin-bottom:0.6rem; }
.hero-subtitle { font-size:1.0rem; color:#6b6560; max-width:860px; line-height:1.72; }
.section-header { margin-top:2.25rem; margin-bottom:0.2rem; }
.section-title { font-size:1.2rem; font-weight:700; color:#1a1a1a; letter-spacing:-0.01em; }
.section-desc { font-size:0.88rem; color:#7a7570; margin-top:0.3rem; margin-bottom:1.1rem; line-height:1.65; max-width:960px; }
</style>
"""

CSS_CARDS = """
<style>
div[data-testid="stMetric"] { background:#ffffff !important; border:1px solid rgba(163,139,92,0.22) !important; border-radius:16px !important; padding:18px 20px !important; box-shadow:0 2px 10px rgba(0,0,0,0.04) !important; }
div[data-testid="stMetricLabel"] > div { font-size:0.76rem !important; font-weight:600 !important; color:#7a7570 !important; text-transform:uppercase !important; letter-spacing:0.06em !important; }
div[data-testid="stMetricValue"] > div { font-size:1.7rem !important; font-weight:700 !important; color:#1a1a1a !important; font-family:'DM Serif Display',serif !important; }
div[data-testid="stMetricDelta"] svg { display:none; }
.nav-card { background:#ffffff; border:1px solid rgba(163,139,92,0.22); border-radius:18px; padding:22px 22px 18px 22px; min-height:185px; box-shadow:0 3px 14px rgba(0,0,0,0.04); transition:box-shadow 0.2s ease, transform 0.18s ease; }
.nav-card:hover { box-shadow:0 10px 28px rgba(163,139,92,0.18); transform:translateY(-3px); }
.nav-card-tag { font-size:0.68rem; font-weight:700; text-transform:uppercase; letter-spacing:0.1em; color:#a38b5c; margin-bottom:0.55rem; }
.nav-card-title { font-size:1.05rem; font-weight:700; color:#1a1a1a; margin-bottom:0.45rem; }
.nav-card-text { font-size:0.86rem; color:#6b6560; line-height:1.62; }
</style>
"""

CSS_PANELS = """
<style>
.info-panel { background:#fff9f2; border:1px solid rgba(163,139,92,0.28); border-left:4px solid #a38b5c; border-radius:0 12px 12px 0; padding:14px 18px; margin:0.8rem 0; font-size:0.91rem; color:#3a3530; line-height:1.7; }
.info-panel strong { color:#1a1a1a; font-weight:600; }
.alert-panel { background:#fff8e1; border:1px solid #ffd54f; border-left:4px solid #f9a825; border-radius:0 12px 12px 0; padding:14px 18px; margin:0.8rem 0; font-size:0.91rem; color:#5d4037; line-height:1.7; }
.danger-panel { background:#ffebee; border:1px solid #ef9a9a; border-left:4px solid #e53935; border-radius:0 12px 12px 0; padding:14px 18px; margin:0.8rem 0; font-size:0.91rem; color:#b71c1c; line-height:1.7; }
.success-panel { background:#e8f5e9; border:1px solid #a5d6a7; border-left:4px solid #43a047; border-radius:0 12px 12px 0; padding:14px 18px; margin:0.8rem 0; font-size:0.91rem; color:#1b5e20; line-height:1.7; }
.badge { display:inline-flex; align-items:center; gap:5px; padding:5px 15px; border-radius:20px; font-weight:700; font-size:0.9rem; letter-spacing:0.03em; }
.badge-buy { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }
.badge-sell, .badge-avoid { background:#ffebee; color:#b71c1c; border:1px solid #ef9a9a; }
.badge-hold { background:#fff8e1; color:#e65100; border:1px solid #ffcc02; }
.badge-low { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }
.badge-medium { background:#fff8e1; color:#e65100; border:1px solid #ffcc02; }
.badge-high { background:#ffebee; color:#b71c1c; border:1px solid #ef9a9a; }
.badge-up { background:#e8f5e9; color:#1b5e20; border:1px solid #a5d6a7; }
.badge-down { background:#ffebee; color:#b71c1c; border:1px solid #ef9a9a; }
.badge-normal { background:#e3f2fd; color:#0d47a1; border:1px solid #90caf9; }
.badge-warning { background:#fff8e1; color:#e65100; border:1px solid #ffcc02; }
</style>
"""

CSS_MISC = """
<style>
.stButton > button { background:#ffffff !important; color:#1a1a1a !important; border:1.5px solid #a38b5c !important; border-radius:10px !important; font-weight:600 !important; font-size:0.88rem !important; transition:all 0.18s ease !important; }
.stButton > button:hover { background:#a38b5c !important; color:white !important; }
.stSelectbox [data-baseweb="select"] > div { background:#ffffff !important; border:1.5px solid rgba(163,139,92,0.3) !important; border-radius:10px !important; }
.streamlit-expanderHeader { background:#ffffff !important; border:1px solid rgba(163,139,92,0.22) !important; border-radius:10px !important; font-weight:600 !important; }
.stDataFrame, .stDataFrame > div { border:1px solid rgba(163,139,92,0.15) !important; border-radius:12px !important; }
.custom-divider { border:none; border-top:1px solid rgba(163,139,92,0.15); margin:1.75rem 0; }
.soft-panel { background:#ffffff; border:1px solid rgba(163,139,92,0.18); border-radius:16px; padding:20px 24px; margin:0.75rem 0; }
.big-score { font-family:'DM Serif Display',serif; font-size:4.5rem; color:#1a1a1a; line-height:1; }
.score-label { font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#7a7570; margin-top:0.5rem; }
</style>
"""


def set_app_style():
    st.markdown(CSS_BASE, unsafe_allow_html=True)
    st.markdown(CSS_NAV, unsafe_allow_html=True)
    st.markdown(CSS_HERO, unsafe_allow_html=True)
    st.markdown(CSS_CARDS, unsafe_allow_html=True)
    st.markdown(CSS_PANELS, unsafe_allow_html=True)
    st.markdown(CSS_MISC, unsafe_allow_html=True)


def render_hero(eyebrow: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">{eyebrow}</div>
        <div class="hero-title">{title}</div>
        <div class="hero-subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


def render_section(title: str, description: str = ""):
    desc = f'<div class="section-desc">{description}</div>' if description else ""
    st.markdown(f"""
    <div class="section-header">
        <div class="section-title">{title}</div>
        {desc}
    </div>
    """, unsafe_allow_html=True)


def render_info(text: str):
    st.markdown(f'<div class="info-panel">{text}</div>', unsafe_allow_html=True)


def render_alert(text: str):
    st.markdown(f'<div class="alert-panel">{text}</div>', unsafe_allow_html=True)


def render_danger(text: str):
    st.markdown(f'<div class="danger-panel">{text}</div>', unsafe_allow_html=True)


def render_success(text: str):
    st.markdown(f'<div class="success-panel">{text}</div>', unsafe_allow_html=True)


def render_badge(text: str):
    cls = "badge-" + text.lower().replace(" ", "-")
    st.markdown(
        f'<span class="badge {cls}">{text.upper()}</span>',
        unsafe_allow_html=True,
    )


def divider():
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)


def chart_layout(fig, title: str = "", height: int = 500):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1a1a1a", family="DM Sans"), x=0),
        paper_bgcolor="#faf6f0",
        plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=height,
        margin=dict(l=10, r=10, t=45 if title else 20, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(163,139,92,0.2)", borderwidth=1, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(163,139,92,0.10)", linecolor="rgba(163,139,92,0.15)", tickfont=dict(size=11, color="#6b6560"), showgrid=True),
        yaxis=dict(gridcolor="rgba(163,139,92,0.10)", linecolor="rgba(163,139,92,0.15)", tickfont=dict(size=11, color="#6b6560"), showgrid=True),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)", font=dict(family="DM Sans", size=12, color="#1a1a1a")),
    )
    return fig