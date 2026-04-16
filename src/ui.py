import streamlit as st

def set_app_style():
    st.markdown(
        """
        <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #9aa0a6;
            margin-bottom: 1.5rem;
        }

        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.75rem;
        }

        div[data-testid="stMetric"] {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 14px;
            border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_title(title: str, subtitle: str):
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">{subtitle}</div>', unsafe_allow_html=True)

def render_section_header(text: str):
    st.markdown(f'<div class="section-header">{text}</div>', unsafe_allow_html=True)