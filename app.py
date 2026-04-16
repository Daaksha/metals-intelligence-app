import streamlit as st
from src.ui import set_app_style, render_title

st.set_page_config(
    page_title="Metals Intelligence Platform",
    page_icon="⛏️",
    layout="wide"
)

set_app_style()

render_title(
    "Metals Intelligence Platform",
    "A financial analytics dashboard for metals investors."
)

st.write("Use the sidebar to navigate through the app.")

st.markdown("""
### Modules
- Overview
- Direction Prediction
- Risk and Volatility
- Fraud / Manipulation Detection
- Recommendation Engine
- Macro Dashboard
- Stop-Loss Assistant
""")