import streamlit as st

from src.ui import set_app_style, render_hero
from src.data_loader import get_ticker_options, get_ticker_symbol, load_price_data, add_basic_features

import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="Metals Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

set_app_style()


# -----------------------
# TOP NAVIGATION (REAL APP STYLE)
# -----------------------
nav = st.radio(
    "Navigation",
    ["Dashboard", "Overview", "Direction", "Risk"],
    horizontal=True,
    label_visibility="collapsed"
)

# -----------------------
# DASHBOARD PAGE
# -----------------------
if nav == "Dashboard":

    render_hero(
        "Metals Intelligence Dashboard",
        "A unified analytics platform for metals equities — combining market data, prediction models, and risk diagnostics."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Coverage", "Metals Equities")
    col2.metric("Modules", "4 Active")
    col3.metric("Model Layer", "Live + ML")
    col4.metric("Interface", "Dashboard")

    st.markdown("### Modules")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Open Overview"):
            st.session_state.page = "Overview"

    with c2:
        if st.button("Open Direction"):
            st.session_state.page = "Direction"

    with c3:
        if st.button("Open Risk"):
            st.session_state.page = "Risk"


# -----------------------
# OVERVIEW PAGE
# -----------------------
elif nav == "Overview":

    render_hero(
        "Overview",
        "Live market snapshot for selected metals stocks."
    )

    ticker_options = get_ticker_options()
    labels = list(ticker_options.keys())

    selected_label = st.selectbox("Select Stock", labels)

    ticker = get_ticker_symbol(selected_label)

    data = load_price_data(ticker, "6mo")

    latest = float(data["Close"].iloc[-1])
    prev = float(data["Close"].iloc[-2])
    change_pct = (latest - prev) / prev * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", ticker)
    col2.metric("Price", f"${latest:.2f}", f"{change_pct:.2f}%")
    col3.metric("Volume", f"{data['Volume'].mean():,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"]
    ))

    st.plotly_chart(fig, use_container_width=True)


# -----------------------
# DIRECTION PAGE
# -----------------------
elif nav == "Direction":

    render_hero(
        "Direction Prediction",
        "Short-term price direction using ML."
    )

    ticker_options = get_ticker_options()
    labels = list(ticker_options.keys())

    selected_label = st.selectbox("Select Stock", labels)
    ticker = get_ticker_symbol(selected_label)

    df = load_price_data(ticker, "2y")
    data = add_basic_features(df)

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)
    data["MA_Signal"] = data["MA_5"] - data["MA_20"]
    data = data.dropna()

    X = data[["Return_1d", "Return_5d", "MA_5", "MA_20", "Volatility_5", "MA_Signal"]]
    y = data["Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    pred = model.predict(X.iloc[[-1]])[0]
    prob = model.predict_proba(X.iloc[[-1]])[0][1]
    acc = accuracy_score(y_test, model.predict(X_test))

    label = "UP" if pred == 1 else "DOWN"

    col1, col2, col3 = st.columns(3)
    col1.metric("Ticker", ticker)
    col2.metric("Prediction", label)
    col3.metric("Probability", f"{prob:.2%}")

    st.write(f"Model Accuracy: {acc:.2%}")


# -----------------------
# RISK PAGE
# -----------------------
elif nav == "Risk":

    render_hero(
        "Risk & Volatility",
        "Measure volatility and downside risk."
    )

    ticker_options = get_ticker_options()
    labels = list(ticker_options.keys())

    selected_label = st.selectbox("Select Stock", labels)
    ticker = get_ticker_symbol(selected_label)

    data = load_price_data(ticker, "2y")

    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(20).std() * np.sqrt(252)

    vol = data["Volatility"].dropna().iloc[-1]
    var = np.percentile(data["Return"].dropna(), 5)

    col1, col2 = st.columns(2)
    col1.metric("Volatility (20d)", f"{vol:.2%}")
    col2.metric("VaR (95%)", f"{var:.2%}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Volatility"]))

    st.plotly_chart(fig, use_container_width=True)