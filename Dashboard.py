import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.ui import (
    set_app_style, render_hero, render_section,
    render_info, render_alert, render_danger, render_success,
    render_badge, divider, CHART,
)
from src.data_loader import (
    get_ticker_options, get_ticker_symbol,
    load_price_data, add_technical_indicators,
    get_ml_features, compute_drawdown,
    get_beneish_scores, load_macro_data,
    load_sector_normalised,
)

st.set_page_config(
    page_title="Metals Intelligence Dashboard",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
set_app_style()

ticker_options = get_ticker_options()
ticker_labels  = list(ticker_options.keys())

nav = st.radio(
    "Navigation",
    ["Dashboard", "Overview", "Direction", "Risk",
     "Fraud", "Recommendation", "Stop-Loss", "Macro"],
    horizontal=True,
    label_visibility="collapsed",
)

# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if nav == "Dashboard":

    render_hero(
        "Metals Intelligence Platform  ◈  Spring 2026",
        "Metals Intelligence Dashboard",
        "An institutional-grade decision-support system for metals sector equities — "
        "integrating live market data, machine learning prediction models, financial "
        "integrity screening, and macroeconomic analysis into a single unified platform.",
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks Covered",    "8 Tickers",    "Metals Universe")
    c2.metric("Analytics Modules", "7 Active",     "End-to-End Coverage")
    c3.metric("Data Layer",        "Live APIs",    "Real-Time Market Data")
    c4.metric("ML Models",         "4 Algorithms", "Predictive Intelligence")

    divider()

    render_section(
        "Platform Modules",
        "Navigate using the tab bar above. Each module is independently powered "
        "but shares a common data and feature layer.",
    )

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 01 — Market Data</div>
            <div class="nav-card-title">Overview</div>
            <div class="nav-card-text">Live candlestick price chart with volume bars,
            Bollinger Bands, moving averages (MA10/20/50), RSI momentum indicator,
            and MACD trend signal. Includes period selector and CSV download.</div>
        </div>""", unsafe_allow_html=True)
    with r1c2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 02 — Machine Learning</div>
            <div class="nav-card-title">Direction Prediction</div>
            <div class="nav-card-text">Short-horizon price direction signal (Up / Down)
            powered by a Random Forest classifier trained on 2 years of technical features.
            Shows prediction probability, model accuracy, and feature importance.</div>
        </div>""", unsafe_allow_html=True)
    with r1c3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 03 — Risk Management</div>
            <div class="nav-card-title">Risk & Volatility</div>
            <div class="nav-card-text">Annualised volatility forecasting, rolling drawdown
            analysis, Value-at-Risk (VaR) at 95% confidence, and risk-level classification
            (Low / Medium / High) based on rolling metrics.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 04 — Integrity</div>
            <div class="nav-card-title">Fraud Detection</div>
            <div class="nav-card-text">Beneish-style financial statement quality screening
            using gross margin, asset quality, leverage, and profitability ratios to flag
            potential earnings manipulation risk.</div>
        </div>""", unsafe_allow_html=True)
    with r2c2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 05 — Decision Engine</div>
            <div class="nav-card-title">Recommendation</div>
            <div class="nav-card-text">Hybrid MetalScore (0–100) aggregating direction
            signals, momentum, volatility penalty, and accounting quality into a final
            Buy / Hold / Avoid recommendation with signal breakdown.</div>
        </div>""", unsafe_allow_html=True)
    with r2c3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 06 — Risk Control</div>
            <div class="nav-card-title">Stop-Loss Assistant</div>
            <div class="nav-card-text">Personalised stop-loss price and position sizing
            calculator calibrated to the user's risk tolerance, investment amount,
            and the stock's current ATR-based volatility profile.</div>
        </div>""", unsafe_allow_html=True)
    with r2c4:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Module 07 — Macro Context</div>
            <div class="nav-card-title">Macro Dashboard</div>
            <div class="nav-card-text">Gold, silver, copper, and crude oil price trends
            alongside treasury yields, the VIX fear index, and the US Dollar Index —
            the macro backdrop driving metals markets globally.</div>
        </div>""", unsafe_allow_html=True)

    divider()

    col_a, col_b = st.columns([2, 1])
    with col_a:
        render_section("How This Platform Works")
        render_info(
            "<strong>Data Layer</strong> — Live market prices and financial statement data "
            "are fetched from Yahoo Finance APIs. Historical data (up to 2 years) powers "
            "model training while the latest data drives live recommendations.<br><br>"
            "<strong>Feature Layer</strong> — Raw price data is transformed into 15+ technical "
            "indicators including RSI, MACD, Bollinger Bands, ATR, and rolling volatility.<br><br>"
            "<strong>Model Layer</strong> — Separate machine learning models handle each task: "
            "classification for price direction, regression for volatility, rule-based scoring "
            "for fraud screening, and a hybrid weighted engine for final recommendations.<br><br>"
            "<strong>Decision Layer</strong> — All model outputs converge into a single "
            "MetalScore and a plain-language recommendation with full signal attribution."
        )
    with col_b:
        render_section("Why Metals?")
        st.markdown("""
        <div class="soft-panel">
            <div style="font-size:0.88rem;color:#3a3530;line-height:1.75;">
                Metals stocks are among the most volatile and macro-sensitive equities
                in any market. They respond to commodity price cycles, global industrial
                demand, geopolitical risk, and monetary policy — making them both
                high-opportunity and high-risk investments that require systematic,
                data-driven analysis rather than intuition alone.<br><br>
                This platform gives retail investors access to the same analytical
                framework used by professional metals-sector analysts.
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Overview":

    render_hero(
        "Module 01 — Market Data",
        "Market Overview",
        "Live price action, volume analysis, and technical indicators for the selected "
        "metals stock. All charts update automatically with the latest available market data.",
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels)
    with col_sel2:
        period = st.selectbox(
            "Time Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=2,
        )

    ticker = get_ticker_symbol(selected_label)
    df_raw = load_price_data(ticker, period)

    if df_raw.empty:
        st.error("No data returned for this ticker. Try another.")
        st.stop()

    df = add_technical_indicators(df_raw)

    close        = df["Close"].squeeze()
    prev_close   = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
    latest       = float(close.iloc[-1])
    chg          = latest - prev_close
    chg_pct      = chg / prev_close * 100
    period_high  = float(df["High"].squeeze().max())
    period_low   = float(df["Low"].squeeze().min())
    avg_vol      = float(df["Volume"].squeeze().mean())
    latest_rsi   = float(df["RSI"].dropna().iloc[-1])
    latest_vol20 = float(df["Volatility_20d"].dropna().iloc[-1]) * 100
    latest_macd  = float(df["MACD"].dropna().iloc[-1])
    macd_sig     = float(df["MACD_Signal"].dropna().iloc[-1])
    rsi_label    = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
    macd_label   = "Bullish" if latest_macd > macd_sig else "Bearish"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Last Price",     f"${latest:,.2f}",      f"{chg_pct:+.2f}%")
    k2.metric("Period High",    f"${period_high:,.2f}")
    k3.metric("Period Low",     f"${period_low:,.2f}")
    k4.metric("RSI (14)",       f"{latest_rsi:.1f}",    rsi_label)
    k5.metric("20d Volatility", f"{latest_vol20:.1f}%")
    k6.metric("MACD Signal",    macd_label,             f"MACD {latest_macd:+.3f}")

    render_info(
        f"<strong>{ticker}</strong> last traded at <strong>${latest:,.2f}</strong>, "
        f"{'up' if chg >= 0 else 'down'} <strong>{abs(chg_pct):.2f}%</strong> from "
        f"the prior session. The 14-period RSI of <strong>{latest_rsi:.1f}</strong> "
        f"indicates <strong>{rsi_label.lower()}</strong> momentum. "
        f"MACD is <strong>{macd_label.lower()}</strong> with the MACD line "
        f"{'above' if latest_macd > macd_sig else 'below'} the signal line. "
        f"Annualised 20-day volatility is <strong>{latest_vol20:.1f}%</strong>, "
        f"with average daily volume of <strong>{avg_vol:,.0f}</strong> shares."
    )

    divider()

    render_section(
        "Price Chart — Candlestick with Bollinger Bands & Moving Averages",
        "Green candles = price closed higher than open (bullish). "
        "Red candles = price closed lower (bearish). "
        "Bollinger Bands expand during high volatility and contract during low volatility. "
        "Moving averages (MA10, MA20, MA50) reveal the underlying price trend. "
        "Volume bars confirm the strength of each price move.",
    )

    fig1 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
    )
    fig1.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["BB_Upper"].squeeze()) + list(df["BB_Lower"].squeeze()[::-1]),
        fill="toself", fillcolor="rgba(163,139,92,0.09)",
        line=dict(color="rgba(255,255,255,0)"), name="Bollinger Band",
    ), row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"].squeeze(),
        line=dict(color="rgba(163,139,92,0.5)", width=1, dash="dot"),
        name="BB Upper", showlegend=False,
    ), row=1, col=1)
    fig1.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"].squeeze(),
        line=dict(color="rgba(163,139,92,0.5)", width=1, dash="dot"),
        name="BB Lower", showlegend=False,
    ), row=1, col=1)
    fig1.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=df["Close"].squeeze(),
        increasing=dict(line=dict(color="#27ae60"), fillcolor="#27ae60"),
        decreasing=dict(line=dict(color="#e74c3c"), fillcolor="#e74c3c"),
        name=ticker,
    ), row=1, col=1)
    for ma, color, lbl in [
        ("MA_10", "#2471a3", "MA 10"),
        ("MA_20", "#d35400", "MA 20"),
        ("MA_50", "#7d3c98", "MA 50"),
    ]:
        if ma in df.columns:
            fig1.add_trace(go.Scatter(
                x=df.index, y=df[ma].squeeze(),
                line=dict(color=color, width=1.5), name=lbl,
            ), row=1, col=1)
    vol_colors = [
        "#27ae60" if float(df["Close"].squeeze().iloc[i]) >= float(df["Open"].squeeze().iloc[i])
        else "#e74c3c" for i in range(len(df))
    ]
    fig1.add_trace(go.Bar(
        x=df.index, y=df["Volume"].squeeze(),
        marker_color=vol_colors, marker_opacity=0.55, name="Volume",
    ), row=2, col=1)
    fig1.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=640, margin=dict(l=10, r=10, t=20, b=10),
        xaxis_rangeslider_visible=False,
        legend=dict(bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
    )
    for row in [1, 2]:
        fig1.update_xaxes(gridcolor="rgba(163,139,92,0.10)",
                          linecolor="rgba(163,139,92,0.15)",
                          tickfont=dict(size=11, color="#6b6560"),
                          showgrid=True, row=row, col=1)
        fig1.update_yaxes(gridcolor="rgba(163,139,92,0.10)",
                          linecolor="rgba(163,139,92,0.15)",
                          tickfont=dict(size=11, color="#6b6560"),
                          showgrid=True, row=row, col=1)
    fig1.update_yaxes(title_text="Price (USD)",
                      title_font=dict(size=11, color="#6b6560"), row=1, col=1)
    fig1.update_yaxes(title_text="Volume",
                      title_font=dict(size=11, color="#6b6560"), row=2, col=1)
    st.plotly_chart(fig1, use_container_width=True)

    divider()

    render_section(
        "Momentum Indicators — RSI & MACD",
        "RSI above 70 = overbought (potential sell signal). RSI below 30 = oversold (potential buy signal). "
        "MACD crossing above the signal line = bullish crossover. "
        "Growing green histogram bars = strengthening bullish momentum.",
    )

    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=[
            "RSI — Relative Strength Index (14-Period)",
            "MACD — Moving Average Convergence Divergence (12/26/9)",
        ],
        row_heights=[0.45, 0.55], vertical_spacing=0.12,
    )
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["RSI"].squeeze(),
        line=dict(color="#2471a3", width=2),
        fill="tozeroy", fillcolor="rgba(36,113,163,0.07)", name="RSI",
    ), row=1, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.07)",
                   line_width=0, row=1, col=1)
    fig2.add_hrect(y0=0, y1=30, fillcolor="rgba(39,174,96,0.07)",
                   line_width=0, row=1, col=1)
    fig2.add_hline(y=70, line=dict(color="#e74c3c", width=1, dash="dash"), row=1, col=1)
    fig2.add_hline(y=30, line=dict(color="#27ae60", width=1, dash="dash"), row=1, col=1)
    fig2.add_hline(y=50, line=dict(color="#6b6560", width=1, dash="dot"),  row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["MACD"].squeeze(),
        line=dict(color="#2471a3", width=2), name="MACD",
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"].squeeze(),
        line=dict(color="#d35400", width=1.5, dash="dash"), name="Signal Line",
    ), row=2, col=1)
    macd_hist   = df["MACD_Hist"].squeeze()
    hist_colors = ["#27ae60" if v >= 0 else "#e74c3c" for v in macd_hist]
    fig2.add_trace(go.Bar(
        x=df.index, y=macd_hist,
        marker_color=hist_colors, marker_opacity=0.65, name="Histogram",
    ), row=2, col=1)
    fig2.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=540, margin=dict(l=10, r=10, t=35, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="left", x=0, font=dict(size=11)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
    )
    for row in [1, 2]:
        fig2.update_xaxes(gridcolor="rgba(163,139,92,0.10)",
                          linecolor="rgba(163,139,92,0.15)",
                          tickfont=dict(size=11, color="#6b6560"),
                          showgrid=True, row=row, col=1)
        fig2.update_yaxes(gridcolor="rgba(163,139,92,0.10)",
                          linecolor="rgba(163,139,92,0.15)",
                          tickfont=dict(size=11, color="#6b6560"),
                          showgrid=True, row=row, col=1)
    fig2.update_yaxes(title_text="RSI", range=[0, 100],
                      title_font=dict(size=11, color="#6b6560"), row=1, col=1)
    fig2.update_yaxes(title_text="MACD",
                      title_font=dict(size=11, color="#6b6560"), row=2, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    divider()

    render_section(
        "Recent Trading Sessions",
        "Last 15 sessions showing OHLCV data alongside key indicators. Most recent first.",
    )
    display_cols = ["Open", "High", "Low", "Close", "Volume",
                    "RSI", "MACD", "BB_Pct", "Volatility_20d"]
    available = [c for c in display_cols if c in df.columns]
    recent    = df[available].tail(15).copy()
    for col in ["Open", "High", "Low", "Close"]:
        if col in recent.columns:
            recent[col] = recent[col].squeeze().apply(lambda x: f"${x:,.2f}")
    if "Volume" in recent.columns:
        recent["Volume"] = recent["Volume"].squeeze().apply(lambda x: f"{int(x):,}")
    if "RSI" in recent.columns:
        recent["RSI"] = recent["RSI"].squeeze().apply(lambda x: f"{x:.1f}")
    if "MACD" in recent.columns:
        recent["MACD"] = recent["MACD"].squeeze().apply(lambda x: f"{x:+.4f}")
    if "BB_Pct" in recent.columns:
        recent["BB_Pct"] = recent["BB_Pct"].squeeze().apply(lambda x: f"{x*100:.1f}%")
        recent = recent.rename(columns={"BB_Pct": "BB Position"})
    if "Volatility_20d" in recent.columns:
        recent["Volatility_20d"] = recent["Volatility_20d"].squeeze().apply(
            lambda x: f"{x*100:.1f}%")
        recent = recent.rename(columns={"Volatility_20d": "20d Vol (Ann.)"})
    st.dataframe(recent.iloc[::-1], use_container_width=True)
    st.download_button(
        label="Download Full Dataset as CSV",
        data=df_raw.to_csv().encode("utf-8"),
        file_name=f"{ticker}_{period}_data.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Direction":

    render_hero(
        "Module 02 — Machine Learning",
        "Direction Prediction",
        "A machine learning model trained on 2 years of technical indicators to predict "
        "whether the selected metals stock is likely to move Up or Down over the next "
        "trading session. Outputs include a directional signal, confidence probability, "
        "model accuracy, and feature importance.",
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="dir_ticker")
    with col_sel2:
        model_choice = st.selectbox(
            "Model Type",
            ["Random Forest (Recommended)", "Logistic Regression (Baseline)"],
            key="dir_model",
        )

    ticker = get_ticker_symbol(selected_label)

    with st.spinner(f"Loading 2 years of data and training model for {ticker}..."):
        df_raw = load_price_data(ticker, "2y")

    if df_raw.empty:
        st.error("No data returned. Try another ticker.")
        st.stop()

    data, feature_cols = get_ml_features(df_raw)
    data["Target"] = (data["Close"].squeeze().shift(-1) > data["Close"].squeeze()).astype(int)
    data = data.dropna(subset=feature_cols + ["Target"])

    if len(data) < 100:
        st.error("Not enough data to train the model.")
        st.stop()

    X = data[feature_cols]
    y = data["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    if "Random Forest" in model_choice:
        model = RandomForestClassifier(n_estimators=200, max_depth=6,
                                       random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=1000, random_state=42)

    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    accuracy    = accuracy_score(y_test, y_pred)
    latest_feat = X.iloc[[-1]]
    prediction  = model.predict(latest_feat)[0]
    probability = model.predict_proba(latest_feat)[0]
    prob_up     = probability[1]
    prob_down   = probability[0]
    signal      = "UP" if prediction == 1 else "DOWN"
    sig_color   = "#27ae60" if signal == "UP" else "#e74c3c"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ticker",              ticker)
    k2.metric("Predicted Direction", signal,            f"{prob_up:.1%} confidence up")
    k3.metric("Model Accuracy",      f"{accuracy:.1%}", "on held-out test set")
    k4.metric("Training Samples",    f"{len(X_train):,}", f"{len(X_test):,} test samples")

    if signal == "UP":
        render_success(
            f"<strong>Bullish Signal — {ticker} is predicted to move UP.</strong> "
            f"The model assigns a <strong>{prob_up:.1%}</strong> probability of an upward move "
            f"in the next session. Model test accuracy: <strong>{accuracy:.1%}</strong>."
        )
    else:
        render_danger(
            f"<strong>Bearish Signal — {ticker} is predicted to move DOWN.</strong> "
            f"The model assigns a <strong>{prob_down:.1%}</strong> probability of a downward move "
            f"in the next session. Model test accuracy: <strong>{accuracy:.1%}</strong>."
        )

    divider()

    render_section(
        "Prediction Confidence Gauge",
        "Values above 60% indicate a stronger signal. Values near 50% suggest high uncertainty — "
        "the model sees roughly equal probability of either direction.",
    )

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob_up * 100,
            delta={"reference": 50, "valueformat": ".1f",
                   "increasing": {"color": "#27ae60"},
                   "decreasing": {"color": "#e74c3c"}},
            number={"suffix": "%", "font": {"size": 42, "color": "#1a1a1a",
                                            "family": "DM Serif Display"}},
            title={"text": "Probability of UP Move",
                   "font": {"size": 14, "color": "#6b6560", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100],
                         "tickwidth": 1, "tickcolor": "#6b6560",
                         "tickfont": {"size": 11}},
                "bar": {"color": sig_color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1,
                "bordercolor": "rgba(163,139,92,0.2)",
                "steps": [
                    {"range": [0,  30],  "color": "rgba(231,76,60,0.12)"},
                    {"range": [30, 50],  "color": "rgba(231,76,60,0.05)"},
                    {"range": [50, 70],  "color": "rgba(39,174,96,0.05)"},
                    {"range": [70, 100], "color": "rgba(39,174,96,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#a38b5c", "width": 3},
                    "thickness": 0.75, "value": 50,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#faf6f0",
            font=dict(family="DM Sans"),
            height=300,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["DOWN", "UP"],
            y=[prob_down * 100, prob_up * 100],
            marker_color=["#e74c3c", "#27ae60"],
            marker_opacity=0.85,
            text=[f"{prob_down:.1%}", f"{prob_up:.1%}"],
            textposition="outside",
            textfont=dict(size=14, color="#1a1a1a", family="DM Serif Display"),
            width=0.45,
        ))
        fig_bar.add_hline(y=50, line=dict(color="#a38b5c", width=1.5, dash="dash"))
        fig_bar.update_layout(
            paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
            font=dict(family="DM Sans", color="#1a1a1a", size=12),
            height=300, margin=dict(l=10, r=10, t=40, b=10),
            title=dict(text="Up vs Down Probability",
                       font=dict(size=14, color="#1a1a1a"), x=0),
            yaxis=dict(range=[0, 110], title="Probability (%)",
                       gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            xaxis=dict(tickfont=dict(size=13, color="#1a1a1a")),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    divider()

    render_section(
        "Feature Importance — What Is Driving the Prediction?",
        "Higher importance = the model relied more heavily on that signal. "
        "RSI and MACD features typically dominate in trending markets. "
        "Volatility and Bollinger Band features dominate during choppy conditions.",
    )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    feat_df = pd.DataFrame({
        "Feature":    feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=True)

    FEATURE_LABELS = {
        "Return_1d":      "1-Day Return",
        "Return_5d":      "5-Day Return",
        "Return_10d":     "10-Day Return",
        "Return_20d":     "20-Day Return",
        "MACD":           "MACD",
        "MACD_Signal":    "MACD Signal Line",
        "MACD_Hist":      "MACD Histogram",
        "RSI":            "RSI (14)",
        "BB_Width":       "Bollinger Band Width",
        "BB_Pct":         "BB Position (%)",
        "ATR_Pct":        "ATR (Normalised)",
        "Volatility_20d": "20-Day Volatility",
        "Volume_Ratio":   "Volume Ratio",
        "Momentum_10":    "10-Day Momentum",
        "Momentum_20":    "20-Day Momentum",
    }
    feat_df["Label"] = feat_df["Feature"].map(FEATURE_LABELS).fillna(feat_df["Feature"])
    bar_colors = [
        "#27ae60" if v >= feat_df["Importance"].median() else "#a38b5c"
        for v in feat_df["Importance"]
    ]
    fig_feat = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Label"],
        orientation="h",
        marker_color=bar_colors, marker_opacity=0.85,
        text=[f"{v:.3f}" for v in feat_df["Importance"]],
        textposition="outside",
        textfont=dict(size=10, color="#6b6560"),
    ))
    fig_feat.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=460, margin=dict(l=10, r=60, t=20, b=10),
        xaxis=dict(title="Importance Score",
                   gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        yaxis=dict(tickfont=dict(size=11, color="#1a1a1a")),
        showlegend=False,
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    divider()

    render_section(
        "Recent Prediction vs Actual — Signal History",
        "Green markers = correct predictions. Red markers = incorrect predictions. "
        "The price line shows actual closing price over the same period.",
    )

    results_df = X_test.copy()
    results_df["Actual"]    = y_test.values
    results_df["Predicted"] = y_pred
    results_df["Correct"]   = (results_df["Actual"] == results_df["Predicted"])
    results_df["Close"]     = data.loc[X_test.index, "Close"].squeeze().values

    correct   = results_df[results_df["Correct"]]
    incorrect = results_df[~results_df["Correct"]]

    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(
        x=results_df.index, y=results_df["Close"],
        line=dict(color="#a38b5c", width=2), name="Close Price",
    ))
    fig_hist.add_trace(go.Scatter(
        x=correct.index, y=correct["Close"],
        mode="markers",
        marker=dict(color="#27ae60", size=8, symbol="circle",
                    line=dict(color="white", width=1)),
        name="Correct Prediction",
    ))
    fig_hist.add_trace(go.Scatter(
        x=incorrect.index, y=incorrect["Close"],
        mode="markers",
        marker=dict(color="#e74c3c", size=8, symbol="x",
                    line=dict(color="#e74c3c", width=2)),
        name="Incorrect Prediction",
    ))
    fig_hist.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        yaxis=dict(title="Price (USD)", gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    with st.expander("Full Classification Report — Precision, Recall & F1 Score"):
        render_info(
            "Precision = how often correct when predicting a class. "
            "Recall = how often all actual instances of a class are found. "
            "F1 = harmonic mean of precision and recall."
        )
        st.code(classification_report(y_test, y_pred,
                                      target_names=["DOWN (0)", "UP (1)"]))

    render_info(
        "<strong>Disclaimer:</strong> This model is for educational purposes only. "
        "No model can consistently predict short-term market movements. "
        "Always combine signals with fundamental analysis and your own judgment."
    )


# ══════════════════════════════════════════════════════════════════════════════
# RISK
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Risk":

    render_hero(
        "Module 03 — Risk Management",
        "Risk & Volatility",
        "Quantitative risk analysis for the selected metals stock — covering annualised "
        "volatility, rolling drawdown, Value-at-Risk (VaR), and risk classification. "
        "All metrics are computed from live market data and updated automatically.",
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="risk_ticker")
    with col_sel2:
        period = st.selectbox(
            "Time Period", ["6mo", "1y", "2y"],
            index=1, key="risk_period",
        )

    ticker = get_ticker_symbol(selected_label)
    df_raw = load_price_data(ticker, period)

    if df_raw.empty:
        st.error("No data returned. Try another ticker.")
        st.stop()

    df       = add_technical_indicators(df_raw)
    returns  = df["Return_1d"].dropna()
    vol_20   = float(df["Volatility_20d"].dropna().iloc[-1])
    vol_60   = float(df["Volatility_60d"].dropna().iloc[-1])
    var_95   = float(np.percentile(returns, 5))
    var_99   = float(np.percentile(returns, 1))
    avg_ret  = float(returns.mean())
    drawdown = compute_drawdown(df)
    max_dd   = float(drawdown.min())
    curr_dd  = float(drawdown.iloc[-1])
    sharpe   = (avg_ret * 252) / vol_20 if vol_20 > 0 else 0

    if vol_20 < 0.25:
        risk_level = "Low"
        risk_fn    = render_success
    elif vol_20 < 0.45:
        risk_level = "Medium"
        risk_fn    = render_alert
    else:
        risk_level = "High"
        risk_fn    = render_danger

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Ticker",           ticker)
    k2.metric("20d Volatility",   f"{vol_20*100:.1f}%",  "Annualised")
    k3.metric("60d Volatility",   f"{vol_60*100:.1f}%",  "Annualised")
    k4.metric("Risk Level",       risk_level)
    k5.metric("VaR (95%)",        f"{var_95*100:.2f}%",  "Daily worst-case")
    k6.metric("Max Drawdown",     f"{max_dd*100:.1f}%",  f"Current: {curr_dd*100:.1f}%")

    risk_fn(
        f"<strong>{ticker} Risk Level: {risk_level}.</strong> "
        f"The stock's 20-day annualised volatility is <strong>{vol_20*100:.1f}%</strong>. "
        f"At 95% confidence, the maximum expected daily loss (Value-at-Risk) is "
        f"<strong>{abs(var_95)*100:.2f}%</strong> of position value. "
        f"The maximum historical drawdown is <strong>{abs(max_dd)*100:.1f}%</strong>. "
        f"The annualised Sharpe ratio is <strong>{sharpe:.2f}</strong> — "
        f"{'above 1.0 is generally considered good.' if sharpe > 1 else 'below 1.0 suggests risk may not be fully compensated by returns.'}"
    )

    divider()

    render_section(
        "Rolling Volatility — 20-Day vs 60-Day Annualised",
        "Annualised volatility expressed as a percentage. "
        "The 20-day line reacts quickly to recent events (short-term risk). "
        "The 60-day line is smoother and reflects medium-term risk. "
        "When the 20-day line spikes above the 60-day line it signals sudden increased risk.",
    )

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=df.index, y=df["Volatility_20d"].squeeze() * 100,
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.07)",
        name="20-Day Volatility",
    ))
    fig_vol.add_trace(go.Scatter(
        x=df.index, y=df["Volatility_60d"].squeeze() * 100,
        line=dict(color="#2471a3", width=2, dash="dash"),
        name="60-Day Volatility",
    ))
    fig_vol.add_hline(y=25, line=dict(color="#27ae60", width=1, dash="dot"),
                      annotation_text="Low Risk (25%)",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color="#27ae60"))
    fig_vol.add_hline(y=45, line=dict(color="#e74c3c", width=1, dash="dot"),
                      annotation_text="High Risk (45%)",
                      annotation_position="bottom right",
                      annotation_font=dict(size=10, color="#e74c3c"))
    fig_vol.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="left", x=0, font=dict(size=11)),
        yaxis=dict(title="Annualised Volatility (%)",
                   gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    divider()

    render_section(
        "Rolling Drawdown — Peak-to-Trough Decline",
        "Drawdown measures how far the stock has fallen from its most recent peak. "
        "A drawdown of -20% means the stock is 20% below its highest recent point. "
        "Maximum drawdown is the largest peak-to-trough decline — a key risk management metric.",
    )

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="#e74c3c", width=1.5),
        name="Drawdown",
    ))
    fig_dd.add_hline(y=max_dd * 100,
                     line=dict(color="#7d3c98", width=1.5, dash="dash"),
                     annotation_text=f"Max Drawdown: {max_dd*100:.1f}%",
                     annotation_position="bottom right",
                     annotation_font=dict(size=10, color="#7d3c98"))
    fig_dd.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=380, margin=dict(l=10, r=10, t=20, b=10),
        yaxis=dict(title="Drawdown (%)",
                   gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
        showlegend=False,
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    divider()

    render_section(
        "Daily Return Distribution & Value-at-Risk (VaR)",
        "VaR at 95% confidence: 5% chance of losing more than this on any given day. "
        "VaR at 99% confidence: only 1% chance of exceeding this loss. "
        "These metrics are used by portfolio managers to size positions and set stop-loss levels.",
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns * 100, nbinsx=60,
        marker_color="#a38b5c", marker_opacity=0.75,
        name="Daily Returns",
    ))
    fig_dist.add_vline(x=var_95 * 100,
                       line=dict(color="#e74c3c", width=2, dash="dash"),
                       annotation_text=f"VaR 95%: {var_95*100:.2f}%",
                       annotation_position="top left",
                       annotation_font=dict(size=11, color="#e74c3c"))
    fig_dist.add_vline(x=var_99 * 100,
                       line=dict(color="#7d3c98", width=2, dash="dash"),
                       annotation_text=f"VaR 99%: {var_99*100:.2f}%",
                       annotation_position="top left",
                       annotation_font=dict(size=11, color="#7d3c98"))
    fig_dist.add_vline(x=avg_ret * 100,
                       line=dict(color="#27ae60", width=1.5, dash="dot"),
                       annotation_text=f"Avg: {avg_ret*100:.3f}%",
                       annotation_position="top right",
                       annotation_font=dict(size=11, color="#27ae60"))
    fig_dist.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Daily Return (%)",
                   gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        yaxis=dict(title="Frequency",
                   gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        bargap=0.05, showlegend=False,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    divider()

    render_section("Risk Summary Statistics")
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Volatility Metrics</strong>", unsafe_allow_html=True)
        for label, value in {
            "20-Day Annualised Volatility": f"{vol_20*100:.2f}%",
            "60-Day Annualised Volatility": f"{vol_60*100:.2f}%",
            "Risk Classification":          risk_level,
            "Annualised Sharpe Ratio":      f"{sharpe:.3f}",
            "Average Daily Return":         f"{avg_ret*100:.3f}%",
            "Annualised Return (est.)":     f"{avg_ret*252*100:.1f}%",
        }.items():
            st.markdown(
                f'<div class="stat-row"><span class="stat-label">{label}</span>'
                f'<span class="stat-value">{value}</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_t2:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Downside Risk Metrics</strong>", unsafe_allow_html=True)
        for label, value in {
            "VaR (95% Confidence)": f"{var_95*100:.3f}%",
            "VaR (99% Confidence)": f"{var_99*100:.3f}%",
            "Maximum Drawdown":     f"{max_dd*100:.2f}%",
            "Current Drawdown":     f"{curr_dd*100:.2f}%",
            "Worst Single Day":     f"{returns.min()*100:.2f}%",
            "Best Single Day":      f"{returns.max()*100:.2f}%",
        }.items():
            st.markdown(
                f'<div class="stat-row"><span class="stat-label">{label}</span>'
                f'<span class="stat-value">{value}</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# REMAINING PLACEHOLDERS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Fraud":
    render_hero("Module 04", "Fraud Detection", "Coming in Phase 5.")
    st.info("Phase 5 code goes here.")

elif nav == "Recommendation":
    render_hero("Module 05", "Recommendation Engine", "Coming in Phase 6.")
    st.info("Phase 6 code goes here.")

elif nav == "Stop-Loss":
    render_hero("Module 06", "Stop-Loss Assistant", "Coming in Phase 7.")
    st.info("Phase 7 code goes here.")

elif nav == "Macro":
    render_hero("Module 07", "Macro Dashboard", "Coming in Phase 7.")
    st.info("Phase 7 code goes here.")