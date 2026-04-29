import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
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

# ── Fix selectbox visibility
st.markdown("""
<style>
.stSelectbox [data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1.5px solid rgba(163,139,92,0.35) !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {
    color: #1a1a1a !important;
    font-weight: 500 !important;
}
[data-baseweb="popover"] {
    background: #ffffff !important;
}
[data-baseweb="popover"] li {
    color: #1a1a1a !important;
    background: #ffffff !important;
}
[data-baseweb="popover"] li:hover {
    background: rgba(163,139,92,0.12) !important;
    color: #1a1a1a !important;
}
[data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid rgba(163,139,92,0.25) !important;
    border-radius: 10px !important;
}
[role="option"] {
    color: #1a1a1a !important;
    background: #ffffff !important;
}
[role="option"]:hover {
    background: rgba(163,139,92,0.10) !important;
}
</style>
""", unsafe_allow_html=True)

ticker_options = get_ticker_options()
ticker_labels  = list(ticker_options.keys())

nav = st.radio(
    "Navigation",
    ["Dashboard", "Overview", "Direction", "Risk",
     "Fraud", "Recommendation", "Stop-Loss", "Macro & News"],
    horizontal=True,
    label_visibility="collapsed",
)

# ── Ticker metadata
TICKER_META = {
    "FCX":  {"name": "Freeport-McMoRan",  "metal": "Copper Mining",   "desc": "World's largest publicly traded copper producer. Highly sensitive to global industrial demand and China economic cycles."},
    "NEM":  {"name": "Newmont Corp",       "metal": "Gold Mining",     "desc": "Largest gold mining company globally. Performs well during inflation, dollar weakness, and geopolitical uncertainty."},
    "AA":   {"name": "Alcoa Corp",         "metal": "Aluminum",        "desc": "Primary aluminum producer. Affected by energy costs (smelting), global manufacturing demand, and trade policy."},
    "CLF":  {"name": "Cleveland-Cliffs",   "metal": "Steel",           "desc": "Major flat-rolled steel producer. Tied to US auto industry, construction activity, and domestic infrastructure spending."},
    "X":    {"name": "U.S. Steel",         "metal": "Steel",           "desc": "Integrated steel producer. Sensitive to US manufacturing cycles, tariff policy, and construction sector health."},
    "NUE":  {"name": "Nucor Corp",         "metal": "Steel",           "desc": "Largest US steel producer by volume. Uses electric arc furnace technology — more energy efficient and flexible than blast furnace peers."},
    "SCCO": {"name": "Southern Copper",    "metal": "Copper",          "desc": "Low-cost copper and molybdenum producer in Mexico and Peru. Among the most profitable copper miners globally."},
    "GOLD": {"name": "Barrick Gold",       "metal": "Gold Mining",     "desc": "Second largest gold miner globally. Key assets in Nevada (US) and Kibali (DRC). Tracks gold spot price closely."},
}


@st.cache_data(ttl=300)
def get_quick_prices():
    results = {}
    for ticker in TICKER_META.keys():
        try:
            df = yf.download(ticker, period="2d", auto_adjust=False, progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close = df["Close"].squeeze()
            latest = float(close.iloc[-1])
            prev   = float(close.iloc[-2]) if len(close) > 1 else latest
            chg    = (latest - prev) / prev * 100
            results[ticker] = {"price": latest, "change": chg}
        except Exception:
            pass
    return results


@st.cache_data(ttl=1800)
def get_ticker_news(symbol, max_items=6):
    try:
        t    = yf.Ticker(symbol)
        news = t.news
        if news:
            return news[:max_items]
        return []
    except Exception:
        return []


@st.cache_data(ttl=1800)
def get_metals_news():
    all_news = []
    for ticker in ["FCX", "NEM", "AA", "GLD", "COPX", "SLX"]:
        try:
            t    = yf.Ticker(ticker)
            news = t.news
            if news:
                all_news.extend(news[:3])
        except Exception:
            pass
    seen   = set()
    unique = []
    for item in all_news:
        title = item.get("title", "")
        if title not in seen:
            seen.add(title)
            unique.append(item)
    return unique[:12]


def render_news_card(item):
    title     = item.get("title", "No title")
    publisher = item.get("publisher", "Unknown")
    link      = item.get("link", "#")
    ts        = item.get("providerPublishTime", 0)
    date_str  = datetime.fromtimestamp(ts).strftime("%b %d, %Y") if ts else ""
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(163,139,92,0.2);
                border-radius:12px;padding:16px 18px;margin:0.5rem 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);">
        <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:#a38b5c;margin-bottom:0.4rem;">
            {publisher} &nbsp;·&nbsp; {date_str}
        </div>
        <a href="{link}" target="_blank"
           style="font-size:0.94rem;font-weight:600;color:#1a1a1a;
                  text-decoration:none;line-height:1.5;">
            {title}
        </a>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if nav == "Dashboard":

    render_hero(
        "Metals Intelligence Platform",
        "Metals Intelligence Dashboard",
        "An institutional-grade decision-support system for metals sector equities — "
        "integrating live market data, machine learning prediction models, financial "
        "integrity screening, and macroeconomic analysis into a single unified platform.",
    )

    divider()

    # ── Live Ticker Grid
    render_section(
        "Metals Universe — 8 Covered Equities",
        "Live prices and key descriptions for every stock covered by this platform. "
        "Prices refresh every 5 minutes. Click any tab above to analyse a specific stock in depth.",
    )

    with st.spinner("Loading live prices..."):
        prices = get_quick_prices()

    rows = [list(TICKER_META.keys())[i:i+4] for i in range(0, 8, 4)]
    for row in rows:
        cols = st.columns(4)
        for col, ticker in zip(cols, row):
            meta   = TICKER_META[ticker]
            p_data = prices.get(ticker, {})
            price  = p_data.get("price", None)
            change = p_data.get("change", None)
            price_str  = f"${price:,.2f}" if price else "—"
            change_str = f"{change:+.2f}%" if change is not None else ""
            chg_color  = "#27ae60" if (change or 0) >= 0 else "#e74c3c"
            with col:
                st.markdown(f"""
                <div style="background:#ffffff;border:1px solid rgba(163,139,92,0.22);
                            border-radius:16px;padding:18px 20px;
                            box-shadow:0 2px 10px rgba(0,0,0,0.04);margin-bottom:1rem;">
                    <div style="font-size:0.68rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:0.1em;color:#a38b5c;margin-bottom:0.3rem;">
                        {ticker} — {meta['metal']}
                    </div>
                    <div style="font-size:1.05rem;font-weight:700;color:#1a1a1a;
                                margin-bottom:0.25rem;">{meta['name']}</div>
                    <div style="display:flex;align-items:baseline;gap:0.5rem;margin-bottom:0.6rem;">
                        <span style="font-size:1.5rem;font-weight:700;color:#1a1a1a;
                                     font-family:'DM Serif Display',serif;">{price_str}</span>
                        <span style="font-size:0.85rem;font-weight:600;color:{chg_color};">
                            {change_str}
                        </span>
                    </div>
                    <div style="font-size:0.8rem;color:#6b6560;line-height:1.55;">
                        {meta['desc']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    divider()

    # ── Module Cards
    render_section(
        "Platform Modules",
        "Navigate using the tab bar above. Each module is independently powered "
        "but shares a common data and feature layer.",
    )

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Market Analytics</div>
            <div class="nav-card-title">Overview</div>
            <div class="nav-card-text">Live candlestick price chart with volume bars,
            Bollinger Bands, moving averages (MA10/20/50), RSI momentum indicator,
            and MACD trend signal. Includes period selector and CSV download.</div>
        </div>""", unsafe_allow_html=True)
    with r1c2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Predictive Intelligence</div>
            <div class="nav-card-title">Direction Prediction</div>
            <div class="nav-card-text">Short-horizon price direction signal (Up / Down)
            powered by a Random Forest classifier trained on 2 years of technical features.
            Shows prediction probability, model accuracy, and feature importance.</div>
        </div>""", unsafe_allow_html=True)
    with r1c3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Risk Analytics</div>
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
            <div class="nav-card-tag">Financial Integrity</div>
            <div class="nav-card-title">Fraud Detection</div>
            <div class="nav-card-text">Beneish-style financial statement quality screening
            using gross margin, asset quality, leverage, and profitability ratios to flag
            potential earnings manipulation risk.</div>
        </div>""", unsafe_allow_html=True)
    with r2c2:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Investment Intelligence</div>
            <div class="nav-card-title">Recommendation</div>
            <div class="nav-card-text">Hybrid MetalScore (0–100) aggregating direction
            signals, momentum, volatility penalty, and accounting quality into a final
            Buy / Hold / Avoid recommendation with signal breakdown.</div>
        </div>""", unsafe_allow_html=True)
    with r2c3:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Position Risk Management</div>
            <div class="nav-card-title">Stop-Loss Assistant</div>
            <div class="nav-card-text">Personalised stop-loss price and position sizing
            calculator calibrated to the user's risk tolerance, investment amount,
            and the stock's current ATR-based volatility profile.</div>
        </div>""", unsafe_allow_html=True)
    with r2c4:
        st.markdown("""
        <div class="nav-card">
            <div class="nav-card-tag">Global Markets & Economics</div>
            <div class="nav-card-title">Macro & News</div>
            <div class="nav-card-text">Gold, silver, copper, and crude oil price trends
            alongside treasury yields, VIX, US Dollar Index, live metals news,
            and macro/micro economic analysis driving metals markets.</div>
        </div>""", unsafe_allow_html=True)

    divider()

    # ── Latest News on Dashboard
    render_section(
        "Latest Metals Market News",
        "Live headlines from across the metals sector. Updated every 30 minutes.",
    )

    with st.spinner("Loading latest news..."):
        dash_news = get_metals_news()

    if dash_news:
        nc1, nc2 = st.columns(2)
        for i, item in enumerate(dash_news[:8]):
            with (nc1 if i % 2 == 0 else nc2):
                render_news_card(item)
    else:
        st.info("News feed temporarily unavailable.")

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
                demand, geopolitical risk, and monetary policy.<br><br>
                Gold miners hedge against inflation and dollar weakness. Copper producers
                track global industrial growth. Steel companies reflect domestic
                construction and manufacturing cycles.<br><br>
                This platform gives investors access to the same analytical
                framework used by professional metals-sector analysts.
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Overview":

    render_hero(
        "Market Analytics",
        "Market Overview",
        "Live price action, volume analysis, and technical indicators for the selected "
        "metals stock. All charts update automatically with the latest available market data.",
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels)
    with col_sel2:
        period = st.selectbox(
            "Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=2,
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
    meta         = TICKER_META.get(ticker, {})

    if meta:
        render_info(
            f"<strong>{meta.get('name', ticker)} ({ticker})</strong> — {meta.get('metal', '')}. "
            f"{meta.get('desc', '')}"
        )

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

    # ── Stock-specific news
    render_section(
        f"Latest News — {ticker}",
        f"Recent headlines directly related to {ticker} and the {meta.get('metal', 'metals')} sector.",
    )
    with st.spinner("Loading news..."):
        stock_news = get_ticker_news(ticker, max_items=6)
    if stock_news:
        nc1, nc2 = st.columns(2)
        for i, item in enumerate(stock_news):
            with (nc1 if i % 2 == 0 else nc2):
                render_news_card(item)
    else:
        st.info("No recent news found for this ticker.")

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
        "Predictive Intelligence",
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
    meta   = TICKER_META.get(ticker, {})
    if meta:
        render_info(f"<strong>{meta.get('name', ticker)} ({ticker})</strong> — {meta.get('desc', '')}")

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
        "Values above 60% indicate a stronger signal. Values near 50% suggest high uncertainty.",
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
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#6b6560", "tickfont": {"size": 11}},
                "bar": {"color": sig_color, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1, "bordercolor": "rgba(163,139,92,0.2)",
                "steps": [
                    {"range": [0,  30],  "color": "rgba(231,76,60,0.12)"},
                    {"range": [30, 50],  "color": "rgba(231,76,60,0.05)"},
                    {"range": [50, 70],  "color": "rgba(39,174,96,0.05)"},
                    {"range": [70, 100], "color": "rgba(39,174,96,0.12)"},
                ],
                "threshold": {"line": {"color": "#a38b5c", "width": 3},
                              "thickness": 0.75, "value": 50},
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#faf6f0", font=dict(family="DM Sans"),
            height=300, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_g2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=["DOWN", "UP"],
            y=[prob_down * 100, prob_up * 100],
            marker_color=["#e74c3c", "#27ae60"], marker_opacity=0.85,
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
        "Higher importance = the model relied more heavily on that signal.",
    )

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = np.abs(model.coef_[0])

    feat_df = pd.DataFrame({
        "Feature": feature_cols, "Importance": importances,
    }).sort_values("Importance", ascending=True)

    FEATURE_LABELS = {
        "Return_1d": "1-Day Return", "Return_5d": "5-Day Return",
        "Return_10d": "10-Day Return", "Return_20d": "20-Day Return",
        "MACD": "MACD", "MACD_Signal": "MACD Signal Line",
        "MACD_Hist": "MACD Histogram", "RSI": "RSI (14)",
        "BB_Width": "Bollinger Band Width", "BB_Pct": "BB Position (%)",
        "ATR_Pct": "ATR (Normalised)", "Volatility_20d": "20-Day Volatility",
        "Volume_Ratio": "Volume Ratio", "Momentum_10": "10-Day Momentum",
        "Momentum_20": "20-Day Momentum",
    }
    feat_df["Label"] = feat_df["Feature"].map(FEATURE_LABELS).fillna(feat_df["Feature"])
    bar_colors = [
        "#27ae60" if v >= feat_df["Importance"].median() else "#a38b5c"
        for v in feat_df["Importance"]
    ]
    fig_feat = go.Figure(go.Bar(
        x=feat_df["Importance"], y=feat_df["Label"], orientation="h",
        marker_color=bar_colors, marker_opacity=0.85,
        text=[f"{v:.3f}" for v in feat_df["Importance"]],
        textposition="outside", textfont=dict(size=10, color="#6b6560"),
    ))
    fig_feat.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=460, margin=dict(l=10, r=60, t=20, b=10),
        xaxis=dict(title="Importance Score", gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        yaxis=dict(tickfont=dict(size=11, color="#1a1a1a")),
        showlegend=False,
    )
    st.plotly_chart(fig_feat, use_container_width=True)

    divider()

    render_section(
        "Recent Prediction vs Actual — Signal History",
        "Green markers = correct predictions. Red markers = incorrect predictions.",
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
        x=correct.index, y=correct["Close"], mode="markers",
        marker=dict(color="#27ae60", size=8, symbol="circle",
                    line=dict(color="white", width=1)),
        name="Correct Prediction",
    ))
    fig_hist.add_trace(go.Scatter(
        x=incorrect.index, y=incorrect["Close"], mode="markers",
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
            "Recall = how often all actual instances found. "
            "F1 = harmonic mean of precision and recall."
        )
        st.code(classification_report(y_test, y_pred,
                                      target_names=["DOWN (0)", "UP (1)"]))

    render_info(
        "<strong>Disclaimer:</strong> This model is for analytical purposes only. "
        "No model can consistently predict short-term market movements. "
        "Always combine signals with fundamental analysis and your own judgment."
    )


# ══════════════════════════════════════════════════════════════════════════════
# RISK
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Risk":

    render_hero(
        "Risk Analytics",
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
            "Time Period", ["6mo", "1y", "2y"], index=1, key="risk_period",
        )

    ticker = get_ticker_symbol(selected_label)
    meta   = TICKER_META.get(ticker, {})
    if meta:
        render_info(f"<strong>{meta.get('name', ticker)} ({ticker})</strong> — {meta.get('desc', '')}")

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
    k1.metric("Ticker",         ticker)
    k2.metric("20d Volatility", f"{vol_20*100:.1f}%",  "Annualised")
    k3.metric("60d Volatility", f"{vol_60*100:.1f}%",  "Annualised")
    k4.metric("Risk Level",     risk_level)
    k5.metric("VaR (95%)",      f"{var_95*100:.2f}%",  "Daily worst-case")
    k6.metric("Max Drawdown",   f"{max_dd*100:.1f}%",  f"Current: {curr_dd*100:.1f}%")

    risk_fn(
        f"<strong>{ticker} Risk Level: {risk_level}.</strong> "
        f"20-day annualised volatility: <strong>{vol_20*100:.1f}%</strong>. "
        f"At 95% confidence, max expected daily loss (VaR): <strong>{abs(var_95)*100:.2f}%</strong>. "
        f"Maximum historical drawdown: <strong>{abs(max_dd)*100:.1f}%</strong>. "
        f"Annualised Sharpe ratio: <strong>{sharpe:.2f}</strong> — "
        f"{'above 1.0 is generally considered good.' if sharpe > 1 else 'below 1.0 suggests risk may not be fully compensated by returns.'}"
    )

    divider()

    render_section(
        "Rolling Volatility — 20-Day vs 60-Day Annualised",
        "The 20-day line reacts quickly to recent events. "
        "The 60-day line reflects medium-term risk. "
        "When the 20-day spikes above the 60-day it signals sudden increased risk.",
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
                      annotation_text="Low Risk (25%)", annotation_position="bottom right",
                      annotation_font=dict(size=10, color="#27ae60"))
    fig_vol.add_hline(y=45, line=dict(color="#e74c3c", width=1, dash="dot"),
                      annotation_text="High Risk (45%)", annotation_position="bottom right",
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
        "Drawdown = how far the stock has fallen from its most recent peak. "
        "Maximum drawdown is the largest peak-to-trough decline — a key risk management metric.",
    )

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown.values * 100,
        fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="#e74c3c", width=1.5), name="Drawdown",
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
        yaxis=dict(title="Drawdown (%)", gridcolor="rgba(163,139,92,0.10)",
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
        "VaR 95%: 5% chance of losing more than this on any given day. "
        "VaR 99%: only 1% chance of exceeding this loss.",
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns * 100, nbinsx=60,
        marker_color="#a38b5c", marker_opacity=0.75, name="Daily Returns",
    ))
    fig_dist.add_vline(x=var_95 * 100, line=dict(color="#e74c3c", width=2, dash="dash"),
                       annotation_text=f"VaR 95%: {var_95*100:.2f}%",
                       annotation_position="top left",
                       annotation_font=dict(size=11, color="#e74c3c"))
    fig_dist.add_vline(x=var_99 * 100, line=dict(color="#7d3c98", width=2, dash="dash"),
                       annotation_text=f"VaR 99%: {var_99*100:.2f}%",
                       annotation_position="top left",
                       annotation_font=dict(size=11, color="#7d3c98"))
    fig_dist.add_vline(x=avg_ret * 100, line=dict(color="#27ae60", width=1.5, dash="dot"),
                       annotation_text=f"Avg: {avg_ret*100:.3f}%",
                       annotation_position="top right",
                       annotation_font=dict(size=11, color="#27ae60"))
    fig_dist.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(title="Daily Return (%)", gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        yaxis=dict(title="Frequency", gridcolor="rgba(163,139,92,0.10)",
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
# FRAUD
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Fraud":

    render_hero(
        "Financial Integrity Screening",
        "Fraud & Manipulation Detection",
        "A Beneish-style financial statement quality screening tool that analyses key "
        "accounting ratios to detect potential earnings manipulation or financial statement "
        "irregularities in metals sector companies.",
    )

    selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="fraud_ticker")
    ticker = get_ticker_symbol(selected_label)
    meta   = TICKER_META.get(ticker, {})
    if meta:
        render_info(f"<strong>{meta.get('name', ticker)} ({ticker})</strong> — {meta.get('desc', '')}")

    with st.spinner(f"Fetching financial statement data for {ticker}..."):
        scores = get_beneish_scores(ticker)

    if "Error" in scores:
        st.error(f"Could not retrieve financial data: {scores['Error']}")
        st.stop()

    overall = scores.get("Overall Risk", "Unknown")
    flags   = scores.get("Flags Triggered", "0 / 0")

    if "High" in overall:
        render_danger(
            f"<strong>{ticker} — {overall}.</strong> "
            f"{flags} accounting quality flags were triggered. "
            f"This suggests potential financial statement manipulation or significant "
            f"accounting irregularities. Investors should conduct deeper due diligence."
        )
    elif "Moderate" in overall:
        render_alert(
            f"<strong>{ticker} — {overall}.</strong> "
            f"{flags} accounting quality flags were triggered. "
            f"Some indicators suggest caution. Monitor closely around earnings announcements."
        )
    else:
        render_success(
            f"<strong>{ticker} — {overall}.</strong> "
            f"{flags} accounting quality flags were triggered. "
            f"Financial statement quality appears acceptable."
        )

    divider()

    render_section(
        "Financial Ratio Analysis — Beneish-Style Screening",
        "Each ratio measures a different dimension of financial statement quality. "
        "Warning flags indicate ratios outside normal ranges that may suggest aggressive accounting.",
    )

    ratio_keys = [
        "Gross Margin (%)", "Asset Quality Index", "Leverage Ratio",
        "Net Profit Margin (%)", "Current Ratio", "FCF Margin (%)",
    ]
    flag_keys = ["GM Flag", "AQI Flag", "LEV Flag", "NPM Flag", "CR Flag", "FCF Flag"]
    descriptions = {
        "Gross Margin (%)":      "Revenue minus COGS as % of revenue. Low margins (<20%) may signal pricing pressure.",
        "Asset Quality Index":   "Non-current assets as proportion of total assets. High values (>0.75) suggest hard-to-value assets.",
        "Leverage Ratio":        "Total debt as proportion of total assets. High leverage (>0.60) increases default risk.",
        "Net Profit Margin (%)": "Net income as % of revenue. Very low margins (<3%) may signal profitability problems.",
        "Current Ratio":         "Current assets / current liabilities. Below 1.0 = potential short-term liquidity risk.",
        "FCF Margin (%)":        "Free cash flow as % of revenue. Negative FCF signals cash consumed faster than generated.",
    }

    col1, col2 = st.columns(2)
    for i, (ratio, flag) in enumerate(zip(ratio_keys, flag_keys)):
        value    = scores.get(ratio, "N/A")
        flag_val = scores.get(flag, "Normal")
        desc     = descriptions.get(ratio, "")
        is_warn  = flag_val == "Warning"
        col = col1 if i % 2 == 0 else col2
        with col:
            color  = "#ffebee" if is_warn else "#e8f5e9"
            border = "#ef9a9a" if is_warn else "#a5d6a7"
            label  = "#b71c1c" if is_warn else "#1b5e20"
            icon   = "⚠" if is_warn else "✓"
            st.markdown(f"""
            <div style="background:{color};border:1px solid {border};
                        border-radius:12px;padding:16px 18px;margin:0.5rem 0;">
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:6px;">
                    <span style="font-weight:700;font-size:0.9rem;color:#1a1a1a;">{ratio}</span>
                    <span style="font-weight:700;font-size:0.85rem;color:{label};">{icon} {flag_val}</span>
                </div>
                <div style="font-size:1.4rem;font-weight:700;color:#1a1a1a;
                            font-family:'DM Serif Display',serif;margin-bottom:6px;">{value}</div>
                <div style="font-size:0.82rem;color:#5a5550;line-height:1.55;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    render_section("Risk Score Visual")
    ratio_colors_fraud = [
        "#e74c3c" if scores.get(f, "Normal") == "Warning" else "#27ae60"
        for f in flag_keys
    ]
    fig_fraud = go.Figure(go.Bar(
        x=ratio_keys, y=[1] * len(ratio_keys),
        marker_color=ratio_colors_fraud, marker_opacity=0.85,
        text=[scores.get(r, "N/A") for r in ratio_keys],
        textposition="inside",
        textfont=dict(size=11, color="white", family="DM Sans"),
    ))
    fig_fraud.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=280, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(tickfont=dict(size=10, color="#1a1a1a"),
                   gridcolor="rgba(163,139,92,0.10)"),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    st.plotly_chart(fig_fraud, use_container_width=True)

    render_info(
        "<strong>About the Beneish M-Score:</strong> Developed by Professor Messod Beneish, "
        "the M-Score uses financial statement ratios to identify companies that may have manipulated earnings. "
        "This is a screening tool — use it alongside fundamental analysis, not as a definitive verdict."
    )


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Recommendation":

    render_hero(
        "Investment Intelligence",
        "Recommendation Engine",
        "A hybrid MetalScore (0–100) that aggregates signals from price direction, "
        "momentum, volatility risk, and financial statement quality into a single "
        "Buy / Hold / Avoid recommendation with full signal attribution.",
    )

    selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="rec_ticker")
    ticker = get_ticker_symbol(selected_label)
    meta   = TICKER_META.get(ticker, {})
    if meta:
        render_info(f"<strong>{meta.get('name', ticker)} ({ticker})</strong> — {meta.get('desc', '')}")

    with st.spinner(f"Computing MetalScore for {ticker}..."):
        df_raw = load_price_data(ticker, "1y")

    if df_raw.empty:
        st.error("No data returned. Try another ticker.")
        st.stop()

    df = add_technical_indicators(df_raw)
    returns     = df["Return_1d"].dropna()
    latest_rsi  = float(df["RSI"].dropna().iloc[-1])
    latest_macd = float(df["MACD"].dropna().iloc[-1])
    macd_sig    = float(df["MACD_Signal"].dropna().iloc[-1])
    vol_20      = float(df["Volatility_20d"].dropna().iloc[-1])
    momentum_20 = float(df["Momentum_20"].dropna().iloc[-1])
    bb_pct      = float(df["BB_Pct"].dropna().iloc[-1])
    drawdown    = compute_drawdown(df)
    avg_ret     = float(returns.mean())

    if latest_rsi < 30:
        rsi_score = 75
    elif latest_rsi > 70:
        rsi_score = 25
    else:
        rsi_score = 50 + (50 - latest_rsi) * 0.5

    macd_score = 70 if latest_macd > macd_sig else 30

    if momentum_20 > 0.10:
        mom_score = 80
    elif momentum_20 > 0:
        mom_score = 60
    elif momentum_20 > -0.10:
        mom_score = 40
    else:
        mom_score = 20

    if vol_20 < 0.25:
        vol_score = 80
    elif vol_20 < 0.45:
        vol_score = 55
    else:
        vol_score = 25

    bb_score  = 75 if bb_pct < 0.2 else 35 if bb_pct > 0.8 else 65
    ret_score = min(max(50 + avg_ret * 10000, 10), 90)

    metal_score = round(min(max(
        rsi_score  * 0.20 +
        macd_score * 0.20 +
        mom_score  * 0.25 +
        vol_score  * 0.15 +
        bb_score   * 0.10 +
        ret_score  * 0.10, 0), 100), 1)

    if metal_score >= 65:
        recommendation = "BUY"
        rec_color      = "#27ae60"
        rec_fn         = render_success
    elif metal_score >= 45:
        recommendation = "HOLD"
        rec_color      = "#f9a825"
        rec_fn         = render_alert
    else:
        recommendation = "AVOID"
        rec_color      = "#e74c3c"
        rec_fn         = render_danger

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ticker",         ticker)
    k2.metric("MetalScore",     f"{metal_score} / 100")
    k3.metric("Recommendation", recommendation)
    k4.metric("Momentum (20d)", f"{momentum_20*100:+.1f}%")

    rec_fn(
        f"<strong>{ticker} — {recommendation} (MetalScore: {metal_score}/100)</strong><br>"
        f"RSI: <strong>{latest_rsi:.1f}</strong> — "
        f"{'overbought, caution advised' if latest_rsi > 70 else 'oversold, potential opportunity' if latest_rsi < 30 else 'neutral'}. "
        f"MACD is <strong>{'bullish' if latest_macd > macd_sig else 'bearish'}</strong>. "
        f"20-day momentum: <strong>{momentum_20*100:+.1f}%</strong>. "
        f"20-day annualised volatility: <strong>{vol_20*100:.1f}%</strong>."
    )

    divider()

    render_section("MetalScore Gauge & Signal Breakdown")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        fig_score = go.Figure(go.Indicator(
            mode="gauge+number",
            value=metal_score,
            number={"font": {"size": 52, "color": rec_color,
                             "family": "DM Serif Display"}},
            title={"text": f"MetalScore — {recommendation}",
                   "font": {"size": 14, "color": "#6b6560", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1,
                         "tickcolor": "#6b6560", "tickfont": {"size": 11}},
                "bar": {"color": rec_color, "thickness": 0.3},
                "bgcolor": "#ffffff",
                "borderwidth": 1, "bordercolor": "rgba(163,139,92,0.2)",
                "steps": [
                    {"range": [0,  45], "color": "rgba(231,76,60,0.12)"},
                    {"range": [45, 65], "color": "rgba(249,168,37,0.12)"},
                    {"range": [65, 100],"color": "rgba(39,174,96,0.12)"},
                ],
                "threshold": {"line": {"color": "#a38b5c", "width": 3},
                              "thickness": 0.75, "value": metal_score},
            },
        ))
        fig_score.update_layout(
            paper_bgcolor="#faf6f0", font=dict(family="DM Sans"),
            height=320, margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_score, use_container_width=True)

    with col_g2:
        signal_names   = ["RSI Score", "MACD Score", "Momentum", "Volatility", "BB Position", "Return Quality"]
        signal_scores  = [rsi_score, macd_score, mom_score, vol_score, bb_score, ret_score]
        signal_weights = [0.20, 0.20, 0.25, 0.15, 0.10, 0.10]
        bar_cols = ["#27ae60" if s >= 60 else "#f9a825" if s >= 40 else "#e74c3c"
                    for s in signal_scores]
        fig_signals = go.Figure(go.Bar(
            x=signal_scores, y=signal_names, orientation="h",
            marker_color=bar_cols, marker_opacity=0.85,
            text=[f"{s:.0f} (wt: {w:.0%})" for s, w in zip(signal_scores, signal_weights)],
            textposition="outside", textfont=dict(size=10, color="#6b6560"),
        ))
        fig_signals.add_vline(x=50, line=dict(color="#a38b5c", width=1.5, dash="dash"))
        fig_signals.update_layout(
            paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
            font=dict(family="DM Sans", color="#1a1a1a", size=12),
            height=320, margin=dict(l=10, r=80, t=20, b=10),
            title=dict(text="Signal Breakdown",
                       font=dict(size=13, color="#1a1a1a"), x=0),
            xaxis=dict(range=[0, 120], title="Score (0–100)",
                       gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            yaxis=dict(tickfont=dict(size=11, color="#1a1a1a")),
            showlegend=False,
        )
        st.plotly_chart(fig_signals, use_container_width=True)

    divider()

    render_section(
        "Sector Comparison — Relative Performance",
        "All metals stocks rebased to 100. Lines above 100 = positive performance since period start.",
    )

    with st.spinner("Loading sector comparison..."):
        sector_df = load_sector_normalised("6mo")

    if not sector_df.empty:
        fig_sector = go.Figure()
        colors_sector = ["#a38b5c", "#2471a3", "#27ae60", "#e74c3c",
                         "#7d3c98", "#d35400", "#1a9c8a", "#c0392b"]
        for i, col in enumerate(sector_df.columns):
            lw = 2.5 if col == ticker else 1.2
            op = 1.0 if col == ticker else 0.55
            fig_sector.add_trace(go.Scatter(
                x=sector_df.index, y=sector_df[col],
                line=dict(color=colors_sector[i % len(colors_sector)], width=lw),
                opacity=op, name=col,
            ))
        fig_sector.add_hline(y=100, line=dict(color="#6b6560", width=1, dash="dot"))
        fig_sector.update_layout(
            paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
            font=dict(family="DM Sans", color="#1a1a1a", size=12),
            height=420, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(bgcolor="rgba(255,255,255,0.88)",
                        bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                        orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=11)),
            yaxis=dict(title="Indexed Performance (Base = 100)",
                       gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                            font=dict(family="DM Sans", size=12, color="#1a1a1a")),
        )
        st.plotly_chart(fig_sector, use_container_width=True)

    render_info(
        "<strong>Disclaimer:</strong> The MetalScore and recommendation are generated by a "
        "quantitative model using technical indicators only. They do not constitute financial advice."
    )


# ══════════════════════════════════════════════════════════════════════════════
# STOP-LOSS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Stop-Loss":

    render_hero(
        "Position Risk Management",
        "Stop-Loss Assistant",
        "A personalised stop-loss price and position sizing calculator that calibrates "
        "exit levels based on your risk tolerance, investment amount, and the stock's "
        "current volatility profile. Uses ATR-based volatility to set realistic stop levels.",
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="sl_ticker")
    with col_sel2:
        risk_profile = st.selectbox(
            "Risk Profile", ["Conservative", "Moderate", "Aggressive"],
            index=1, key="sl_risk",
        )

    ticker = get_ticker_symbol(selected_label)
    meta   = TICKER_META.get(ticker, {})
    if meta:
        render_info(f"<strong>{meta.get('name', ticker)} ({ticker})</strong> — {meta.get('desc', '')}")

    df_raw = load_price_data(ticker, "3mo")
    if df_raw.empty:
        st.error("No data returned. Try another ticker.")
        st.stop()

    df           = add_technical_indicators(df_raw)
    latest_price = float(df["Close"].squeeze().iloc[-1])
    atr          = float(df["ATR"].dropna().iloc[-1])
    vol_20       = float(df["Volatility_20d"].dropna().iloc[-1])

    divider()

    render_section("Position Parameters")

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        investment = st.number_input(
            "Investment Amount ($)", min_value=100, max_value=1000000,
            value=10000, step=500,
        )
    with col_i2:
        max_loss_pct = st.slider("Maximum Acceptable Loss (%)", min_value=1, max_value=30, value=10)
    with col_i3:
        st.metric("Current Price", f"${latest_price:,.2f}")
        st.metric("ATR (14-Day)",  f"${atr:,.2f}")

    atr_mult     = {"Conservative": 1.5, "Moderate": 2.0, "Aggressive": 3.0}[risk_profile]
    pct_mult     = {"Conservative": 0.5, "Moderate": 1.0, "Aggressive": 1.5}[risk_profile]
    stop_atr     = latest_price - (atr * atr_mult)
    stop_pct     = latest_price * (1 - (max_loss_pct / 100) * pct_mult)
    stop_final   = max(stop_atr, stop_pct)
    stop_dist    = latest_price - stop_final
    stop_dist_pct= stop_dist / latest_price * 100
    max_loss_usd = investment * (max_loss_pct / 100)
    shares       = int(max_loss_usd / stop_dist) if stop_dist > 0 else 0
    position_val = shares * latest_price
    take_profit  = latest_price + (stop_dist * 2)

    divider()

    render_section(
        "Your Stop-Loss Recommendation",
        f"Based on {risk_profile} risk profile, ATR of ${atr:.2f}, and "
        f"maximum acceptable loss of {max_loss_pct}% (${max_loss_usd:,.0f}).",
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Entry Price",       f"${latest_price:,.2f}")
    k2.metric("Stop-Loss Price",   f"${stop_final:,.2f}",   f"-{stop_dist_pct:.1f}%")
    k3.metric("Take-Profit (2:1)", f"${take_profit:,.2f}",  f"+{stop_dist_pct*2:.1f}%")
    k4.metric("Suggested Shares",  f"{shares:,}",           f"${position_val:,.0f} position")
    k5.metric("Max Loss (USD)",    f"${max_loss_usd:,.0f}",  f"{max_loss_pct}% of investment")

    if stop_dist_pct < 3:
        render_alert(
            f"<strong>Stop-loss is very tight ({stop_dist_pct:.1f}% below entry).</strong> "
            f"Normal daily price fluctuations for {ticker} (ATR: ${atr:.2f}) could trigger "
            f"this stop prematurely. Consider widening your stop or reducing position size."
        )
    else:
        render_info(
            f"<strong>Stop-loss set at ${stop_final:,.2f} ({stop_dist_pct:.1f}% below entry).</strong> "
            f"Based on {atr_mult}x the 14-day ATR of ${atr:.2f}. "
            f"Take-profit target: ${take_profit:,.2f}. "
            f"Suggested position: <strong>{shares} shares</strong> at ${position_val:,.0f} total."
        )

    divider()

    render_section("Price Levels Chart")
    price_series = df["Close"].squeeze()
    fig_sl = go.Figure()
    fig_sl.add_trace(go.Scatter(
        x=price_series.index, y=price_series,
        line=dict(color="#a38b5c", width=2), name="Price",
    ))
    fig_sl.add_hline(y=latest_price,
                     line=dict(color="#2471a3", width=2, dash="dash"),
                     annotation_text=f"Entry: ${latest_price:,.2f}",
                     annotation_position="right",
                     annotation_font=dict(size=11, color="#2471a3"))
    fig_sl.add_hline(y=stop_final,
                     line=dict(color="#e74c3c", width=2, dash="dash"),
                     annotation_text=f"Stop: ${stop_final:,.2f}",
                     annotation_position="right",
                     annotation_font=dict(size=11, color="#e74c3c"))
    fig_sl.add_hline(y=take_profit,
                     line=dict(color="#27ae60", width=2, dash="dash"),
                     annotation_text=f"Target: ${take_profit:,.2f}",
                     annotation_position="right",
                     annotation_font=dict(size=11, color="#27ae60"))
    fig_sl.add_hrect(y0=stop_final, y1=latest_price,
                     fillcolor="rgba(231,76,60,0.07)", line_width=0)
    fig_sl.add_hrect(y0=latest_price, y1=take_profit,
                     fillcolor="rgba(39,174,96,0.07)", line_width=0)
    fig_sl.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=460, margin=dict(l=10, r=130, t=20, b=10),
        yaxis=dict(title="Price (USD)", gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                        font=dict(family="DM Sans", size=12, color="#1a1a1a")),
        showlegend=True,
        legend=dict(bgcolor="rgba(255,255,255,0.88)",
                    bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                    font=dict(size=11)),
    )
    st.plotly_chart(fig_sl, use_container_width=True)

    divider()

    render_section("Risk Summary")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Position Details</strong>", unsafe_allow_html=True)
        for label, value in {
            "Risk Profile":          risk_profile,
            "Entry Price":           f"${latest_price:,.2f}",
            "Stop-Loss Price":       f"${stop_final:,.2f}",
            "Stop Distance":         f"${stop_dist:,.2f} ({stop_dist_pct:.1f}%)",
            "Take-Profit (2:1 R:R)": f"${take_profit:,.2f}",
            "ATR Multiplier Used":   f"{atr_mult}x",
        }.items():
            st.markdown(
                f'<div class="stat-row"><span class="stat-label">{label}</span>'
                f'<span class="stat-value">{value}</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with col_s2:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Risk & Sizing</strong>", unsafe_allow_html=True)
        for label, value in {
            "Investment Amount":     f"${investment:,.0f}",
            "Max Acceptable Loss %": f"{max_loss_pct}%",
            "Max Loss (USD)":        f"${max_loss_usd:,.0f}",
            "Suggested Shares":      f"{shares:,}",
            "Position Value":        f"${position_val:,.0f}",
            "20d Annualised Vol":    f"{vol_20*100:.1f}%",
        }.items():
            st.markdown(
                f'<div class="stat-row"><span class="stat-label">{label}</span>'
                f'<span class="stat-value">{value}</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    render_info(
        "<strong>Important:</strong> Stop-loss levels are suggestions based on quantitative "
        "volatility analysis. They do not guarantee against losses exceeding the stated amount "
        "during gap openings, market halts, or periods of extreme illiquidity."
    )


# ══════════════════════════════════════════════════════════════════════════════
# MACRO & NEWS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Macro & News":

    render_hero(
        "Global Markets & Economics",
        "Macro & News Intelligence",
        "The macroeconomic forces, commodity trends, and live news driving metals markets. "
        "Understanding macro context is critical — metals stocks are among the most "
        "macro-sensitive equities in global markets.",
    )

    period = st.selectbox(
        "Time Period", ["3mo", "6mo", "1y", "2y"], index=1, key="macro_period",
    )

    # ── Macro Context Explanations
    render_section(
        "Macro & Micro Economic Drivers — What Moves Metals Stocks",
        "A plain-language guide to the forces currently affecting metals sector equities.",
    )

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown("""
        <div class="soft-panel">
            <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#a38b5c;margin-bottom:0.75rem;">
                Macroeconomic Factors (Global)
            </div>
            <div style="font-size:0.88rem;color:#3a3530;line-height:1.8;">
                <strong>Interest Rates (Fed Policy)</strong><br>
                Rising rates = stronger USD = lower commodity prices (gold, copper priced in USD).
                Rate hikes also slow economic growth, reducing industrial metal demand.
                When the Fed cuts rates, metals typically rally.<br><br>
                <strong>US Dollar Index (DXY)</strong><br>
                Metals are priced in USD globally. A stronger dollar makes metals more expensive
                for foreign buyers, reducing demand and prices. Dollar weakness is bullish for
                all metals stocks — especially gold miners.<br><br>
                <strong>China Economic Growth</strong><br>
                China consumes ~50% of global copper, steel, and aluminum. Any slowdown in
                Chinese manufacturing, construction, or infrastructure spending directly
                pressures FCX, SCCO, AA, CLF, X, and NUE.<br><br>
                <strong>Geopolitical Risk</strong><br>
                Wars, sanctions, and trade disputes drive gold higher as a safe-haven asset.
                Supply disruptions in mining regions (Peru, DRC, Chile) create price spikes
                for copper and gold regardless of broader market conditions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_m2:
        st.markdown("""
        <div class="soft-panel">
            <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.08em;color:#a38b5c;margin-bottom:0.75rem;">
                Microeconomic Factors (Company & Industry Level)
            </div>
            <div style="font-size:0.88rem;color:#3a3530;line-height:1.8;">
                <strong>Commodity Spot Prices</strong><br>
                The most direct driver. Copper at $4.50/lb vs $3.50/lb can double FCX's
                earnings. Gold miners' margins expand dramatically when gold rises above
                their all-in sustaining costs (AISC).<br><br>
                <strong>Energy Costs</strong><br>
                Aluminum smelting and mining operations are energy-intensive. High electricity
                and natural gas prices compress margins for AA, CLF, and NUE, even when
                metal prices are rising.<br><br>
                <strong>Earnings & Production Reports</strong><br>
                Quarterly production guidance, reserve estimates, cost guidance, and
                capex plans directly move individual stock prices. Earnings misses in
                metals can cause 10–20% single-day moves.<br><br>
                <strong>EV & Green Energy Demand</strong><br>
                Electric vehicles require 3–4x more copper than conventional cars.
                Solar panels need aluminum and silver. This structural demand growth
                is a long-term bullish catalyst for FCX, SCCO, and AA.
            </div>
        </div>
        """, unsafe_allow_html=True)

    divider()

    # ── Commodity Charts
    render_section(
        "Commodity Prices — Gold, Silver, Copper, Crude Oil & Natural Gas",
        "Rebased to 100 at period start for easy comparison. "
        "Rising commodity prices generally lift metals sector equities. "
        "Divergences between commodity prices and stock prices can signal mispricing opportunities.",
    )

    with st.spinner("Loading commodity data..."):
        macro_df = load_macro_data(period)

    if not macro_df.empty:
        commodity_cols = [c for c in macro_df.columns if any(
            x in c for x in ["Gold", "Silver", "Copper", "Crude", "Nat."])]

        if commodity_cols:
            fig_comm = go.Figure()
            comm_colors = ["#a38b5c", "#7d7d7d", "#d35400", "#2471a3", "#27ae60"]
            for i, col in enumerate(commodity_cols):
                series = macro_df[col].dropna()
                if series.empty:
                    continue
                rebased = series / series.iloc[0] * 100
                fig_comm.add_trace(go.Scatter(
                    x=rebased.index, y=rebased,
                    line=dict(color=comm_colors[i % len(comm_colors)], width=2),
                    name=col,
                ))
            fig_comm.add_hline(y=100, line=dict(color="#6b6560", width=1, dash="dot"),
                               annotation_text="Base (period start)",
                               annotation_position="right",
                               annotation_font=dict(size=10, color="#6b6560"))
            fig_comm.update_layout(
                paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
                font=dict(family="DM Sans", color="#1a1a1a", size=12),
                height=440, margin=dict(l=10, r=10, t=20, b=10),
                legend=dict(bgcolor="rgba(255,255,255,0.88)",
                            bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                            orientation="h", yanchor="bottom", y=1.01,
                            xanchor="left", x=0, font=dict(size=11)),
                yaxis=dict(title="Indexed Price (Base = 100)",
                           gridcolor="rgba(163,139,92,0.10)",
                           tickfont=dict(size=11, color="#6b6560")),
                xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                           tickfont=dict(size=11, color="#6b6560")),
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                                font=dict(family="DM Sans", size=12, color="#1a1a1a")),
            )
            st.plotly_chart(fig_comm, use_container_width=True)

        divider()

        render_section(
            "Market Risk Indicators — VIX, S&P 500, 10Y Treasury & US Dollar",
            "VIX above 30 = high market fear → typically bullish for gold. "
            "Rising 10Y Treasury yields → bearish for gold (higher opportunity cost of holding non-yielding assets). "
            "Rising US Dollar → bearish for commodity prices. "
            "S&P 500 weakness → mixed for metals (industrial metals fall, gold rises).",
        )

        macro_cols = [c for c in macro_df.columns if any(
            x in c for x in ["S&P", "VIX", "Treasury", "Dollar"])]

        if macro_cols:
            fig_macro = make_subplots(
                rows=len(macro_cols), cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=macro_cols,
            )
            macro_colors = ["#2471a3", "#e74c3c", "#a38b5c", "#27ae60"]
            for i, col in enumerate(macro_cols):
                series = macro_df[col].dropna()
                if series.empty:
                    continue
                try:
                    r = int(macro_colors[i % len(macro_colors)][1:3], 16)
                    g = int(macro_colors[i % len(macro_colors)][3:5], 16)
                    b = int(macro_colors[i % len(macro_colors)][5:7], 16)
                    fill_color = f"rgba({r},{g},{b},0.07)"
                except Exception:
                    fill_color = "rgba(163,139,92,0.07)"
                fig_macro.add_trace(go.Scatter(
                    x=series.index, y=series,
                    line=dict(color=macro_colors[i % len(macro_colors)], width=2),
                    fill="tozeroy", fillcolor=fill_color,
                    name=col,
                ), row=i+1, col=1)
                fig_macro.update_yaxes(
                    gridcolor="rgba(163,139,92,0.10)",
                    tickfont=dict(size=10, color="#6b6560"),
                    row=i+1, col=1,
                )
                fig_macro.update_xaxes(
                    gridcolor="rgba(163,139,92,0.10)",
                    tickfont=dict(size=10, color="#6b6560"),
                    row=i+1, col=1,
                )
            fig_macro.update_layout(
                paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
                font=dict(family="DM Sans", color="#1a1a1a", size=12),
                height=max(120 * len(macro_cols) + 60, 300),
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False,
                hovermode="x unified",
                hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                                font=dict(family="DM Sans", size=12, color="#1a1a1a")),
            )
            st.plotly_chart(fig_macro, use_container_width=True)

    divider()

    render_section(
        "Metals Sector — Relative Performance",
        "All 8 metals stocks rebased to 100. Identify sector leaders and laggards at a glance.",
    )

    with st.spinner("Loading sector data..."):
        sector_df = load_sector_normalised(period)

    if not sector_df.empty:
        fig_sec = go.Figure()
        colors_s = ["#a38b5c", "#2471a3", "#27ae60", "#e74c3c",
                    "#7d3c98", "#d35400", "#1a9c8a", "#c0392b"]
        for i, col in enumerate(sector_df.columns):
            fig_sec.add_trace(go.Scatter(
                x=sector_df.index, y=sector_df[col],
                line=dict(color=colors_s[i % len(colors_s)], width=1.8),
                name=col,
            ))
        fig_sec.add_hline(y=100, line=dict(color="#6b6560", width=1, dash="dot"))
        fig_sec.update_layout(
            paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
            font=dict(family="DM Sans", color="#1a1a1a", size=12),
            height=440, margin=dict(l=10, r=10, t=20, b=10),
            legend=dict(bgcolor="rgba(255,255,255,0.88)",
                        bordercolor="rgba(163,139,92,0.22)", borderwidth=1,
                        orientation="h", yanchor="bottom", y=1.01,
                        xanchor="left", x=0, font=dict(size=11)),
            yaxis=dict(title="Indexed Performance (Base = 100)",
                       gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            xaxis=dict(gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#ffffff", bordercolor="rgba(163,139,92,0.3)",
                            font=dict(family="DM Sans", size=12, color="#1a1a1a")),
        )
        st.plotly_chart(fig_sec, use_container_width=True)

    divider()

    # ── Live News Feed
    render_section(
        "Live Metals Market News",
        "Real-time headlines from across the metals sector. "
        "News is one of the fastest-moving inputs to metals stock prices — "
        "supply disruptions, Fed announcements, China data, and geopolitical events "
        "can move metals stocks 5–15% in a single session.",
    )

    with st.spinner("Loading latest metals news..."):
        all_news = get_metals_news()

    if all_news:
        render_info(
            "<strong>How to read news for metals stocks:</strong> "
            "Look for keywords like <em>copper supply disruption</em> (bullish FCX/SCCO), "
            "<em>Fed rate cut</em> (bullish gold/NEM/GOLD), "
            "<em>China stimulus</em> (bullish copper/steel), "
            "<em>strong dollar</em> (bearish all metals), "
            "<em>tariffs on steel</em> (bullish CLF/NUE/X), "
            "<em>inflation data</em> (bullish gold if high), "
            "<em>recession fears</em> (bearish copper/steel, bullish gold)."
        )
        nc1, nc2 = st.columns(2)
        for i, item in enumerate(all_news):
            with (nc1 if i % 2 == 0 else nc2):
                render_news_card(item)
    else:
        st.info("News feed temporarily unavailable. Check your internet connection.")

    divider()

    render_info(
        "<strong>Key Macro Signals to Watch:</strong> "
        "Rising gold = inflation fears or geopolitical risk increasing. "
        "Rising copper = global growth expectations improving. "
        "Falling steel prices = construction/manufacturing slowdown. "
        "VIX above 25 = elevated market fear, consider reducing metals exposure. "
        "Dollar weakening = broad commodity tailwind. "
        "China PMI above 50 = industrial metals demand expanding."
    )