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

st.markdown("""
<style>
.stSelectbox [data-baseweb="select"] > div {
    background:#ffffff !important;
    border:1.5px solid rgba(163,139,92,0.35) !important;
    border-radius:10px !important;
    color:#1a1a1a !important;
}
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] div {
    color:#1a1a1a !important;
    font-weight:500 !important;
}
[data-baseweb="popover"],[data-baseweb="menu"] {
    background:#ffffff !important;
    border:1px solid rgba(163,139,92,0.25) !important;
    border-radius:10px !important;
}
[role="option"] { color:#1a1a1a !important; background:#ffffff !important; }
[role="option"]:hover { background:rgba(163,139,92,0.10) !important; }
.ticker-card {
    background:#ffffff;
    border:1px solid rgba(163,139,92,0.22);
    border-radius:16px;
    padding:18px 20px;
    box-shadow:0 2px 10px rgba(0,0,0,0.04);
    margin-bottom:1rem;
    min-height:210px;
}
.ticker-tag {
    font-size:0.68rem;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:0.1em;
    color:#a38b5c;
    margin-bottom:4px;
}
.ticker-name {
    font-size:1.05rem;
    font-weight:700;
    color:#1a1a1a;
    margin-bottom:8px;
}
.ticker-price {
    font-size:1.55rem;
    font-weight:700;
    color:#1a1a1a;
    font-family:'DM Serif Display',serif;
    display:inline;
}
.ticker-up   { font-size:0.85rem; font-weight:600; color:#27ae60; margin-left:8px; }
.ticker-down { font-size:0.85rem; font-weight:600; color:#e74c3c; margin-left:8px; }
.ticker-desc {
    font-size:0.8rem;
    color:#6b6560;
    line-height:1.55;
    margin-top:8px;
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

TICKER_META = {
    "FCX":  {
        "name": "Freeport-McMoRan",
        "metal": "Copper Mining",
        "desc": "World's largest publicly traded copper producer. Highly sensitive to global industrial demand and China economic cycles.",
        "why": "Copper is the backbone of EV manufacturing, renewable energy infrastructure, and industrial production. FCX directly benefits when global growth accelerates.",
    },
    "NEM":  {
        "name": "Newmont Corp",
        "metal": "Gold Mining",
        "desc": "Largest gold mining company globally. Performs well during inflation, dollar weakness, and geopolitical uncertainty.",
        "why": "Gold is the world's primary safe-haven asset. NEM earnings expand dramatically when gold prices rise above its all-in sustaining costs.",
    },
    "AA":   {
        "name": "Alcoa Corp",
        "metal": "Aluminum",
        "desc": "Primary aluminum producer. Affected by energy costs, global manufacturing demand, and trade policy.",
        "why": "Aluminum demand is growing due to lightweighting in automotive and aerospace. Energy costs are the key margin driver for Alcoa.",
    },
    "CLF":  {
        "name": "Cleveland-Cliffs",
        "metal": "Steel",
        "desc": "Major flat-rolled steel producer. Tied to US auto industry, construction activity, and domestic infrastructure spending.",
        "why": "CLF is the largest supplier of steel to the US automotive industry. Infrastructure bills and tariff protection are key tailwinds.",
    },
    "X":    {
        "name": "U.S. Steel",
        "metal": "Steel",
        "desc": "Integrated steel producer. Sensitive to US manufacturing cycles, tariff policy, and construction sector health.",
        "why": "US Steel fortune is closely tied to US manufacturing activity and government trade policy.",
    },
    "NUE":  {
        "name": "Nucor Corp",
        "metal": "Steel",
        "desc": "Largest US steel producer by volume. Uses electric arc furnace technology, more energy efficient than blast furnace peers.",
        "why": "Nucor EAF technology gives it a structural cost advantage. Most consistently profitable US steel producer across cycles.",
    },
    "SCCO": {
        "name": "Southern Copper",
        "metal": "Copper",
        "desc": "Low-cost copper and molybdenum producer in Mexico and Peru. Among the most profitable copper miners globally.",
        "why": "SCCO has some of the largest and lowest-cost copper reserves in the world. High copper prices flow almost directly to its bottom line.",
    },
    "GOLD": {
        "name": "Barrick Gold",
        "metal": "Gold Mining",
        "desc": "Second largest gold miner globally. Key assets in Nevada and DRC. Tracks gold spot price closely.",
        "why": "Barrick is a pure-play gold miner with tier-one assets offering leveraged exposure to gold prices.",
    },
}


@st.cache_data(ttl=300)
def get_quick_prices():
    results = {}
    for sym in TICKER_META:
        try:
            df = yf.download(sym, period="2d", auto_adjust=False, progress=False)
            if df is None or df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            c  = df["Close"].squeeze()
            p  = float(c.iloc[-1])
            pv = float(c.iloc[-2]) if len(c) > 1 else p
            results[sym] = {"price": p, "change": (p - pv) / pv * 100}
        except Exception:
            pass
    return results


@st.cache_data(ttl=1800)
def get_ticker_news(symbol, max_items=6):
    try:
        n = yf.Ticker(symbol).news
        return n[:max_items] if n and isinstance(n, list) else []
    except Exception:
        return []


@st.cache_data(ttl=1800)
def get_metals_news():
    all_news = []
    for sym in ["FCX", "NEM", "AA", "GLD", "COPX", "SLX", "X", "GOLD"]:
        try:
            n = yf.Ticker(sym).news
            if n and isinstance(n, list):
                all_news.extend(n[:3])
        except Exception:
            pass
    seen, unique = set(), []
    for item in all_news:
        if "content" in item and isinstance(item.get("content"), dict):
            t = item["content"].get("title", "")
        else:
            t = item.get("title", "")
        if t and t not in seen:
            seen.add(t)
            unique.append(item)
    return unique[:12]


def parse_news_item(item):
    if "content" in item and isinstance(item.get("content"), dict):
        c  = item["content"]
        title     = c.get("title", "")
        link      = c.get("canonicalUrl", {}).get("url", "#")
        publisher = c.get("provider", {}).get("displayName", "")
        pub_date  = c.get("pubDate", "")
        date_str  = pub_date[:10] if pub_date else ""
    else:
        title     = item.get("title", "")
        link      = item.get("link", "#")
        publisher = item.get("publisher", "")
        ts        = item.get("providerPublishTime", 0)
        date_str  = datetime.fromtimestamp(ts).strftime("%b %d, %Y") if ts else ""
    return title, link, publisher, date_str


def render_news_card(item):
    title, link, publisher, date_str = parse_news_item(item)
    if not title:
        return
    pub_html  = f"{publisher} &nbsp;·&nbsp; {date_str}" if publisher else date_str
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(163,139,92,0.2);
                border-radius:12px;padding:16px 18px;margin:0.5rem 0;
                box-shadow:0 2px 8px rgba(0,0,0,0.04);">
        <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:#a38b5c;margin-bottom:0.4rem;">
            {pub_html}
        </div>
        <a href="{link}" target="_blank"
           style="font-size:0.94rem;font-weight:600;color:#1a1a1a;
                  text-decoration:none;line-height:1.5;">{title}</a>
    </div>""", unsafe_allow_html=True)


def stock_info_banner(ticker):
    meta = TICKER_META.get(ticker, {})
    if meta:
        render_info(
            f"<strong>{meta.get('name', ticker)} ({ticker})</strong> "
            f"— {meta.get('metal', '')}. "
            f"{meta.get('desc', '')} "
            f"{meta.get('why', '')}"
        )


# ── Shared chart layout helper
def apply_chart_style(fig, height=500):
    fig.update_layout(
        paper_bgcolor="#faf6f0",
        plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="rgba(163,139,92,0.2)",
            borderwidth=1,
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="left", x=0,
            font=dict(size=11),
        ),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor="rgba(163,139,92,0.3)",
            font=dict(family="DM Sans", size=12, color="#1a1a1a"),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(163,139,92,0.10)",
        linecolor="rgba(163,139,92,0.15)",
        tickfont=dict(size=11, color="#6b6560"),
        showgrid=True,
    )
    fig.update_yaxes(
        gridcolor="rgba(163,139,92,0.10)",
        linecolor="rgba(163,139,92,0.15)",
        tickfont=dict(size=11, color="#6b6560"),
        showgrid=True,
    )
    return fig


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

    render_section(
        "Metals Universe — 8 Covered Equities",
        "Live prices and descriptions for every stock in this platform. "
        "Prices refresh every 5 minutes. Use the navigation bar above to analyse any stock in depth.",
    )

    with st.spinner("Loading live prices..."):
        prices = get_quick_prices()

    tickers = list(TICKER_META.keys())

    for row_tickers in [tickers[:4], tickers[4:]]:
        cols = st.columns(4)
        for col, sym in zip(cols, row_tickers):
            meta   = TICKER_META[sym]
            pd_    = prices.get(sym, {})
            price  = pd_.get("price", None)
            change = pd_.get("change", None)
            ps     = f"${price:,.2f}" if price else "—"
            cs     = f"{change:+.2f}%" if change is not None else ""
            cc     = "ticker-up" if (change or 0) >= 0 else "ticker-down"
            with col:
                st.markdown(f"""
                <div class="ticker-card">
                    <div class="ticker-tag">{sym} &mdash; {meta['metal']}</div>
                    <div class="ticker-name">{meta['name']}</div>
                    <div>
                        <span class="ticker-price">{ps}</span>
                        <span class="{cc}">{cs}</span>
                    </div>
                    <div class="ticker-desc">{meta['desc']}</div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown("<br style='display:block;margin:0;padding:0;'>",
                    unsafe_allow_html=True)

    divider()

    render_section(
        "Platform Modules",
        "Navigate using the tab bar above. Each module shares a common live data layer.",
    )

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Market Analytics</div>
            <div class="nav-card-title">Overview</div>
            <div class="nav-card-text">Live candlestick chart with Bollinger Bands,
            MA10/20/50, RSI, MACD, and volume. Period selector and CSV download.</div>
        </div>""", unsafe_allow_html=True)
    with r1c2:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Predictive Intelligence</div>
            <div class="nav-card-title">Direction Prediction</div>
            <div class="nav-card-text">Random Forest classifier trained on 2 years of
            technical data. Shows direction signal, probability gauge, and feature importance.</div>
        </div>""", unsafe_allow_html=True)
    with r1c3:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Risk Analytics</div>
            <div class="nav-card-title">Risk &amp; Volatility</div>
            <div class="nav-card-text">Annualised volatility, rolling drawdown, VaR at
            95% and 99% confidence, Sharpe ratio, and full risk summary statistics.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Financial Integrity</div>
            <div class="nav-card-title">Fraud Detection</div>
            <div class="nav-card-text">Beneish-style screening using 6 accounting
            ratios to flag potential earnings manipulation or financial irregularities.</div>
        </div>""", unsafe_allow_html=True)
    with r2c2:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Investment Intelligence</div>
            <div class="nav-card-title">Recommendation</div>
            <div class="nav-card-text">MetalScore (0-100) combining RSI, MACD, momentum,
            volatility, and Bollinger position into a Buy / Hold / Avoid signal.</div>
        </div>""", unsafe_allow_html=True)
    with r2c3:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Position Risk Management</div>
            <div class="nav-card-title">Stop-Loss Assistant</div>
            <div class="nav-card-text">ATR-based stop-loss calculator with position
            sizing, take-profit targets, and a price levels chart.</div>
        </div>""", unsafe_allow_html=True)
    with r2c4:
        st.markdown("""<div class="nav-card">
            <div class="nav-card-tag">Global Markets &amp; Economics</div>
            <div class="nav-card-title">Macro &amp; News</div>
            <div class="nav-card-text">Commodity price charts, VIX, treasury yields,
            US Dollar, sector comparison, live news, and macro driver explanations.</div>
        </div>""", unsafe_allow_html=True)

    divider()

    render_section(
        "Latest Metals Market News",
        "Live headlines from across the metals sector. Updated every 30 minutes.",
    )
    with st.spinner("Loading news..."):
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
            "<strong>Data Layer</strong> — Live prices and financial data from Yahoo Finance APIs. "
            "Historical data (up to 2 years) trains the models. The latest data drives live outputs.<br><br>"
            "<strong>Feature Layer</strong> — 15+ technical indicators computed from raw price data: "
            "RSI, MACD, Bollinger Bands, ATR, rolling volatility, momentum, and volume ratios.<br><br>"
            "<strong>Model Layer</strong> — Separate ML models for each task: Random Forest for "
            "price direction, rule-based Beneish scoring for fraud, and a hybrid weighted "
            "scoring engine for the MetalScore recommendation.<br><br>"
            "<strong>Decision Layer</strong> — All outputs converge into plain-language signals "
            "with full attribution so you understand exactly why each recommendation was made."
        )
    with col_b:
        render_section("Why Metals?")
        st.markdown("""<div class="soft-panel">
            <div style="font-size:0.88rem;color:#3a3530;line-height:1.8;">
                Metals stocks are among the most volatile and macro-sensitive equities in any
                market — responding to commodity price cycles, global industrial demand,
                geopolitical risk, and monetary policy.<br><br>
                <strong>Gold miners</strong> hedge against inflation and dollar weakness.
                <strong>Copper producers</strong> track global growth and EV demand.
                <strong>Steel companies</strong> reflect construction and manufacturing cycles.<br><br>
                This diversity makes metals a uniquely rich sector for multi-factor analytics —
                and this platform gives retail investors access to the same framework used by
                professional metals analysts.
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
        "metals stock. All charts update with the latest available market data.",
    )

    col_sel1, col_sel2 = st.columns([2, 1])
    with col_sel1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="ov_t")
    with col_sel2:
        period = st.selectbox("Time Period",
                              ["1mo", "3mo", "6mo", "1y", "2y"], index=2, key="ov_p")

    ticker = get_ticker_symbol(selected_label)
    stock_info_banner(ticker)

    df_raw = load_price_data(ticker, period)
    if df_raw.empty:
        st.error("No data returned. Try another ticker or wait 30 seconds.")
        st.stop()

    df = add_technical_indicators(df_raw)
    close       = df["Close"].squeeze()
    prev        = float(close.iloc[-2]) if len(close) > 1 else float(close.iloc[-1])
    latest      = float(close.iloc[-1])
    chg         = latest - prev
    chg_pct     = chg / prev * 100
    phi         = float(df["High"].squeeze().max())
    plo         = float(df["Low"].squeeze().min())
    avg_vol     = float(df["Volume"].squeeze().mean())
    rsi         = float(df["RSI"].dropna().iloc[-1])
    vol20       = float(df["Volatility_20d"].dropna().iloc[-1]) * 100
    macd_v      = float(df["MACD"].dropna().iloc[-1])
    macd_s      = float(df["MACD_Signal"].dropna().iloc[-1])
    rsi_lbl     = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
    macd_lbl    = "Bullish" if macd_v > macd_s else "Bearish"

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Last Price",   f"${latest:,.2f}",  f"{chg_pct:+.2f}%")
    k2.metric("Period High",  f"${phi:,.2f}")
    k3.metric("Period Low",   f"${plo:,.2f}")
    k4.metric("RSI (14)",     f"{rsi:.1f}",        rsi_lbl)
    k5.metric("20d Vol",      f"{vol20:.1f}%")
    k6.metric("MACD",         macd_lbl,            f"{macd_v:+.3f}")

    pf = render_success if chg >= 0 else render_danger
    pf(
        f"<strong>{ticker}</strong> closed at <strong>${latest:,.2f}</strong>, "
        f"{'up' if chg >= 0 else 'down'} <strong>{abs(chg_pct):.2f}%</strong>. "
        f"RSI <strong>{rsi:.1f}</strong> — {rsi_lbl.lower()} momentum "
        f"({'may be due for pullback' if rsi > 70 else 'potential buying zone' if rsi < 30 else 'no extreme readings'}). "
        f"MACD <strong>{macd_lbl.lower()}</strong>. "
        f"20-day annualised volatility: <strong>{vol20:.1f}%</strong>. "
        f"Avg daily volume: <strong>{avg_vol:,.0f}</strong> shares."
    )

    divider()

    render_section(
        "Price Chart — Candlestick with Bollinger Bands & Moving Averages",
        "Green candles = bullish session (closed above open). Red candles = bearish (closed below open). "
        "Bollinger Bands (gold shading) expand during volatile periods and contract when calm. "
        "A price touching the upper band = potentially overbought. Lower band = potentially oversold. "
        "MA10 (blue), MA20 (orange), MA50 (purple) smooth short-term noise to reveal the trend. "
        "Volume bars below confirm the strength of each move — high volume on up-days is strongly bullish.",
    )

    fig1 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[0.75, 0.25], vertical_spacing=0.03)
    fig1.add_trace(go.Scatter(
        x=list(df.index) + list(df.index[::-1]),
        y=list(df["BB_Upper"].squeeze()) + list(df["BB_Lower"].squeeze()[::-1]),
        fill="toself", fillcolor="rgba(163,139,92,0.09)",
        line=dict(color="rgba(255,255,255,0)"), name="Bollinger Band",
    ), row=1, col=1)
    for col_name, dash in [("BB_Upper", "dot"), ("BB_Lower", "dot")]:
        fig1.add_trace(go.Scatter(
            x=df.index, y=df[col_name].squeeze(),
            line=dict(color="rgba(163,139,92,0.5)", width=1, dash=dash),
            showlegend=False,
        ), row=1, col=1)
    fig1.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"].squeeze(), high=df["High"].squeeze(),
        low=df["Low"].squeeze(),   close=df["Close"].squeeze(),
        increasing=dict(line=dict(color="#27ae60"), fillcolor="#27ae60"),
        decreasing=dict(line=dict(color="#e74c3c"), fillcolor="#e74c3c"),
        name=ticker,
    ), row=1, col=1)
    for ma, color, lbl in [("MA_10","#2471a3","MA10"),
                            ("MA_20","#d35400","MA20"),
                            ("MA_50","#7d3c98","MA50")]:
        if ma in df.columns:
            fig1.add_trace(go.Scatter(
                x=df.index, y=df[ma].squeeze(),
                line=dict(color=color, width=1.5), name=lbl,
            ), row=1, col=1)
    vc = ["#27ae60" if float(df["Close"].squeeze().iloc[i]) >= float(df["Open"].squeeze().iloc[i])
          else "#e74c3c" for i in range(len(df))]
    fig1.add_trace(go.Bar(
        x=df.index, y=df["Volume"].squeeze(),
        marker_color=vc, marker_opacity=0.55, name="Volume",
    ), row=2, col=1)
    apply_chart_style(fig1, height=660)
    fig1.update_layout(xaxis_rangeslider_visible=False)
    fig1.update_yaxes(title_text="Price (USD)", row=1, col=1,
                      title_font=dict(size=11, color="#6b6560"))
    fig1.update_yaxes(title_text="Volume",      row=2, col=1,
                      title_font=dict(size=11, color="#6b6560"))
    for row in [1, 2]:
        fig1.update_xaxes(gridcolor="rgba(163,139,92,0.10)",
                          tickfont=dict(size=11, color="#6b6560"),
                          row=row, col=1)
        fig1.update_yaxes(gridcolor="rgba(163,139,92,0.10)",
                          tickfont=dict(size=11, color="#6b6560"),
                          row=row, col=1)
    st.plotly_chart(fig1, use_container_width=True)

    divider()

    render_section(
        "Momentum Indicators — RSI & MACD",
        "RSI (0-100): above 70 = overbought (potential sell zone), below 30 = oversold (potential buy zone), "
        "50 = neutral dividing line. "
        "MACD: when the blue MACD line crosses above the orange signal line it is a bullish crossover — "
        "a widely used entry signal. Green histogram bars = bullish momentum building. "
        "Red histogram bars = bearish momentum building.",
    )

    fig2 = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["RSI — Relative Strength Index (14-Period)",
                        "MACD — Moving Average Convergence Divergence (12/26/9)"],
        row_heights=[0.42, 0.58], vertical_spacing=0.13,
    )
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["RSI"].squeeze(),
        line=dict(color="#2471a3", width=2.5),
        fill="tozeroy", fillcolor="rgba(36,113,163,0.06)", name="RSI",
    ), row=1, col=1)
    fig2.add_hrect(y0=70, y1=100, fillcolor="rgba(231,76,60,0.07)",
                   line_width=0, row=1, col=1)
    fig2.add_hrect(y0=0,  y1=30,  fillcolor="rgba(39,174,96,0.07)",
                   line_width=0, row=1, col=1)
    for y, c, d in [(70,"#e74c3c","dash"),(30,"#27ae60","dash"),(50,"#6b6560","dot")]:
        fig2.add_hline(y=y, line=dict(color=c, width=1.2, dash=d), row=1, col=1)
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["MACD"].squeeze(),
        line=dict(color="#2471a3", width=2.5), name="MACD",
    ), row=2, col=1)
    fig2.add_trace(go.Scatter(
        x=df.index, y=df["MACD_Signal"].squeeze(),
        line=dict(color="#d35400", width=1.8, dash="dash"), name="Signal",
    ), row=2, col=1)
    mh  = df["MACD_Hist"].squeeze()
    mhc = ["#27ae60" if v >= 0 else "#e74c3c" for v in mh]
    fig2.add_trace(go.Bar(
        x=df.index, y=mh, marker_color=mhc,
        marker_opacity=0.7, name="Histogram",
    ), row=2, col=1)
    apply_chart_style(fig2, height=560)
    fig2.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1,
                      title_font=dict(size=11, color="#6b6560"),
                      gridcolor="rgba(163,139,92,0.10)",
                      tickfont=dict(size=11, color="#6b6560"))
    fig2.update_yaxes(title_text="MACD", row=2, col=1,
                      title_font=dict(size=11, color="#6b6560"),
                      gridcolor="rgba(163,139,92,0.10)",
                      tickfont=dict(size=11, color="#6b6560"))
    for row in [1, 2]:
        fig2.update_xaxes(gridcolor="rgba(163,139,92,0.10)",
                          tickfont=dict(size=11, color="#6b6560"),
                          row=row, col=1)
    st.plotly_chart(fig2, use_container_width=True)

    divider()

    render_section(f"Latest News — {ticker}",
                   "Recent headlines directly related to this stock.")
    with st.spinner("Loading news..."):
        sn = get_ticker_news(ticker)
    if sn:
        nc1, nc2 = st.columns(2)
        for i, item in enumerate(sn):
            with (nc1 if i % 2 == 0 else nc2):
                render_news_card(item)
    else:
        st.info("No recent news found for this ticker.")

    divider()

    render_section("Recent Trading Sessions",
                   "Last 15 sessions — OHLCV plus key indicators. Most recent first.")
    dcols = ["Open","High","Low","Close","Volume","RSI","MACD","BB_Pct","Volatility_20d"]
    avail = [c for c in dcols if c in df.columns]
    rec   = df[avail].tail(15).copy()
    for c in ["Open","High","Low","Close"]:
        if c in rec.columns:
            rec[c] = rec[c].squeeze().apply(lambda x: f"${x:,.2f}")
    if "Volume" in rec.columns:
        rec["Volume"] = rec["Volume"].squeeze().apply(lambda x: f"{int(x):,}")
    if "RSI" in rec.columns:
        rec["RSI"] = rec["RSI"].squeeze().apply(lambda x: f"{x:.1f}")
    if "MACD" in rec.columns:
        rec["MACD"] = rec["MACD"].squeeze().apply(lambda x: f"{x:+.4f}")
    if "BB_Pct" in rec.columns:
        rec["BB_Pct"] = rec["BB_Pct"].squeeze().apply(lambda x: f"{x*100:.1f}%")
        rec = rec.rename(columns={"BB_Pct": "BB Position"})
    if "Volatility_20d" in rec.columns:
        rec["Volatility_20d"] = rec["Volatility_20d"].squeeze().apply(lambda x: f"{x*100:.1f}%")
        rec = rec.rename(columns={"Volatility_20d": "20d Vol"})
    st.dataframe(rec.iloc[::-1], use_container_width=True)
    st.download_button("Download CSV",
                       df_raw.to_csv().encode("utf-8"),
                       f"{ticker}_{period}.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# DIRECTION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Direction":

    render_hero(
        "Predictive Intelligence",
        "Direction Prediction",
        "A machine learning model trained on 2 years of technical indicators to predict "
        "whether the selected metals stock will move Up or Down in the next trading session.",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="dir_t")
    with c2:
        model_choice = st.selectbox("Model",
            ["Random Forest (Recommended)", "Logistic Regression (Baseline)"],
            key="dir_m")

    ticker = get_ticker_symbol(selected_label)
    stock_info_banner(ticker)

    with st.spinner(f"Loading data and training model for {ticker}..."):
        df_raw = load_price_data(ticker, "2y")

    if df_raw.empty:
        st.error("No data returned. Wait 30 seconds and try again.")
        st.stop()

    data, fcols = get_ml_features(df_raw)
    data["Target"] = (data["Close"].squeeze().shift(-1) > data["Close"].squeeze()).astype(int)
    data = data.dropna(subset=fcols + ["Target"])

    if len(data) < 100:
        st.error("Not enough data to train reliably.")
        st.stop()

    X = data[fcols]
    y = data["Target"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, shuffle=False)

    mdl = (RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1)
           if "Random Forest" in model_choice
           else LogisticRegression(max_iter=1000, random_state=42))
    mdl.fit(Xtr, ytr)
    ypred   = mdl.predict(Xte)
    acc     = accuracy_score(yte, ypred)
    lf      = X.iloc[[-1]]
    pred    = mdl.predict(lf)[0]
    proba   = mdl.predict_proba(lf)[0]
    p_up    = proba[1]
    p_dn    = proba[0]
    sig     = "UP" if pred == 1 else "DOWN"
    sc      = "#27ae60" if sig == "UP" else "#e74c3c"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Ticker",     ticker)
    k2.metric("Direction",  sig,            f"{p_up:.1%} probability up")
    k3.metric("Accuracy",   f"{acc:.1%}",   "held-out test set")
    k4.metric("Trained on", f"{len(Xtr):,}", f"{len(Xte):,} test samples")

    (render_success if sig == "UP" else render_danger)(
        f"<strong>{'Bullish' if sig == 'UP' else 'Bearish'} Signal — {ticker} is predicted to move {sig}.</strong> "
        f"Model confidence: <strong>{(p_up if sig=='UP' else p_dn):.1%}</strong>. "
        f"Trained on {len(Xtr):,} days of technical data. Test accuracy: <strong>{acc:.1%}</strong>. "
        f"This is one signal — always combine with fundamentals and risk management."
    )

    divider()

    render_section("Confidence Gauge",
                   "Above 65% = strong bullish signal. Below 35% = strong bearish. "
                   "35-65% = uncertain — treat with caution.")

    g1, g2 = st.columns(2)
    with g1:
        fg = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=p_up * 100,
            delta={"reference": 50, "valueformat": ".1f",
                   "increasing": {"color": "#27ae60"},
                   "decreasing": {"color": "#e74c3c"}},
            number={"suffix": "%", "font": {"size": 52, "color": sc,
                                            "family": "DM Serif Display"}},
            title={"text": "Probability of UP Move Tomorrow",
                   "font": {"size": 13, "color": "#6b6560", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0,100], "tickfont": {"size": 11}},
                "bar": {"color": sc, "thickness": 0.28},
                "bgcolor": "#ffffff",
                "borderwidth": 1, "bordercolor": "rgba(163,139,92,0.2)",
                "steps": [
                    {"range": [0,35],   "color": "rgba(231,76,60,0.14)"},
                    {"range": [35,50],  "color": "rgba(231,76,60,0.05)"},
                    {"range": [50,65],  "color": "rgba(39,174,96,0.05)"},
                    {"range": [65,100], "color": "rgba(39,174,96,0.14)"},
                ],
                "threshold": {"line": {"color": "#a38b5c", "width": 3},
                              "thickness": 0.75, "value": 50},
            },
        ))
        fg.update_layout(paper_bgcolor="#faf6f0", font=dict(family="DM Sans"),
                         height=320, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fg, use_container_width=True)

    with g2:
        fb = go.Figure()
        fb.add_trace(go.Bar(
            x=["DOWN","UP"], y=[p_dn*100, p_up*100],
            marker_color=["#e74c3c","#27ae60"], marker_opacity=0.85,
            text=[f"{p_dn:.1%}", f"{p_up:.1%}"],
            textposition="outside",
            textfont=dict(size=16, color="#1a1a1a", family="DM Serif Display"),
            width=0.45,
        ))
        fb.add_hline(y=50, line=dict(color="#a38b5c", width=2, dash="dash"))
        fb.update_layout(
            paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
            font=dict(family="DM Sans", color="#1a1a1a", size=12),
            height=320, margin=dict(l=10,r=10,t=50,b=10),
            title=dict(text="Up vs Down Probability",
                       font=dict(size=14, color="#1a1a1a"), x=0),
            yaxis=dict(range=[0,115], title="Probability (%)",
                       gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11, color="#6b6560")),
            xaxis=dict(tickfont=dict(size=14, color="#1a1a1a")),
            showlegend=False,
        )
        st.plotly_chart(fb, use_container_width=True)

    divider()

    render_section("Feature Importance — What Is Driving This Prediction?",
                   "Higher score = the model relied more on that signal. "
                   "Green = above median importance (primary drivers). "
                   "Gold = secondary signals.")

    imps = (mdl.feature_importances_ if hasattr(mdl, "feature_importances_")
            else np.abs(mdl.coef_[0]))
    FLABELS = {
        "Return_1d": "1-Day Return", "Return_5d": "5-Day Return",
        "Return_10d": "10-Day Return", "Return_20d": "20-Day Return",
        "MACD": "MACD", "MACD_Signal": "MACD Signal",
        "MACD_Hist": "MACD Histogram", "RSI": "RSI (14)",
        "BB_Width": "Bollinger Band Width", "BB_Pct": "BB Position",
        "ATR_Pct": "ATR (Normalised)", "Volatility_20d": "20d Volatility",
        "Volume_Ratio": "Volume Ratio", "Momentum_10": "10d Momentum",
        "Momentum_20": "20d Momentum",
    }
    fdf = pd.DataFrame({"Feature": fcols, "Importance": imps}).sort_values("Importance", ascending=True)
    fdf["Label"] = fdf["Feature"].map(FLABELS).fillna(fdf["Feature"])
    med = fdf["Importance"].median()
    fc_colors = ["#27ae60" if v >= med else "#a38b5c" for v in fdf["Importance"]]
    ff = go.Figure(go.Bar(
        x=fdf["Importance"], y=fdf["Label"], orientation="h",
        marker_color=fc_colors, marker_opacity=0.85,
        text=[f"{v:.4f}" for v in fdf["Importance"]],
        textposition="outside", textfont=dict(size=10, color="#6b6560"),
    ))
    ff.update_layout(
        paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
        font=dict(family="DM Sans", color="#1a1a1a", size=12),
        height=480, margin=dict(l=10, r=70, t=20, b=10),
        xaxis=dict(title="Importance Score",
                   gridcolor="rgba(163,139,92,0.10)",
                   tickfont=dict(size=11, color="#6b6560")),
        yaxis=dict(tickfont=dict(size=11, color="#1a1a1a")),
        showlegend=False,
    )
    st.plotly_chart(ff, use_container_width=True)

    divider()

    render_section("Signal History — Predictions vs Actual Outcomes",
                   "Green circle = correct prediction. Red X = incorrect. "
                   "Errors often cluster around sharp news-driven reversals.")

    rdf = Xte.copy()
    rdf["Actual"]    = yte.values
    rdf["Predicted"] = ypred
    rdf["Correct"]   = rdf["Actual"] == rdf["Predicted"]
    rdf["Close"]     = data.loc[Xte.index, "Close"].squeeze().values
    ok   = rdf[rdf["Correct"]]
    bad  = rdf[~rdf["Correct"]]

    fh = go.Figure()
    fh.add_trace(go.Scatter(x=rdf.index, y=rdf["Close"],
                            line=dict(color="#a38b5c", width=2),
                            fill="tozeroy", fillcolor="rgba(163,139,92,0.05)",
                            name="Close Price"))
    fh.add_trace(go.Scatter(x=ok.index, y=ok["Close"], mode="markers",
                            marker=dict(color="#27ae60", size=9, symbol="circle",
                                        line=dict(color="white", width=1.5)),
                            name=f"Correct ({len(ok)}/{len(rdf)})"))
    fh.add_trace(go.Scatter(x=bad.index, y=bad["Close"], mode="markers",
                            marker=dict(color="#e74c3c", size=9, symbol="x",
                                        line=dict(color="#e74c3c", width=2)),
                            name=f"Incorrect ({len(bad)}/{len(rdf)})"))
    apply_chart_style(fh, height=420)
    fh.update_yaxes(title_text="Price (USD)",
                    title_font=dict(size=11, color="#6b6560"))
    st.plotly_chart(fh, use_container_width=True)

    with st.expander("Full Classification Report"):
        render_info("Precision = correct when predicting a class. "
                    "Recall = how many actual instances found. "
                    "F1 = harmonic mean of both.")
        st.code(classification_report(yte, ypred, target_names=["DOWN","UP"]))

    render_alert("Disclaimer: For educational purposes only. No model can consistently "
                 "predict short-term market movements. Always combine with fundamentals "
                 "and sound risk management.")


# ══════════════════════════════════════════════════════════════════════════════
# RISK
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Risk":

    render_hero(
        "Risk Analytics",
        "Risk & Volatility",
        "Quantitative risk metrics for the selected metals stock — annualised volatility, "
        "rolling drawdown, Value-at-Risk, Sharpe ratio, and complete risk statistics.",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="risk_t")
    with c2:
        period = st.selectbox("Time Period", ["6mo","1y","2y"], index=1, key="risk_p")

    ticker = get_ticker_symbol(selected_label)
    stock_info_banner(ticker)

    df_raw = load_price_data(ticker, period)
    if df_raw.empty:
        st.error("No data returned.")
        st.stop()

    df      = add_technical_indicators(df_raw)
    rets    = df["Return_1d"].dropna()
    v20     = float(df["Volatility_20d"].dropna().iloc[-1])
    v60     = float(df["Volatility_60d"].dropna().iloc[-1])
    var95   = float(np.percentile(rets, 5))
    var99   = float(np.percentile(rets, 1))
    avgr    = float(rets.mean())
    dd      = compute_drawdown(df)
    maxdd   = float(dd.min())
    curdd   = float(dd.iloc[-1])
    sharpe  = (avgr * 252) / v20 if v20 > 0 else 0

    rl, rfn = (("Low", render_success) if v20 < 0.25
               else ("Medium", render_alert) if v20 < 0.45
               else ("High", render_danger))

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Ticker",      ticker)
    k2.metric("20d Vol",     f"{v20*100:.1f}%",  "Annualised")
    k3.metric("60d Vol",     f"{v60*100:.1f}%",  "Annualised")
    k4.metric("Risk Level",  rl)
    k5.metric("VaR (95%)",   f"{var95*100:.2f}%","Max daily loss")
    k6.metric("Max Drawdown",f"{maxdd*100:.1f}%",f"Current {curdd*100:.1f}%")

    rfn(
        f"<strong>{ticker} — Risk Level: {rl}.</strong> "
        f"20-day annualised volatility: <strong>{v20*100:.1f}%</strong>. "
        f"VaR 95%: max expected daily loss <strong>{abs(var95)*100:.2f}%</strong> "
        f"(1-in-20 chance of exceeding this). "
        f"Maximum drawdown: <strong>{abs(maxdd)*100:.1f}%</strong>. "
        f"Sharpe ratio: <strong>{sharpe:.2f}</strong> "
        f"({'good risk-adjusted return' if sharpe > 1 else 'risk not fully compensated by return'})."
    )

    divider()

    render_section("Rolling Volatility — 20d vs 60d Annualised",
                   "20-day (red) = short-term risk gauge. 60-day (blue) = medium-term. "
                   "When 20d spikes above 60d it signals sudden elevated risk — "
                   "common around earnings, macro surprises, or commodity shocks.")

    fv = go.Figure()
    fv.add_trace(go.Scatter(x=df.index, y=df["Volatility_20d"].squeeze()*100,
                            line=dict(color="#e74c3c", width=2.5),
                            fill="tozeroy", fillcolor="rgba(231,76,60,0.07)",
                            name="20-Day"))
    fv.add_trace(go.Scatter(x=df.index, y=df["Volatility_60d"].squeeze()*100,
                            line=dict(color="#2471a3", width=2, dash="dash"),
                            name="60-Day"))
    fv.add_hline(y=25, line=dict(color="#27ae60", width=1.5, dash="dot"),
                 annotation_text="Low Risk (25%)", annotation_position="bottom right",
                 annotation_font=dict(size=10, color="#27ae60"))
    fv.add_hline(y=45, line=dict(color="#e74c3c", width=1.5, dash="dot"),
                 annotation_text="High Risk (45%)", annotation_position="bottom right",
                 annotation_font=dict(size=10, color="#e74c3c"))
    apply_chart_style(fv, height=440)
    fv.update_yaxes(title_text="Annualised Volatility (%)",
                    title_font=dict(size=11, color="#6b6560"))
    st.plotly_chart(fv, use_container_width=True)

    divider()

    render_section("Rolling Drawdown — Peak-to-Trough Decline",
                   "Shows how far the stock has fallen from its most recent peak. "
                   "-20% = stock is 20% below its high. The purple line marks max drawdown.")

    fd = go.Figure()
    fd.add_trace(go.Scatter(x=dd.index, y=dd.values*100,
                            fill="tozeroy", fillcolor="rgba(231,76,60,0.15)",
                            line=dict(color="#e74c3c", width=2), name="Drawdown"))
    fd.add_hline(y=maxdd*100, line=dict(color="#7d3c98", width=2, dash="dash"),
                 annotation_text=f"Max Drawdown: {maxdd*100:.1f}%",
                 annotation_position="bottom right",
                 annotation_font=dict(size=11, color="#7d3c98"))
    apply_chart_style(fd, height=400)
    fd.update_yaxes(title_text="Drawdown (%)",
                    title_font=dict(size=11, color="#6b6560"))
    fd.update_layout(showlegend=False)
    st.plotly_chart(fd, use_container_width=True)

    divider()

    render_section("Daily Return Distribution & VaR",
                   "Frequency histogram of daily returns. VaR 95% (red) = 5% chance of losing more "
                   "than this on any given day. VaR 99% (purple) = 1% chance.")

    fdi = go.Figure()
    fdi.add_trace(go.Histogram(x=rets*100, nbinsx=70,
                               marker_color="#a38b5c", marker_opacity=0.75))
    fdi.add_vline(x=var95*100, line=dict(color="#e74c3c", width=2.5, dash="dash"),
                  annotation_text=f"VaR 95%: {var95*100:.2f}%",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color="#e74c3c"))
    fdi.add_vline(x=var99*100, line=dict(color="#7d3c98", width=2.5, dash="dash"),
                  annotation_text=f"VaR 99%: {var99*100:.2f}%",
                  annotation_position="top left",
                  annotation_font=dict(size=11, color="#7d3c98"))
    fdi.add_vline(x=avgr*100, line=dict(color="#27ae60", width=2, dash="dot"),
                  annotation_text=f"Avg: {avgr*100:.3f}%",
                  annotation_position="top right",
                  annotation_font=dict(size=11, color="#27ae60"))
    apply_chart_style(fdi, height=420)
    fdi.update_xaxes(title_text="Daily Return (%)")
    fdi.update_yaxes(title_text="Number of Days")
    fdi.update_layout(showlegend=False, bargap=0.04)
    st.plotly_chart(fdi, use_container_width=True)

    divider()

    render_section("Complete Risk Statistics")
    rt1, rt2 = st.columns(2)
    with rt1:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Volatility & Return</strong>", unsafe_allow_html=True)
        for lbl, val in {
            "20d Annualised Volatility": f"{v20*100:.2f}%",
            "60d Annualised Volatility": f"{v60*100:.2f}%",
            "Risk Classification":       rl,
            "Sharpe Ratio (Ann.)":       f"{sharpe:.3f}",
            "Avg Daily Return":          f"{avgr*100:.3f}%",
            "Est. Annual Return":        f"{avgr*252*100:.1f}%",
        }.items():
            st.markdown(
                f'<div class="stat-row"><span class="stat-label">{lbl}</span>'
                f'<span class="stat-value">{val}</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with rt2:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Downside Risk</strong>", unsafe_allow_html=True)
        for lbl, val in {
            "VaR (95%)":      f"{var95*100:.3f}%",
            "VaR (99%)":      f"{var99*100:.3f}%",
            "Max Drawdown":   f"{maxdd*100:.2f}%",
            "Current Drawdown": f"{curdd*100:.2f}%",
            "Worst Day":      f"{rets.min()*100:.2f}%",
            "Best Day":       f"{rets.max()*100:.2f}%",
        }.items():
            st.markdown(
                f'<div class="stat-row"><span class="stat-label">{lbl}</span>'
                f'<span class="stat-value">{val}</span></div>',
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FRAUD
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Fraud":

    render_hero(
        "Financial Integrity Screening",
        "Fraud & Manipulation Detection",
        "A Beneish-style accounting quality screening tool that analyses 6 financial "
        "ratios to detect potential earnings manipulation in metals sector companies.",
    )

    selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="fr_t")
    ticker = get_ticker_symbol(selected_label)
    stock_info_banner(ticker)

    with st.spinner(f"Fetching financial data for {ticker}..."):
        scores = get_beneish_scores(ticker)

    if "Error" in scores:
        st.error(f"Could not retrieve data: {scores['Error']}")
        st.stop()

    overall = scores.get("Overall Risk", "Unknown")
    flags   = scores.get("Flags Triggered", "0 / 0")

    (render_danger if "High" in overall else
     render_alert  if "Moderate" in overall else render_success)(
        f"<strong>{ticker} — {overall}.</strong> "
        f"{flags} flags triggered. "
        f"{'Deep due diligence recommended before investing.' if 'High' in overall else 'Monitor closely around earnings.' if 'Moderate' in overall else 'Financial statement quality appears acceptable.'}"
    )

    divider()

    render_section(
        "Financial Ratio Analysis — 6 Beneish-Style Metrics",
        "Each ratio measures a different dimension of financial statement quality. "
        "Warning = outside the normal range for metals companies. "
        "Does not confirm fraud — signals areas requiring closer scrutiny.",
    )

    RK = ["Gross Margin (%)","Asset Quality Index","Leverage Ratio",
          "Net Profit Margin (%)","Current Ratio","FCF Margin (%)"]
    FK = ["GM Flag","AQI Flag","LEV Flag","NPM Flag","CR Flag","FCF Flag"]
    DESC = {
        "Gross Margin (%)":       "Revenue minus COGS as % of revenue. Below 20% may signal pricing pressure or cost inflation.",
        "Asset Quality Index":    "Non-current assets / total assets. Above 0.75 = high proportion of hard-to-value long-term assets.",
        "Leverage Ratio":         "Total debt / total assets. Above 0.60 = high financial risk and elevated default probability in downturns.",
        "Net Profit Margin (%)":  "Net income as % of revenue. Below 3% signals structural profitability problems.",
        "Current Ratio":          "Current assets / current liabilities. Below 1.0 = more short-term obligations than assets — liquidity risk.",
        "FCF Margin (%)":         "Free cash flow / revenue. Negative = cash consumed faster than generated. Sustained negative FCF is a major red flag.",
    }

    c1, c2 = st.columns(2)
    for i, (rk, fk) in enumerate(zip(RK, FK)):
        val   = scores.get(rk, "N/A")
        flag  = scores.get(fk, "Normal")
        warn  = flag == "Warning"
        col   = c1 if i % 2 == 0 else c2
        bg    = "#ffebee" if warn else "#f0faf1"
        bc    = "#ef9a9a" if warn else "#a5d6a7"
        lc    = "#b71c1c" if warn else "#1b5e20"
        icon  = "Warning" if warn else "Normal"
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1px solid {bc};border-left:4px solid {lc};
                        border-radius:0 12px 12px 0;padding:16px 18px;margin:0.6rem 0;">
                <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                    <span style="font-weight:700;font-size:0.92rem;color:#1a1a1a;">{rk}</span>
                    <span style="font-weight:700;font-size:0.82rem;color:{lc};">{icon}</span>
                </div>
                <div style="font-size:1.6rem;font-weight:700;color:#1a1a1a;
                            font-family:'DM Serif Display',serif;margin-bottom:8px;">{val}</div>
                <div style="font-size:0.82rem;color:#5a5550;line-height:1.6;">{DESC.get(rk,'')}</div>
            </div>
            """, unsafe_allow_html=True)

    divider()

    render_section("Flag Summary Chart",
                   "Red = warning triggered. Green = within normal range.")
    rc = ["#e74c3c" if scores.get(f,"Normal") == "Warning" else "#27ae60" for f in FK]
    ffr = go.Figure(go.Bar(
        x=RK, y=[1]*len(RK), marker_color=rc, marker_opacity=0.85,
        text=[scores.get(r,"N/A") for r in RK],
        textposition="inside", textfont=dict(size=12, color="white"),
    ))
    ffr.update_layout(paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
                      font=dict(family="DM Sans", size=12),
                      height=260, margin=dict(l=10,r=10,t=20,b=10),
                      xaxis=dict(tickfont=dict(size=10, color="#1a1a1a")),
                      yaxis=dict(visible=False), showlegend=False)
    st.plotly_chart(ffr, use_container_width=True)

    render_info("Developed by Prof. Messod Beneish (Indiana University). "
                "This is a screening tool — always review primary source filings "
                "and consult a financial professional before drawing conclusions.")


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Recommendation":

    render_hero(
        "Investment Intelligence",
        "Recommendation Engine",
        "A hybrid MetalScore (0-100) aggregating 6 technical signals into a single "
        "Buy / Hold / Avoid recommendation with complete signal attribution.",
    )

    selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="rec_t")
    ticker = get_ticker_symbol(selected_label)
    stock_info_banner(ticker)

    with st.spinner(f"Computing MetalScore for {ticker}..."):
        df_raw = load_price_data(ticker, "1y")

    if df_raw.empty:
        st.error("No data returned.")
        st.stop()

    df   = add_technical_indicators(df_raw)
    rets = df["Return_1d"].dropna()
    rsi  = float(df["RSI"].dropna().iloc[-1])
    macdv= float(df["MACD"].dropna().iloc[-1])
    macds= float(df["MACD_Signal"].dropna().iloc[-1])
    v20  = float(df["Volatility_20d"].dropna().iloc[-1])
    mom  = float(df["Momentum_20"].dropna().iloc[-1])
    bbp  = float(df["BB_Pct"].dropna().iloc[-1])
    avgr = float(rets.mean())

    rs  = 75 if rsi < 30 else 25 if rsi > 70 else 50 + (50-rsi)*0.5
    ms  = 70 if macdv > macds else 30
    mos = 80 if mom > 0.10 else 60 if mom > 0 else 40 if mom > -0.10 else 20
    vs  = 80 if v20 < 0.25 else 55 if v20 < 0.45 else 25
    bs  = 75 if bbp < 0.2 else 35 if bbp > 0.8 else 65
    res = min(max(50 + avgr*10000, 10), 90)

    mscore = round(min(max(rs*0.20 + ms*0.20 + mos*0.25 + vs*0.15 + bs*0.10 + res*0.10, 0), 100), 1)

    rec, rc, rfn = (("BUY",   "#27ae60", render_success) if mscore >= 65 else
                    ("HOLD",  "#f9a825", render_alert)   if mscore >= 45 else
                    ("AVOID", "#e74c3c", render_danger))

    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Ticker",       ticker)
    k2.metric("MetalScore",   f"{mscore} / 100")
    k3.metric("Signal",       rec)
    k4.metric("20d Momentum", f"{mom*100:+.1f}%")

    rfn(
        f"<strong>{ticker} — {rec} (MetalScore: {mscore}/100)</strong><br>"
        f"RSI <strong>{rsi:.1f}</strong> — "
        f"{'overbought, potential pullback risk' if rsi > 70 else 'oversold, potential opportunity' if rsi < 30 else 'neutral'}. "
        f"MACD <strong>{'bullish' if macdv > macds else 'bearish'}</strong>. "
        f"20-day momentum <strong>{mom*100:+.1f}%</strong>. "
        f"Volatility <strong>{v20*100:.1f}%</strong> annualised. "
        f"BB position <strong>{bbp*100:.0f}%</strong> of band width."
    )

    divider()

    render_section("MetalScore Gauge & Signal Breakdown",
                   "Above 65 = Buy. 45-65 = Hold. Below 45 = Avoid. "
                   "The bar chart shows each signal's score and its weight in the composite.")

    sg1, sg2 = st.columns(2)
    with sg1:
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=mscore,
            number={"font": {"size": 56, "color": rc, "family": "DM Serif Display"}},
            title={"text": f"MetalScore — {rec}",
                   "font": {"size": 14, "color": "#6b6560", "family": "DM Sans"}},
            gauge={
                "axis": {"range": [0,100], "tickfont": {"size":11}},
                "bar": {"color": rc, "thickness": 0.3},
                "bgcolor": "#ffffff",
                "borderwidth": 1, "bordercolor": "rgba(163,139,92,0.2)",
                "steps": [{"range":[0,45],   "color":"rgba(231,76,60,0.12)"},
                           {"range":[45,65],  "color":"rgba(249,168,37,0.12)"},
                           {"range":[65,100], "color":"rgba(39,174,96,0.12)"}],
                "threshold": {"line":{"color":"#a38b5c","width":3},
                              "thickness":0.75,"value":mscore},
            },
        ))
        fg.update_layout(paper_bgcolor="#faf6f0", font=dict(family="DM Sans"),
                         height=340, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fg, use_container_width=True)

    with sg2:
        snames  = ["Momentum (20d)","RSI Score","MACD Signal",
                   "Volatility","BB Position","Return Quality"]
        sscores = [mos, rs, ms, vs, bs, res]
        swts    = [0.25, 0.20, 0.20, 0.15, 0.10, 0.10]
        scolors = ["#27ae60" if s>=60 else "#f9a825" if s>=40 else "#e74c3c"
                   for s in sscores]
        fsig = go.Figure(go.Bar(
            x=sscores, y=snames, orientation="h",
            marker_color=scolors, marker_opacity=0.85,
            text=[f"{s:.0f}  (wt {w:.0%})" for s,w in zip(sscores,swts)],
            textposition="outside", textfont=dict(size=10, color="#6b6560"),
        ))
        fsig.add_vline(x=50, line=dict(color="#a38b5c", width=2, dash="dash"))
        fsig.update_layout(
            paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
            font=dict(family="DM Sans", color="#1a1a1a", size=12),
            height=340, margin=dict(l=10,r=100,t=30,b=10),
            title=dict(text="Signal Breakdown", font=dict(size=13,color="#1a1a1a"), x=0),
            xaxis=dict(range=[0,130], title="Score (0-100)",
                       gridcolor="rgba(163,139,92,0.10)",
                       tickfont=dict(size=11,color="#6b6560")),
            yaxis=dict(tickfont=dict(size=11,color="#1a1a1a")),
            showlegend=False,
        )
        st.plotly_chart(fsig, use_container_width=True)

    divider()

    render_section("Sector Comparison — All 8 Stocks Rebased to 100",
                   "Selected ticker shown with thicker line. "
                   "Above 100 = positive return since period start.")

    with st.spinner("Loading sector data..."):
        sdf = load_sector_normalised("6mo")

    if not sdf.empty:
        fsc = go.Figure()
        sc_colors = ["#a38b5c","#2471a3","#27ae60","#e74c3c",
                     "#7d3c98","#d35400","#1a9c8a","#c0392b"]
        for i, col in enumerate(sdf.columns):
            fsc.add_trace(go.Scatter(
                x=sdf.index, y=sdf[col],
                line=dict(color=sc_colors[i % len(sc_colors)],
                          width=2.8 if col == ticker else 1.2),
                opacity=1.0 if col == ticker else 0.5,
                name=col,
            ))
        fsc.add_hline(y=100, line=dict(color="#6b6560", width=1.5, dash="dot"))
        apply_chart_style(fsc, height=440)
        fsc.update_yaxes(title_text="Indexed Performance (Base=100)",
                         title_font=dict(size=11,color="#6b6560"))
        st.plotly_chart(fsc, use_container_width=True)

    render_alert("Disclaimer: MetalScore uses technical indicators only — no fundamental "
                 "analysis, earnings forecasts, or valuation. Not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# STOP-LOSS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Stop-Loss":

    render_hero(
        "Position Risk Management",
        "Stop-Loss Assistant",
        "ATR-based stop-loss and position sizing calculator. Enter your investment amount "
        "and risk tolerance to get a personalised stop-loss level and position size.",
    )

    c1, c2 = st.columns([2, 1])
    with c1:
        selected_label = st.selectbox("Select a Metals Stock", ticker_labels, key="sl_t")
    with c2:
        rp = st.selectbox("Risk Profile", ["Conservative","Moderate","Aggressive"],
                          index=1, key="sl_r")

    ticker = get_ticker_symbol(selected_label)
    stock_info_banner(ticker)

    df_raw = load_price_data(ticker, "3mo")
    if df_raw.empty:
        st.error("No data returned.")
        st.stop()

    df   = add_technical_indicators(df_raw)
    lp   = float(df["Close"].squeeze().iloc[-1])
    atr  = float(df["ATR"].dropna().iloc[-1])
    v20  = float(df["Volatility_20d"].dropna().iloc[-1])

    render_info(
        f"<strong>Current ATR ({ticker}): ${atr:.2f}</strong> — "
        f"this stock typically moves ±${atr:.2f} per share in a single day. "
        f"Stop-losses set tighter than 1x ATR will be triggered by normal daily price fluctuations. "
        f"The calculator below sizes your stop to the stock's actual volatility."
    )

    divider()

    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        inv = st.number_input("Investment Amount ($)",
                              min_value=100, max_value=1000000, value=10000, step=500)
    with ci2:
        mlp = st.slider("Max Acceptable Loss (%)", 1, 30, 10)
    with ci3:
        st.metric("Current Price", f"${lp:,.2f}")
        st.metric("14-Day ATR",   f"${atr:,.2f}")

    am   = {"Conservative":1.5,"Moderate":2.0,"Aggressive":3.0}[rp]
    pm   = {"Conservative":0.5,"Moderate":1.0,"Aggressive":1.5}[rp]
    satr = lp - atr * am
    spct = lp * (1 - mlp/100 * pm)
    sf   = max(satr, spct)
    sd   = lp - sf
    sdp  = sd / lp * 100
    mlu  = inv * mlp / 100
    shs  = int(mlu / sd) if sd > 0 else 0
    posv = shs * lp
    tp   = lp + sd * 2

    divider()

    render_section(f"Your Stop-Loss — {rp} Profile",
                   f"ATR multiplier: {am}x. Max acceptable loss: {mlp}% (${mlu:,.0f}).")

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Entry",          f"${lp:,.2f}")
    k2.metric("Stop-Loss",      f"${sf:,.2f}",   f"-{sdp:.1f}%")
    k3.metric("Take-Profit 2:1",f"${tp:,.2f}",   f"+{sdp*2:.1f}%")
    k4.metric("Shares",         f"{shs:,}",       f"${posv:,.0f}")
    k5.metric("Max Loss",       f"${mlu:,.0f}",   f"{mlp}% of capital")

    (render_alert if sdp < 3 else render_success)(
        f"<strong>{'Warning: stop is very tight' if sdp < 3 else 'Stop-loss set'} at ${sf:,.2f} ({sdp:.1f}% below entry).</strong> "
        f"{'Normal daily price moves could trigger this prematurely. Consider widening.' if sdp < 3 else ''} "
        f"Based on {am}x ATR (${atr:.2f}). Take-profit: ${tp:,.2f} (+{sdp*2:.1f}%). "
        f"Suggested position: {shs} shares (${posv:,.0f}). "
        f"Risk-reward 2:1 — you only need to be right 34% of the time to be profitable."
    )

    divider()

    render_section("Price Levels Chart",
                   "Entry (blue), stop-loss (red), take-profit (green) on 3-month price history. "
                   "Red zone = risk. Green zone = profit.")

    ps = df["Close"].squeeze()
    fsl = go.Figure()
    fsl.add_trace(go.Scatter(x=ps.index, y=ps,
                             line=dict(color="#a38b5c", width=2.5), name="Price",
                             fill="tozeroy", fillcolor="rgba(163,139,92,0.04)"))
    fsl.add_hline(y=lp, line=dict(color="#2471a3",width=2.5,dash="dash"),
                  annotation_text=f"Entry ${lp:,.2f}",
                  annotation_position="right",
                  annotation_font=dict(size=12,color="#2471a3"))
    fsl.add_hline(y=sf, line=dict(color="#e74c3c",width=2.5,dash="dash"),
                  annotation_text=f"Stop ${sf:,.2f} (-{sdp:.1f}%)",
                  annotation_position="right",
                  annotation_font=dict(size=12,color="#e74c3c"))
    fsl.add_hline(y=tp, line=dict(color="#27ae60",width=2.5,dash="dash"),
                  annotation_text=f"Target ${tp:,.2f} (+{sdp*2:.1f}%)",
                  annotation_position="right",
                  annotation_font=dict(size=12,color="#27ae60"))
    fsl.add_hrect(y0=sf, y1=lp, fillcolor="rgba(231,76,60,0.08)", line_width=0)
    fsl.add_hrect(y0=lp, y1=tp, fillcolor="rgba(39,174,96,0.08)",  line_width=0)
    apply_chart_style(fsl, height=480)
    fsl.update_layout(margin=dict(l=10,r=160,t=20,b=10))
    fsl.update_yaxes(title_text="Price (USD)",
                     title_font=dict(size=11,color="#6b6560"))
    st.plotly_chart(fsl, use_container_width=True)

    divider()

    sr1, sr2 = st.columns(2)
    with sr1:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Position Details</strong>", unsafe_allow_html=True)
        for l,v in {"Risk Profile":rp,"Entry":f"${lp:,.2f}","Stop":f"${sf:,.2f}",
                    "Stop Distance":f"${sd:,.2f} ({sdp:.1f}%)",
                    "Take-Profit":f"${tp:,.2f}","ATR Mult":f"{am}x"}.items():
            st.markdown(f'<div class="stat-row"><span class="stat-label">{l}</span>'
                        f'<span class="stat-value">{v}</span></div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with sr2:
        st.markdown('<div class="soft-panel">', unsafe_allow_html=True)
        st.markdown("<strong>Risk & Sizing</strong>", unsafe_allow_html=True)
        for l,v in {"Investment":f"${inv:,.0f}","Max Loss %":f"{mlp}%",
                    "Max Loss $":f"${mlu:,.0f}","Shares":f"{shs:,}",
                    "Position Value":f"${posv:,.0f}","20d Ann. Vol":f"{v20*100:.1f}%"}.items():
            st.markdown(f'<div class="stat-row"><span class="stat-label">{l}</span>'
                        f'<span class="stat-value">{v}</span></div>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    render_info("Stop-loss levels are suggestions based on volatility analysis. "
                "Gap openings and market halts can cause losses beyond stated amounts. "
                "Confirm all orders with your broker.")


# ══════════════════════════════════════════════════════════════════════════════
# MACRO & NEWS
# ══════════════════════════════════════════════════════════════════════════════
elif nav == "Macro & News":

    render_hero(
        "Global Markets & Economics",
        "Macro & News Intelligence",
        "The macroeconomic forces, commodity trends, and live news driving metals markets. "
        "Metals stocks are among the most macro-sensitive equities in the world.",
    )

    period = st.selectbox("Time Period", ["3mo","6mo","1y","2y"], index=1, key="mac_p")

    render_section("What Moves Metals Stocks — Macro & Micro Drivers",
                   "Plain-language guide to the forces affecting metals equities right now.")

    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown("""<div class="soft-panel">
        <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:#a38b5c;margin-bottom:0.9rem;">
            Macroeconomic Factors
        </div>
        <div style="font-size:0.88rem;color:#3a3530;line-height:1.85;">
            <strong>Fed Interest Rates</strong><br>
            Rising rates = stronger USD = lower commodity prices. Rate cuts = metals rally.
            The Fed is the single most powerful macro force for metals.<br><br>
            <strong>US Dollar Index</strong><br>
            All metals priced in USD globally. Stronger dollar = metals cost more abroad = demand falls.
            Dollar weakness = broad commodity tailwind for all metals stocks.<br><br>
            <strong>China Economic Activity</strong><br>
            China consumes ~50% of global copper, ~55% aluminum, ~60% steel.
            Any China slowdown hits FCX, SCCO, AA, CLF, X, NUE immediately.<br><br>
            <strong>Geopolitical Risk</strong><br>
            Wars and sanctions drive gold higher. Supply disruptions in Peru, DRC,
            or South Africa create price spikes regardless of broader economics.
        </div>
        </div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown("""<div class="soft-panel">
        <div style="font-size:0.78rem;font-weight:700;text-transform:uppercase;
                    letter-spacing:0.08em;color:#a38b5c;margin-bottom:0.9rem;">
            Microeconomic Factors
        </div>
        <div style="font-size:0.88rem;color:#3a3530;line-height:1.85;">
            <strong>Commodity Spot Prices</strong><br>
            The most direct driver. Copper from $3.50 to $4.50/lb can double FCX earnings.
            Gold rising $200/oz adds 30-50% to a miner's free cash flow.<br><br>
            <strong>Energy &amp; Input Costs</strong><br>
            Aluminum smelting is extremely energy-intensive. High electricity prices
            compress Alcoa margins even when aluminum prices rise.<br><br>
            <strong>Earnings &amp; Production Guidance</strong><br>
            Quarterly production reports, AISC guidance, and capex plans move
            individual stocks 10-25% in a single session. Highest-volatility period.<br><br>
            <strong>EV &amp; Green Energy Demand</strong><br>
            EVs need 3-4x more copper than combustion cars. Solar panels need aluminum.
            This structural shift is a long-term bullish catalyst for FCX, SCCO, AA.
        </div>
        </div>""", unsafe_allow_html=True)

    divider()

    render_section("Commodity Price Performance — Rebased to 100",
                   "All commodity prices rebased to 100 at period start. "
                   "Rising gold = inflation/geopolitical risk. Rising copper = growth optimism.")

    with st.spinner("Loading commodity data..."):
        mdf = load_macro_data(period)

    if not mdf.empty:
        com_cols = [c for c in mdf.columns if any(x in c for x in ["Gold","Silver","Copper","Crude","Nat."])]
        if com_cols:
            fco = go.Figure()
            cc  = ["#a38b5c","#7d7d7d","#d35400","#2471a3","#27ae60"]
            for i, col in enumerate(com_cols):
                s = mdf[col].dropna()
                if s.empty: continue
                rb = s / s.iloc[0] * 100
                fco.add_trace(go.Scatter(x=rb.index, y=rb,
                                         line=dict(color=cc[i%len(cc)], width=2.5),
                                         name=col))
            fco.add_hline(y=100, line=dict(color="#6b6560",width=1.5,dash="dot"),
                          annotation_text="Base = 100", annotation_position="right",
                          annotation_font=dict(size=10,color="#6b6560"))
            apply_chart_style(fco, height=460)
            fco.update_yaxes(title_text="Indexed Price (Base=100)",
                             title_font=dict(size=11,color="#6b6560"))
            st.plotly_chart(fco, use_container_width=True)

        divider()

        render_section("Market Risk Indicators — VIX, S&P 500, 10Y Treasury & USD",
                       "VIX above 25 = elevated fear, bullish for gold. "
                       "Rising yields = headwind for gold. Strong dollar = bearish commodities.")

        mac_cols = [c for c in mdf.columns if any(x in c for x in ["S&P","VIX","Treasury","Dollar"])]
        if mac_cols:
            fma = make_subplots(rows=len(mac_cols), cols=1,
                                shared_xaxes=True, vertical_spacing=0.06,
                                subplot_titles=mac_cols)
            mcc = ["#2471a3","#e74c3c","#a38b5c","#27ae60"]
            for i, col in enumerate(mac_cols):
                s = mdf[col].dropna()
                if s.empty: continue
                try:
                    hx = mcc[i%len(mcc)]
                    fc2 = f"rgba({int(hx[1:3],16)},{int(hx[3:5],16)},{int(hx[5:7],16)},0.07)"
                except Exception:
                    fc2 = "rgba(163,139,92,0.07)"
                fma.add_trace(go.Scatter(x=s.index, y=s,
                                         line=dict(color=mcc[i%len(mcc)], width=2),
                                         fill="tozeroy", fillcolor=fc2, name=col),
                              row=i+1, col=1)
                fma.update_yaxes(gridcolor="rgba(163,139,92,0.10)",
                                 tickfont=dict(size=10,color="#6b6560"),
                                 row=i+1, col=1)
                fma.update_xaxes(gridcolor="rgba(163,139,92,0.10)",
                                 tickfont=dict(size=10,color="#6b6560"),
                                 row=i+1, col=1)
            fma.update_layout(paper_bgcolor="#faf6f0", plot_bgcolor="#faf6f0",
                              font=dict(family="DM Sans",color="#1a1a1a",size=12),
                              height=max(130*len(mac_cols)+60,300),
                              margin=dict(l=10,r=10,t=30,b=10),
                              showlegend=False, hovermode="x unified",
                              hoverlabel=dict(bgcolor="#ffffff",
                                             font=dict(family="DM Sans",size=12,color="#1a1a1a")))
            st.plotly_chart(fma, use_container_width=True)

    divider()

    render_section("Metals Sector — All 8 Stocks Relative Performance",
                   "Rebased to 100. Leaders above 100, laggards below.")

    with st.spinner("Loading sector data..."):
        sdf2 = load_sector_normalised(period)

    if not sdf2.empty:
        fs2 = go.Figure()
        sc2 = ["#a38b5c","#2471a3","#27ae60","#e74c3c",
               "#7d3c98","#d35400","#1a9c8a","#c0392b"]
        for i, col in enumerate(sdf2.columns):
            fs2.add_trace(go.Scatter(x=sdf2.index, y=sdf2[col],
                                     line=dict(color=sc2[i%len(sc2)], width=2),
                                     name=col))
        fs2.add_hline(y=100, line=dict(color="#6b6560",width=1.5,dash="dot"))
        apply_chart_style(fs2, height=460)
        fs2.update_yaxes(title_text="Indexed Performance (Base=100)",
                         title_font=dict(size=11,color="#6b6560"))
        st.plotly_chart(fs2, use_container_width=True)

    divider()

    render_section("Live Metals Market News",
                   "Real-time headlines. Supply disruptions, Fed decisions, and China data "
                   "can move metals stocks 5-20% in a single session.")

    render_info(
        "<strong>How to read metals news:</strong> "
        "Copper supply disruption = bullish FCX/SCCO. "
        "Fed rate cut = bullish NEM/GOLD. "
        "China stimulus = bullish copper/steel. "
        "Strong dollar = bearish all metals. "
        "Steel tariffs = bullish CLF/NUE/X. "
        "High inflation = bullish gold. "
        "Recession fears = bearish copper/steel, bullish gold."
    )

    with st.spinner("Loading news..."):
        news_all = get_metals_news()

    if news_all:
        nn1, nn2 = st.columns(2)
        for i, item in enumerate(news_all):
            with (nn1 if i % 2 == 0 else nn2):
                render_news_card(item)
    else:
        st.info("News feed temporarily unavailable.")

    divider()

    render_info(
        "<strong>Quick Macro Reference:</strong><br>"
        "Rising gold = inflation fears or geopolitical risk increasing.<br>"
        "Rising copper = global growth expectations improving.<br>"
        "Falling steel = construction or manufacturing slowdown.<br>"
        "VIX above 25 = elevated fear — consider reducing exposure.<br>"
        "Dollar weakening = broad commodity tailwind.<br>"
        "China PMI above 50 = industrial demand expanding — bullish FCX, SCCO, AA, CLF.<br>"
        "10Y yields rising sharply = headwind for gold miners NEM and GOLD."
    )