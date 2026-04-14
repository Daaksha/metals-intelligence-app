import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go

@st.cache_data(ttl=3600)
def load_price_data(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=False, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()
st.set_page_config(page_title="Overview", layout="wide")

st.title("Overview")
st.write("Live market snapshot for selected metals-related stock.")

METALS_TICKERS = {
    "Freeport-McMoRan (FCX)": "FCX",
    "Newmont (NEM)": "NEM",
    "Alcoa (AA)": "AA",
    "Cleveland-Cliffs (CLF)": "CLF",
    "United States Steel (X)": "X",
}

selected_label = st.selectbox("Select a Metals Stock", list(METALS_TICKERS.keys()))
ticker = METALS_TICKERS[selected_label]

period = st.selectbox("Select time period", ["1mo", "3mo", "6mo", "1y"], index=2)

@st.cache_data(ttl=3600)
def load_price_data(symbol: str, period: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()

data = load_price_data(ticker, period)

if data.empty:
    st.error("No data returned for this ticker. Try another one.")
    st.stop()

latest_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_close
daily_change = latest_close - prev_close
daily_change_pct = (daily_change / prev_close * 100) if prev_close != 0 else 0

high_52_proxy = float(data["High"].max())
low_52_proxy = float(data["Low"].min())
avg_volume = float(data["Volume"].mean())

col1, col2, col3, col4 = st.columns(4)

col1.metric("Ticker", ticker)
col2.metric("Latest Close", f"${latest_close:,.2f}", f"{daily_change_pct:.2f}%")
col3.metric("Period High", f"${high_52_proxy:,.2f}")
col4.metric("Average Volume", f"{avg_volume:,.0f}")

fig = go.Figure()

fig.add_trace(
    go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Price"
    )
)

fig.update_layout(
    title=f"{ticker} Price Chart",
    xaxis_title="Date",
    yaxis_title="Price",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Data")
st.dataframe(data.tail(10), use_container_width=True)

csv = data.to_csv().encode("utf-8")
st.download_button(
    label="Download price data as CSV",
    data=csv,
    file_name=f"{ticker}_{period}_prices.csv",
    mime="text/csv"
)