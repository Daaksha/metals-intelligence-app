import streamlit as st
import plotly.graph_objects as go
from src.data_loader import get_ticker_options, get_ticker_symbol, load_price_data
from src.ui import set_app_style, render_title, render_section_header

st.set_page_config(page_title="Overview", layout="wide")
set_app_style()

render_title(
    "Overview",
    "Live market snapshot for the selected metals stock."
)

ticker_options = get_ticker_options()
labels = list(ticker_options.keys())

selected_label = st.selectbox(
    "Select a Metals Stock",
    labels
)

if not selected_label:
    st.stop()

ticker = get_ticker_symbol(selected_label)

period = st.selectbox(
    "Select time period",
    ["1mo", "3mo", "6mo", "1y"],
    index=2
)

data = load_price_data(ticker, period)

if data.empty:
    st.error("No data returned for this ticker.")
    st.stop()

latest_close = float(data["Close"].iloc[-1])
prev_close = float(data["Close"].iloc[-2]) if len(data) > 1 else latest_close
daily_change = latest_close - prev_close
daily_change_pct = (daily_change / prev_close * 100) if prev_close != 0 else 0
period_high = float(data["High"].max())
avg_volume = float(data["Volume"].mean())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", ticker)
col2.metric("Latest Close", f"${latest_close:,.2f}", f"{daily_change_pct:.2f}%")
col3.metric("Period High", f"${period_high:,.2f}")
col4.metric("Average Volume", f"{avg_volume:,.0f}")

render_section_header("Price Chart")

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

render_section_header("Recent Data")
st.dataframe(data.tail(10), use_container_width=True)