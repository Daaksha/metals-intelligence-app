import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.title("Overview")
st.write("This page will show the summary of the selected metals stock.")

ticker = st.selectbox(
    "Select a Metals Stock",
    ["FCX", "NEM", "AA", "CLF", "X"]
)

data = yf.download(ticker, period="6mo")

st.write(f"Showing data for {ticker}")

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Close Price"
    )
)

fig.update_layout(
    title=f"{ticker} Price Chart",
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig)

st.subheader("Raw Data")
st.dataframe(data.tail())