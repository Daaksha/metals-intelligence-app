
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.title("Risk and Volatility")
st.write("Measure short-term volatility and downside risk for the selected metals stock.")

METALS_TICKERS = {
    "Freeport-McMoRan (FCX)": "FCX",
    "Newmont (NEM)": "NEM",
    "Alcoa (AA)": "AA",
    "Cleveland-Cliffs (CLF)": "CLF",
    "United States Steel (X)": "X",
}

selected_label = st.selectbox("Select a Metals Stock", list(METALS_TICKERS.keys()))
ticker = METALS_TICKERS[selected_label]


@st.cache_data(ttl=3600)
def load_data(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="2y", auto_adjust=False, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna()


df = load_data(ticker)

df["Return_1d"] = df["Close"].pct_change()
df["Rolling_Vol_20"] = df["Return_1d"].rolling(20).std() * np.sqrt(252)
df["Rolling_Vol_60"] = df["Return_1d"].rolling(60).std() * np.sqrt(252)

returns = df["Return_1d"].dropna()

latest_vol_20 = df["Rolling_Vol_20"].dropna().iloc[-1]
latest_vol_60 = df["Rolling_Vol_60"].dropna().iloc[-1]
var_95 = np.percentile(returns, 5)
avg_return = returns.mean()

if latest_vol_20 < 0.25:
    risk_level = "Low"
elif latest_vol_20 < 0.45:
    risk_level = "Medium"
else:
    risk_level = "High"

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ticker", ticker)
col2.metric("20-Day Annualized Volatility", f"{latest_vol_20:.2%}")
col3.metric("60-Day Annualized Volatility", f"{latest_vol_60:.2%}")
col4.metric("Risk Level", risk_level)

st.subheader("Downside Risk")
st.write(f"Average Daily Return: {avg_return:.2%}")
st.write(f"95% Daily Value at Risk (VaR): {var_95:.2%}")

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Rolling_Vol_20"],
        name="20-Day Volatility"
    )
)

fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["Rolling_Vol_60"],
        name="60-Day Volatility"
    )
)

fig.update_layout(
    title=f"{ticker} Rolling Volatility",
    xaxis_title="Date",
    yaxis_title="Annualized Volatility",
    height=500
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Recent Risk Data")
risk_table = df[["Close", "Return_1d", "Rolling_Vol_20", "Rolling_Vol_60"]].dropna().tail(10)
st.dataframe(risk_table, use_container_width=True)