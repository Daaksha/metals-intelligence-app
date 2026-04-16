import pandas as pd
import yfinance as yf

METALS_TICKERS = {
    "Freeport-McMoRan (FCX)": "FCX",
    "Newmont (NEM)": "NEM",
    "Alcoa (AA)": "AA",
    "Cleveland-Cliffs (CLF)": "CLF",
    "United States Steel (X)": "X",
}

def get_ticker_options():
    return METALS_TICKERS

def get_ticker_symbol(label):
    return METALS_TICKERS[label]

def load_price_data(symbol, period="6mo"):
    df = yf.download(symbol, period=period, auto_adjust=False, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()

def add_basic_features(df):
    data = df.copy()

    if data.empty:
        return data

    data["Return_1d"] = data["Close"].pct_change()
    data["Return_5d"] = data["Close"].pct_change(5)
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Volatility_5"] = data["Return_1d"].rolling(5).std()

    return data