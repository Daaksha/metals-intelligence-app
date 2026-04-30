import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

METALS_TICKERS = {
    "Freeport-McMoRan (FCX) — Copper Mining": "FCX",
    "Newmont Corp (NEM) — Gold Mining":        "NEM",
    "Alcoa Corp (AA) — Aluminum":              "AA",
    "Cleveland-Cliffs (CLF) — Steel":          "CLF",
    "United States Steel (X) — Steel":         "X",
    "Nucor Corp (NUE) — Steel":                "NUE",
    "Southern Copper (SCCO) — Copper":         "SCCO",
    "Barrick Gold (GOLD) — Gold Mining":       "GOLD",
}

COMMODITY_TICKERS = {
    "Gold (GC=F)":      "GC=F",
    "Silver (SI=F)":    "SI=F",
    "Copper (HG=F)":    "HG=F",
    "Crude Oil (CL=F)": "CL=F",
    "Nat. Gas (NG=F)":  "NG=F",
}

MACRO_TICKERS = {
    "S&P 500 (^GSPC)":        "^GSPC",
    "VIX Fear Index (^VIX)":  "^VIX",
    "10Y Treasury (^TNX)":    "^TNX",
    "US Dollar Index (DX-Y)": "DX-Y.NYB",
}


def get_ticker_options():
    return METALS_TICKERS


def get_ticker_symbol(label):
    return METALS_TICKERS[label]


@st.cache_data(ttl=1800)
def load_price_data(symbol, period="6mo"):
    try:
        df = yf.download(symbol, period=period,
                         auto_adjust=False, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def add_technical_indicators(df):
    data = df.copy()
    if data.empty or "Close" not in data.columns:
        return data

    close = data["Close"].squeeze()
    high  = data["High"].squeeze()
    low   = data["Low"].squeeze()
    vol   = data["Volume"].squeeze()

    data["Return_1d"]  = close.pct_change()
    data["Return_5d"]  = close.pct_change(5)
    data["Return_10d"] = close.pct_change(10)
    data["Return_20d"] = close.pct_change(20)

    data["MA_10"]  = close.rolling(10).mean()
    data["MA_20"]  = close.rolling(20).mean()
    data["MA_50"]  = close.rolling(50).mean()
    data["MA_200"] = close.rolling(200).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["EMA_12"]      = ema12
    data["EMA_26"]      = ema26
    data["MACD"]        = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"]   = data["MACD"] - data["MACD_Signal"]

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    data["RSI"] = 100 - (100 / (1 + rs))

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    data["BB_Mid"]   = bb_mid
    data["BB_Upper"] = bb_mid + 2 * bb_std
    data["BB_Lower"] = bb_mid - 2 * bb_std
    data["BB_Width"] = (
        (data["BB_Upper"] - data["BB_Lower"]) / bb_mid.replace(0, np.nan)
    )
    data["BB_Pct"] = (close - data["BB_Lower"]) / (
        data["BB_Upper"] - data["BB_Lower"]
    ).replace(0, np.nan)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    data["ATR"]     = tr.rolling(14).mean()
    data["ATR_Pct"] = data["ATR"] / close.replace(0, np.nan)

    data["Volatility_20d"] = data["Return_1d"].rolling(20).std() * np.sqrt(252)
    data["Volatility_60d"] = data["Return_1d"].rolling(60).std() * np.sqrt(252)

    data["Volume_MA20"]  = vol.rolling(20).mean()
    data["Volume_Ratio"] = vol / data["Volume_MA20"].replace(0, np.nan)

    data["Momentum_10"] = close / close.shift(10) - 1
    data["Momentum_20"] = close / close.shift(20) - 1

    return data


def get_ml_features(df):
    data = add_technical_indicators(df)
    cols = [
        "Return_1d", "Return_5d", "Return_10d", "Return_20d",
        "MACD", "MACD_Signal", "MACD_Hist",
        "RSI", "BB_Width", "BB_Pct",
        "ATR_Pct", "Volatility_20d",
        "Volume_Ratio", "Momentum_10", "Momentum_20",
    ]
    available = [c for c in cols if c in data.columns]
    return data, available


def compute_drawdown(df):
    close = df["Close"].squeeze()
    rolling_max = close.cummax()
    return (close - rolling_max) / rolling_max


def get_beneish_scores(ticker):
    try:
        info = yf.Ticker(ticker).info
        rev  = info.get("totalRevenue") or 0
        gp   = info.get("grossProfits") or 0
        ta   = info.get("totalAssets") or 1
        ni   = info.get("netIncomeToCommon") or 0
        ca   = info.get("currentAssets") or 0
        cl   = info.get("currentLiabilities") or 1
        debt = info.get("totalDebt") or 0
        fcf  = info.get("freeCashflow") or 0

        results = {}

        gm = gp / rev if rev else 0
        results["Gross Margin (%)"]      = round(gm * 100, 2)
        results["GM Flag"]               = "Warning" if gm < 0.20 else "Normal"

        nca = max(ta - ca, 0)
        aqi = nca / ta
        results["Asset Quality Index"]   = round(aqi, 3)
        results["AQI Flag"]              = "Warning" if aqi > 0.75 else "Normal"

        lev = debt / ta
        results["Leverage Ratio"]        = round(lev, 3)
        results["LEV Flag"]              = "Warning" if lev > 0.60 else "Normal"

        npm = ni / rev if rev else 0
        results["Net Profit Margin (%)"] = round(npm * 100, 2)
        results["NPM Flag"]              = "Warning" if npm < 0.03 else "Normal"

        cr = ca / cl
        results["Current Ratio"]         = round(cr, 2)
        results["CR Flag"]               = "Warning" if cr < 1.0 else "Normal"

        fcf_m = fcf / rev if rev else 0
        results["FCF Margin (%)"]        = round(fcf_m * 100, 2)
        results["FCF Flag"]              = "Warning" if fcf_m < 0 else "Normal"

        flags = sum(1 for k, v in results.items()
                    if k.endswith("Flag") and v == "Warning")
        total = sum(1 for k in results if k.endswith("Flag"))
        pct   = flags / total if total else 0
        results["Overall Risk"]    = (
            "High Risk"     if pct >= 0.6 else
            "Moderate Risk" if pct >= 0.3 else
            "Low Risk"
        )
        results["Flags Triggered"] = f"{flags} / {total}"
        return results
    except Exception as e:
        return {"Error": str(e)}


@st.cache_data(ttl=3600)
def load_macro_data(period="1y"):
    frames = {}
    for name, ticker in {**COMMODITY_TICKERS, **MACRO_TICKERS}.items():
        try:
            df = yf.download(ticker, period=period,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns:
                frames[name] = df["Close"].squeeze()
        except Exception:
            pass
    return pd.DataFrame(frames).dropna(how="all") if frames else pd.DataFrame()


@st.cache_data(ttl=3600)
def load_sector_normalised(period="6mo"):
    frames = {}
    for label, ticker in METALS_TICKERS.items():
        try:
            df = yf.download(ticker, period=period,
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty and "Close" in df.columns:
                s = df["Close"].squeeze().dropna()
                frames[ticker] = s / s.iloc[0] * 100
        except Exception:
            pass
    return pd.DataFrame(frames).dropna(how="all") if frames else pd.DataFrame()