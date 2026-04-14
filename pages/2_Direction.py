import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

@st.cache_data(ttl=3600)
def load_data(symbol: str) -> pd.DataFrame:
    df = yf.download(symbol, period="2y", auto_adjust=False, progress=False)

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df.dropna()

df = load_data(ticker)
if df.empty:
    st.error("No data returned for this ticker.")
    st.stop()

st.title("Direction Prediction")
st.write("Predict whether the selected metals stock may move up or down next day.")

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


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data["Return_1d"] = data["Close"].pct_change(1)
    data["Return_5d"] = data["Close"].pct_change(5)
    data["MA_5"] = data["Close"].rolling(5).mean()
    data["MA_20"] = data["Close"].rolling(20).mean()
    data["Volatility_5"] = data["Return_1d"].rolling(5).std()
    data["MA_Signal"] = data["MA_5"] - data["MA_20"]

    data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

    data = data.dropna()
    return data


df = load_data(ticker)
data = prepare_features(df)

feature_cols = ["Return_1d", "Return_5d", "MA_5", "MA_20", "Volatility_5", "MA_Signal"]
X = data[feature_cols]
y = data["Target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

latest_features = X.iloc[[-1]]
latest_prediction = model.predict(latest_features)[0]
latest_probability = model.predict_proba(latest_features)[0][1]

prediction_label = "UP" if latest_prediction == 1 else "DOWN"

col1, col2, col3 = st.columns(3)
col1.metric("Ticker", ticker)
col2.metric("Prediction", prediction_label)
col3.metric("Probability of UP", f"{latest_probability:.2%}")

st.subheader("Model Accuracy")
st.write(f"Test Accuracy: {accuracy:.2%}")

st.subheader("Latest Feature Snapshot")
st.dataframe(latest_features, use_container_width=True)

st.subheader("Recent Test Predictions")
results = X_test.copy()
results["Actual"] = y_test.values
results["Predicted"] = y_pred
st.dataframe(results.tail(10), use_container_width=True)

with st.expander("Classification Report"):
    report = classification_report(y_test, y_pred)
    st.text(report)