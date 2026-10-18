import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()  # local dev reads TABPFN_TOKEN from .env; Render sets it in the dashboard

try:  # open weights on your own machine: pip install tabpfn
    from tabpfn import TabPFNRegressor  # type: ignore  # optional, not in requirements.txt
    RUNTIME = "local TabPFN weights"
except ImportError:  # hosted: reads TABPFN_TOKEN from the environment
    from tabpfn_client import TabPFNRegressor
    RUNTIME = "TabPFN API"

HORIZON = 7


def daily(df):
    """Sum to one row per day per item, filling days with no sales as 0."""
    wide = df.groupby(["date", "item"])["units"].sum().unstack(fill_value=0).asfreq("D", fill_value=0)
    return wide.stack().rename("units").reset_index()


def featurize(df, items, t0):
    d = df["date"].dt
    return pd.DataFrame({
        "item": pd.Categorical(df["item"], categories=items).codes,
        "dow": d.dayofweek,
        "dom": d.day,
        "month": d.month,
        "t": (df["date"] - t0).dt.days,
    })


def forecast(hist):
    items = sorted(hist["item"].unique())
    t0 = hist["date"].min()
    dates = pd.date_range(hist["date"].max() + pd.Timedelta(1, "D"), periods=HORIZON)
    future = pd.MultiIndex.from_product([dates, items], names=["date", "item"]).to_frame(index=False)
    model = TabPFNRegressor(categorical_features_indices=[0])
    model.fit(featurize(hist, items, t0), hist["units"])
    low, mid, high = model.predict(featurize(future, items, t0), output_type="quantiles", quantiles=[0.1, 0.5, 0.9])
    return future.assign(low=low.clip(0), units=mid.clip(0), high=high.clip(0))


def backtest(hist):
    """Hide the last week, forecast it, and compare with 'same as last week'."""
    cut = hist["date"].max() - pd.Timedelta(HORIZON, "D")
    actual = hist[hist["date"] > cut]
    pred = forecast(hist[hist["date"] <= cut]).merge(actual, on=["date", "item"], suffixes=("_pred", ""))
    last_week = hist.assign(date=hist["date"] + pd.Timedelta(HORIZON, "D"))
    naive = actual.merge(last_week, on=["date", "item"], suffixes=("", "_naive"))
    return (pred["units_pred"] - pred["units"]).abs().mean(), (naive["units_naive"] - naive["units"]).abs().mean()


@st.cache_data(show_spinner="Asking TabPFN about next week...")
def run(hist):
    return forecast(hist), backtest(hist)


st.set_page_config(page_title="Restock", page_icon="📦")
st.title("📦 Restock")
st.write("Upload your past sales. Get next week's demand per item, with a safe amount to prep or order.")

file = st.file_uploader("Sales CSV: one row per sale or per day, with a date, an item, and units sold", type="csv")
if file is None and st.button("Try it with sample bakery data"):
    st.session_state.sample = True
if file is None and not st.session_state.get("sample"):
    st.stop()

raw = pd.read_csv(file if file is not None else "sample_sales.csv")
cols = list(raw.columns)
guess = lambda *names: next((i for i, c in enumerate(cols) if c.lower() in names), 0)
c1, c2, c3 = st.columns(3)
date_col = c1.selectbox("Date column", cols, guess("date", "day", "sold_at"))
item_col = c2.selectbox("Item column", cols, guess("item", "product", "name", "sku"))
units_col = c3.selectbox("Units column", cols, guess("units", "qty", "quantity", "sold"))

try:
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[date_col]).dt.normalize(),
        "item": raw[item_col].astype(str),
        "units": pd.to_numeric(raw[units_col], errors="raise"),
    })
except (ValueError, TypeError) as e:
    st.error(f"Couldn't read those columns: {e}")
    st.stop()

hist = daily(df)
if hist["date"].nunique() < 3 * HORIZON:
    st.error("Need at least 3 weeks of sales history to forecast.")
    st.stop()

fc, (mae, naive_mae) = run(hist)

st.subheader("Next 7 days")
plan = fc.groupby("item")[["units", "high"]].sum().round().astype(int)
plan.columns = ["Expected", "Safe amount (if every day runs busy)"]
st.dataframe(plan, width="stretch")

item = st.selectbox("Day by day for", plan.index)
recent = hist[(hist["item"] == item) & (hist["date"] > hist["date"].max() - pd.Timedelta(28, "D"))]
chart = pd.concat([
    recent.set_index("date")[["units"]].rename(columns={"units": "Sold"}),
    fc[fc["item"] == item].set_index("date")[["units", "high", "low"]]  # column order = legend/color order
      .rename(columns={"units": "Forecast", "low": "Quiet day", "high": "Busy day"}),
])
st.line_chart(chart, color=["#94a3b8", "#2563eb", "#f59e0b", "#14b8a6"])

better = (1 - mae / naive_mae) * 100
source = "your own data" if file is not None else "the sample data (made up, with clean weekly patterns, so real sales will likely score lower)"
st.caption(
    f"Checked on {source}: forecasting the last 7 days without seeing them, TabPFN was off by "
    f"{mae:.1f} units per item per day, vs {naive_mae:.1f} for 'same as last week' "
    f"({abs(better):.0f}% {'better' if better >= 0 else 'worse'}). Runs on {RUNTIME}."
)
