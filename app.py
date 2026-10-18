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


def parse_dates(s):
    try:
        return pd.to_datetime(s)
    except ValueError:  # e.g. 13/08/2026: day-first, common outside the US
        return pd.to_datetime(s, dayfirst=True)


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


def skip_items(hist):
    """Items too new to forecast (under 3 weeks of sales) and items that stopped selling (none in 2 weeks)."""
    last_day = hist["date"].max()
    seen = hist[hist["units"] > 0].groupby("item")["date"].agg(["min", "max"])
    return (seen.index[seen["min"] > last_day - pd.Timedelta(3 * HORIZON, "D")],
            seen.index[seen["max"] <= last_day - pd.Timedelta(14, "D")])


def order_message(safe, stock, start, end):
    """Order = safe amount minus what's in stock, per item. None means fully stocked."""
    order = (safe - stock)[safe > stock]
    lines = "\n".join(f"- {item}: {qty}" for item, qty in order.items())
    return f"Order to cover {start:%a %d %b} to {end:%a %d %b}:\n{lines}" if lines else None


@st.cache_data(max_entries=20, ttl="1h", show_spinner="Asking TabPFN about next week...")
def run(hist):
    return forecast(hist), backtest(hist)


st.set_page_config(page_title="Restock", page_icon="📦")
st.title("📦 Restock")
st.write("Upload your past sales. Get next week's demand per item, with a safe amount to prep or order.")

st.markdown("""<style>  /* a big drop target: the default upload box is a thin bar */
[data-testid="stFileUploaderDropzone"] { border: 2px dashed #2563eb; background: rgba(37, 99, 235, .08); }
[data-testid="stFileUploaderDropzone"]:has([data-testid="stFileUploaderDropzoneInstructions"]) {
    flex-direction: column; justify-content: center; align-items: center; gap: .75rem; min-height: 12rem;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    order: -1; display: flex; flex-direction: column; align-items: center; margin: 0;
}
[data-testid="stFileUploaderDropzoneInstructions"]::before {  /* Lucide "cloud-upload" icon (ISC license) */
    order: -2; content: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='56' height='56' viewBox='0 0 24 24' fill='none' stroke='%232563eb' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'><path d='M12 13v8'/><path d='M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242'/><path d='m8 17 4-4 4 4'/></svg>");
}
[data-testid="stFileUploaderDropzoneInstructions"]::after {
    content: "Drop your sales CSV here or click Upload"; font-size: 1.1rem; order: -1;
}
</style>""", unsafe_allow_html=True)
file = st.file_uploader("Sales CSV: one row per sale or per day, with a date, an item, and units sold", type="csv")
if RUNTIME == "TabPFN API":
    st.caption("Your daily sales per item are sent to Prior Labs' TabPFN API to make the forecast; "
               "Prior Labs may cache them briefly and never trains on customer data. "
               "Restock never saves your file; it stays in memory for at most an hour.")
if file is None and st.button("Try it with sample bakery data"):
    st.session_state.sample = True
if file is None and not st.session_state.get("sample"):
    st.stop()

raw = pd.read_csv(file if file is not None else "sample_sales.csv")
cols = list(raw.columns)
guess = lambda *names: next((i for i, c in enumerate(cols) if c.lower() in names), 0)
c1, c2, c3, c4 = st.columns(4)
date_col = c1.selectbox("Date column", cols, guess("date", "day", "sold_at"))
item_col = c2.selectbox("Item column", cols, guess("item", "product", "name", "sku"))
units_col = c3.selectbox("Units column", cols, guess("units", "qty", "quantity", "sold"))
stock_col = c4.selectbox("Stock column (optional)", ["(none)"] + cols,  # some POS exports carry stock on hand
    next((i + 1 for i, c in enumerate(cols) if c.lower() in ("stock", "on_hand", "qty_on_hand", "stock_on_hand", "closing_stock", "soh")), 0))

picked = [date_col, item_col, units_col] + ([stock_col] if stock_col != "(none)" else [])
if len(set(picked)) < len(picked):
    st.error("Pick a different column for each box: a date, an item, units sold (and stock, if your file has it).")
    st.stop()

blank = raw[[date_col, item_col, units_col]].isna().any(axis=1)  # a blank item would become an item called "nan"
if blank.any():  # spreadsheet row = index + 2: the header is row 1 and pandas counts from 0
    rows = (raw.index[blank] + 2).tolist()
    shown = ", ".join(map(str, rows[:5])) + (f" and {len(rows) - 5} more" if len(rows) > 5 else "")
    s = "s" * (len(rows) > 1)
    st.info(f"Skipped {len(rows)} row{s} with a missing date, item or units (row{s} {shown}).")
    raw = raw[~blank]

try:
    df = pd.DataFrame({
        "date": parse_dates(raw[date_col]).dt.normalize(),
        "item": raw[item_col].astype(str).str.strip(),
        "units": pd.to_numeric(raw[units_col], errors="raise"),
    })
    if stock_col != "(none)":  # an unreadable stock cell counts as missing; units still has to be clean
        df["stock"] = pd.to_numeric(raw[stock_col], errors="coerce")
except (ValueError, TypeError) as e:  # name the spreadsheet row: pandas' "position" skips the header and dedupes dates
    coerce = lambda **kw: pd.to_datetime(raw[date_col], errors="coerce", **kw)
    bad_date = raw[date_col][coerce().fillna(coerce(dayfirst=True)).isna()]
    bad_units = raw[units_col][pd.to_numeric(raw[units_col], errors="coerce").isna()]
    bad, what = (bad_date, "a date") if len(bad_date) else (bad_units, "a number")
    where = f"Row {bad.index[0] + 2}: \"{bad.iloc[0]}\" isn't {what}." if len(bad) else f"({str(e).splitlines()[0]})"
    st.error(f"Couldn't read your file. {where} Dates should look like 2026-08-01 or 01/08/2026, units should be numbers.")
    st.stop()

today, last = pd.Timestamp.today().normalize(), df["date"].max()
if file is None:  # keep the demo current: move the sample forward by whole weeks, so weekdays still line up
    df["date"] += pd.Timedelta(7 * ((today - pd.Timedelta(1, "D") - last).days // 7), "D")
elif (today - last).days > 2:  # forecast starts the day after the last sale; warn once most of it isn't ahead
    passed = min((today - last).days - 1, HORIZON)
    st.warning(f"Your sales end on {last:%a %d %b}, so {'all 7' if passed == HORIZON else f'{passed} of the 7'} forecast days "
               "have already passed. Upload newer sales for next week's order.")

# "croissant" and "Croissant" are one item: use its most common spelling
df["item"] = df["item"].groupby(df["item"].str.casefold()).transform(lambda s: s.mode()[0])
hist = daily(df)
if hist["date"].nunique() < 3 * HORIZON:
    st.error("Need at least 3 weeks of sales history to forecast.")
    st.stop()

too_new, stopped = skip_items(hist)
if len(too_new):
    st.info(f"Too new to forecast (under 3 weeks of sales): {', '.join(too_new)}")
if len(stopped):
    st.info(f"Left out, no sales in the last 2 weeks: {', '.join(stopped)}")
hist = hist[~hist["item"].isin(too_new.union(stopped))]
if hist.empty:
    st.warning("Nothing left to forecast. Restock needs items with at least 3 weeks of sales, including some in the last 2 weeks.")
    st.stop()

fc, (mae, naive_mae) = run(hist)

st.subheader("Next 7 days")
plan = fc.groupby("item")[["units", "high"]].sum().round().astype(int)
plan.columns = ["Expected", "Safe amount"]
st.table(plan)  # plain table: nothing to click, so it doesn't look editable
st.caption("Safe amount: enough if every day runs busy.")

st.subheader("What to order")
if "stock" in df:  # the latest row's stock, blank or not, never an older count (older = too high = run out)
    latest = df.sort_values("date", kind="stable").groupby("item").tail(1)
    on_hand = latest.set_index("item")["stock"].reindex(plan.index)
    st.caption("✏️ Filled in from your file's latest stock. Change any number in **In stock now**, then press Enter.")
    missing = latest[latest["stock"].isna() & latest["item"].isin(plan.index)]
    if len(missing):
        st.warning("Stock is blank or not a number on the latest row for "
                   + ", ".join(f"{item} (row {i + 2})" for i, item in missing["item"].items())
                   + ". It shows 0 below; type the real count.")
else:
    on_hand = pd.Series(index=plan.index, dtype=float)
    st.caption("✏️ Type how many you have on the shelf now in **In stock now**, then press Enter. The order below updates.")
stock = st.data_editor(
    pd.DataFrame({"In stock now": on_hand.fillna(0).clip(lower=0).round().astype(int)}),
    column_config={"In stock now": st.column_config.NumberColumn(min_value=0, step=1)},
    width="stretch",
)["In stock now"].fillna(0).astype(int)  # a cleared cell comes back empty
start, end = fc["date"].min(), fc["date"].max()
message = order_message(plan["Safe amount"], stock, start, end)
if message:
    st.code(message, language=None)
else:
    st.success(f"Fully stocked for {start:%a %d %b} to {end:%a %d %b}, nothing to order.")

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
