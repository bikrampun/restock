# Restock

Upload past sales, get next week's demand per item with a safe amount to prep or order. Built on [TabPFN](https://github.com/PriorLabs/TabPFN), an open-weight foundation model for tabular data.

**Live demo:** https://restock-4nle.onrender.com (free instance: the first visit can take up to a minute to wake up; click "Try it with sample bakery data").

Before showing a forecast, it hides your last 7 days, forecasts them, and tells you how far off it was compared with "same as last week".

## Run it

```sh
pip install -r requirements.txt
cp .env.example .env   # then paste your free key from https://platform.priorlabs.ai/account/api-keys
streamlit run app.py
```

Private mode: if the `tabpfn` package is installed, the app uses TabPFN's downloadable weights on your own machine instead of the API, so sales data stays on the laptop. Not yet tested end to end; the weights need a license accepted at platform.priorlabs.ai.

CSV format: any columns for date, item and units sold. One row per sale or per day both work. Dates can be `2026-08-01` or day-first `01/08/2026`. At least 3 weeks of history. Optional: a stock-on-hand column (e.g. `stock`, `qty_on_hand`); Restock fills "In stock now" from each item's latest value.

## License

The code is MIT. TabPFN's model weights are under Prior Labs' non-commercial license; commercial use needs a license from them.

## Deploy

`render.yaml` is a Render blueprint. Create a Blueprint from this repo and set `TABPFN_TOKEN`.

`python check.py` runs the whole app headlessly with a stand-in model.
