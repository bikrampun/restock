# Restock

Upload past sales, get next week's demand per item with a safe amount to prep or order. Built on [TabPFN](https://github.com/PriorLabs/TabPFN), an open-weight foundation model for tabular data.

Before showing a forecast, it hides your last 7 days, forecasts them, and tells you how far off it was compared with "same as last week".

## Run it

```sh
pip install -r requirements.txt
cp .env.example .env   # then paste your free key from https://platform.priorlabs.ai/account/api-keys
streamlit run app.py
```

Fully private mode: `pip install tabpfn` and the app runs the open weights on your own machine instead of calling the API. No sales data leaves the laptop.

CSV format: any columns for date, item and units sold. One row per sale or per day both work.

## Deploy

`render.yaml` is a Render blueprint. Create a Blueprint from this repo and set `TABPFN_TOKEN`.

`python check.py` runs the whole app headlessly with a stand-in model.
