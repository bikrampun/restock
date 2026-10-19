# Restock

Upload past sales, get next week's demand per item with a safe amount to prep or order. Built on [TabPFN](https://github.com/PriorLabs/TabPFN), an open-weight foundation model for tabular data.

**Live demo:** https://restock-4nle.onrender.com (click "Try it with sample bakery data"; no file needed).

**How it works:** [Restock on DEV](https://dev.to/bikrampun/restock-next-weeks-order-for-small-shops-from-the-sales-file-they-already-have-5e18), with the architecture diagram and the accuracy check.

Before showing a forecast, it hides your last 7 days, forecasts them, and tells you how far off it was compared with "same as last week".

## Run it

```sh
pip install -r requirements.txt
cp .env.example .env   # then paste your free key from https://platform.priorlabs.ai/account/api-keys
streamlit run app.py
```

Private mode: `pip install tabpfn` and the app runs TabPFN-3.5's downloadable weights on your own machine instead of the API, so sales data stays on the laptop. First accept the non-commercial license at https://platform.priorlabs.ai/accept-license, signed in to the account whose key is in `.env`. The first run downloads the weights once (876 MB); after that it works offline. On Windows, accept the license on the website before running: tabpfn's terminal login prompt crashes there (tabpfn 9.2.0). It runs on your CPU, so it's slower than the API.

CSV format: any columns for date, item and units sold. One row per sale or per day both work. Dates can be `2026-08-01` or day-first `01/08/2026`. At least 3 weeks of history. Optional: a stock-on-hand column (e.g. `stock`, `qty_on_hand`); Restock fills "In stock now" from each item's latest value.

## License

The code is MIT. TabPFN's model weights are under Prior Labs' non-commercial license; commercial use needs a license from them.

## Deploy

`render.yaml` is a Render blueprint. Create a Blueprint from this repo and set `TABPFN_TOKEN`.

`python check.py` runs the whole app headlessly with a stand-in model.

## Changes after the Week 2 deadline

- 2026-10-19: tested private mode end to end (local TabPFN-3.5 weights, offline after the first download) and documented its setup above. README only; the app code is unchanged.
