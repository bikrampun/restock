"""Runs the whole app headlessly on the sample data with a stand-in model (no API token needed)."""
import sys
import types

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


class FakeRegressor:  # predicts each weekday's average, so the plumbing is tested, not TabPFN
    def __init__(self, **_): pass

    def fit(self, X, y):
        self.means = y.groupby([X["item"].values, X["dow"].values]).mean()

    def predict(self, X, output_type, quantiles):
        mid = np.array([self.means[(i, d)] for i, d in zip(X["item"], X["dow"])])
        return [mid * 0.8, mid, mid * 1.2]


sys.modules["tabpfn"] = types.SimpleNamespace(TabPFNRegressor=FakeRegressor)
at = AppTest.from_file("app.py", default_timeout=60).run()
at.button[0].click().run()
assert not at.exception and not at.error, (at.exception, at.error)
plan = at.dataframe[0].value
assert len(plan) == 5 and (plan.iloc[:, 1] >= plan.iloc[:, 0]).all(), plan
accuracy = next(c.value for c in at.caption if "units per item per day" in c.value)
assert "Order to cover" in at.code[0].value, at.code[0].value  # stock defaults to 0, so everything is ordered
print(plan, accuracy, at.code[0].value, sep="\n\n")

from app import order_message, parse_dates, skip_items  # noqa: E402  (importing reruns the app once with the fake model)

days = pd.date_range("2026-08-01", periods=42)
sold = {"Bread": [1] * 42, "New": [0] * 30 + [1] * 12, "Gone": [1] * 20 + [0] * 22}
too_new, stopped = skip_items(pd.DataFrame(
    [(d, item, u) for item, units in sold.items() for d, u in zip(days, units)], columns=["date", "item", "units"]))
assert list(too_new) == ["New"] and list(stopped) == ["Gone"], (too_new, stopped)

safe = pd.Series({"Bread": 10, "Milk": 5})
week = pd.Timestamp("2026-10-19"), pd.Timestamp("2026-10-25")
msg = order_message(safe, pd.Series({"Bread": 4, "Milk": 9}), *week)
assert msg == "Order to cover Mon 19 Oct to Sun 25 Oct:\n- Bread: 6", msg  # Milk has enough, so it's left out
assert order_message(safe, safe, *week) is None

day_first = parse_dates(pd.Series(["01/08/2026", "12/08/2026", "13/08/2026"]))
assert day_first.dt.strftime("%Y-%m-%d").tolist() == ["2026-08-01", "2026-08-12", "2026-08-13"], day_first
us = parse_dates(pd.Series(["08/01/2026", "08/13/2026"]))
assert us.dt.strftime("%Y-%m-%d").tolist() == ["2026-08-01", "2026-08-13"], us
