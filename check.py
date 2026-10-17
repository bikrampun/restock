"""Runs the whole app headlessly on the sample data with a stand-in model (no API token needed)."""
import sys
import types

import numpy as np
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
assert "units per item per day" in at.caption[0].value, at.caption[0].value
print(plan, at.caption[0].value, sep="\n\n")
