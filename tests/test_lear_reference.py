"""LEAR must reproduce the reference code of the paper (epftoolbox).

The expected forecasts in `data/lear_reference.npz` were produced by the estimation steps of
epftoolbox's `LEAR.recalibrate` and `LEAR.predict` (asinh-median scaling, `LassoLarsIC(criterion="aic")`
for the penalty, then `Lasso`), run under scikit-learn 0.24.2, the version family the paper used.
Two synthetic cases: fewer training days than inputs (as in the 56- and 84-day windows) and more.
"""
from pathlib import Path

import numpy as np
import pytest

from src.benchmarks import lear_one_window

DATA = np.load(Path(__file__).parent / "data" / "lear_reference.npz")


@pytest.mark.parametrize("case", ["short", "long"])
def test_lear_matches_reference_code(case):
    pred = lear_one_window(*(DATA[f"{case}_{k}"] for k in ("Xt", "Dt", "Yt", "Xp", "Dp")))
    np.testing.assert_allclose(pred, DATA[f"{case}_expected"], atol=1e-3)
