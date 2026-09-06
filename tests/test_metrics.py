import pytest
import numpy as np
import pandas as pd
from persuadable_churn.evaluation.metrics import UpliftEvaluator


def test_qini_curve_calculation():
    np.random.seed(42)
    n = 600
    w = np.random.binomial(1, 0.5, size=n)
    # Strong positive uplift: treated units retain much more than control
    true_tau = np.random.uniform(0.05, 0.30, size=n)
    retained = np.where(w == 1, np.random.binomial(1, 0.75, size=n), np.random.binomial(1, 0.45, size=n))
    churn = 1 - retained
    # Uplift model with good predictive ranking
    pred_uplift = true_tau + np.random.normal(0, 0.05, size=n)

    res = UpliftEvaluator.compute_qini_curve(churn, w, pred_uplift, n_bins=50)

    # Qini curve should be positive and above random
    assert res.qini_score > 0.05
    assert res.auuc > 0.0
    assert len(res.fractions) == 50
    assert len(res.qini_values) == 50


def test_decile_table():
    np.random.seed(42)
    n = 500
    w = np.random.binomial(1, 0.5, size=n)
    retained = np.random.binomial(1, 0.6, size=n)
    pred_uplift = np.random.normal(0.05, 0.05, size=n)

    dec_df = UpliftEvaluator.compute_decile_table(1 - retained, w, pred_uplift)

    assert len(dec_df) == 10
    assert "Decile" in dec_df.columns
    assert "Incremental Uplift" in dec_df.columns
    assert "Treated Retain %" in dec_df.columns
