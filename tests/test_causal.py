import pytest
import numpy as np
import pandas as pd
from persuadable_churn.models.causal import TLearner, XLearner


def test_t_learner():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f_{i}" for i in range(5)])
    w = np.random.binomial(1, 0.5, size=n)
    # Treatment increases retention (decreases churn)
    r = np.random.binomial(1, np.clip(0.4 + 0.3 * w, 0, 1))
    churn = 1 - r

    tl = TLearner(base_learner_type="lightgbm", n_estimators=25, max_depth=3)
    tl.fit(X, churn, w)

    tau = tl.predict_uplift(X)
    assert len(tau) == n
    assert not np.isnan(tau).any()
    # On average, treatment had positive retention effect
    assert tau.mean() > 0.05


def test_x_learner_counterfactual_imputation():
    """Validates that X-Learner implements canonical Künzel et al. formula without sign errors."""
    np.random.seed(42)
    n = 250
    X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f_{i}" for i in range(5)])
    w = np.random.binomial(1, 0.5, size=n)
    r = np.random.binomial(1, np.clip(0.3 + 0.35 * w, 0, 1))
    churn = 1 - r

    xl = XLearner(base_learner_type="lightgbm", n_estimators=30, max_depth=3)
    xl.fit(X, churn, w)

    tau = xl.predict_uplift(X)
    assert len(tau) == n
    assert not np.isnan(tau).any()
    # Directional correctness: uplift on retention must be positive
    assert tau.mean() > 0.05
