import pytest
import numpy as np
import pandas as pd
from persuadable_churn.models.baseline import BaselineChurnModel


def test_baseline_churn_model():
    np.random.seed(42)
    n = 300
    p = 10
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"feat_{i}" for i in range(p)])
    # Binary target with strong linear component
    logits = X["feat_0"] * 1.5 - X["feat_1"] * 1.2 + np.random.randn(n) * 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)

    model = BaselineChurnModel(model_type="lightgbm", n_estimators=30, max_depth=3)
    cv_res = model.fit_cv(X, y, cv_folds=3)

    assert cv_res["cv_auc_mean"] >= 0.75
    assert len(model.predict_churn_risk(X)) == n

    # Probabilities must be strictly bounded in [0, 1]
    churn_probs = model.predict_churn_risk(X)
    assert np.all((churn_probs >= 0.0) & (churn_probs <= 1.0))

    # Feature importances
    imp_df = model.get_feature_importances()
    assert len(imp_df) == p
    assert "feature" in imp_df.columns
    assert "importance" in imp_df.columns
