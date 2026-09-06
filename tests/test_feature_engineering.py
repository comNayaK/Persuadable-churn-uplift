import pytest
import pandas as pd
import numpy as np
from persuadable_churn.data.loader import DataLoader
from persuadable_churn.data.feature_engineering import FeatureEngineer


def test_feature_engineer(tmp_path):
    loader = DataLoader(data_dir=tmp_path, random_state=42)
    df = loader._synthesize_telco_base(n_samples=200)
    df["treatment"] = np.random.binomial(1, 0.5, size=200)
    df["churn"] = np.random.binomial(1, 0.3, size=200)

    fe = FeatureEngineer()
    X = fe.fit_transform(df)

    # Must have 25+ features
    assert X.shape[1] >= 25
    assert len(X) == 200

    # Must contain key interaction terms and buckets
    assert "tenure_bucket_0_6m" in X.columns
    assert "fiber_no_techsupport" in X.columns
    assert "addon_count" in X.columns
    assert "is_electronic_check" in X.columns

    # Must exclude target/metadata columns
    assert "customerID" not in X.columns
    assert "churn" not in X.columns
    assert "Churn" not in X.columns
    assert "treatment" not in X.columns

    # No NaNs or non-numeric types
    assert not X.isna().any().any()
    assert np.issubdtype(X.values.dtype, np.number)
