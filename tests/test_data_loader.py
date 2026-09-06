import pytest
import pandas as pd
import numpy as np
from persuadable_churn.data.loader import DataLoader, SyntheticTrialGenerator


def test_synthetic_trial_generator():
    generator = SyntheticTrialGenerator(random_state=42)
    # Create minimal mock dataframe with telco features
    df = pd.DataFrame({
        "customerID": ["001-A", "002-B", "003-C", "004-D", "005-E"] * 50,
        "tenure": np.random.randint(1, 72, size=250),
        "MonthlyCharges": np.random.uniform(20, 110, size=250),
        "TotalCharges": np.random.uniform(100, 5000, size=250),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], size=250),
        "InternetService": np.random.choice(["Fiber optic", "DSL", "No"], size=250),
        "TechSupport": np.random.choice(["Yes", "No"], size=250),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Credit card (automatic)"], size=250),
        "PaperlessBilling": np.random.choice(["Yes", "No"], size=250),
        "SeniorCitizen": np.random.choice([0, 1], size=250)
    })

    trial_df = generator.generate_trial(df, is_rct=True, sleeping_dog_rate=0.03)

    assert "treatment" in trial_df.columns
    assert "churn" in trial_df.columns
    assert "retained" in trial_df.columns
    assert "true_tau_churn" in trial_df.columns
    assert len(trial_df) == 250
    # Check retention = 1 - churn
    assert np.all(trial_df["retained"] == 1 - trial_df["churn"])
    # Sleeping dog rate should be realistically bounded
    assert 0.0 <= trial_df["is_sleeping_dog_true"].mean() <= 0.08


def test_data_loader_synthesize(tmp_path):
    loader = DataLoader(data_dir=tmp_path, random_state=42)
    telco_df = loader._synthesize_telco_base(n_samples=100)
    assert len(telco_df) == 100
    assert "customerID" in telco_df.columns
    assert "MonthlyCharges" in telco_df.columns
    assert "Contract" in telco_df.columns

    hillstrom_df = loader._synthesize_hillstrom_base(n_samples=100)
    assert len(hillstrom_df) == 100
    assert "segment" in hillstrom_df.columns
    assert "conversion" in hillstrom_df.columns
