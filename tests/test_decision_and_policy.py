import pytest
import numpy as np
import pandas as pd
from persuadable_churn.config import CampaignEconomics, QuadrantThresholds
from persuadable_churn.decision.quadrant import QuadrantEngine, QuadrantLabel
from persuadable_churn.decision.policy import PolicySimulator


def test_quadrant_engine():
    n = 1000
    # Simulate realistic risk and uplift distributions
    churn_risk = np.random.uniform(0.05, 0.95, size=n)
    uplift_score = np.random.normal(0.03, 0.08, size=n)
    # Inject a realistic 3% sleeping dog tail
    uplift_score[:30] = np.random.uniform(-0.15, -0.05, size=30)
    charges = np.random.uniform(30, 110, size=n)

    engine = QuadrantEngine()
    df_seg = engine.segment_customers(churn_risk, uplift_score, charges)

    assert len(df_seg) == n
    assert "quadrant_label" in df_seg.columns
    assert "recommended_action" in df_seg.columns

    # Check that all 4 quadrants are populated
    labels = set(df_seg["quadrant_label"].unique())
    assert QuadrantLabel.PERSUADABLE.value in labels
    assert QuadrantLabel.SURE_THING.value in labels
    assert QuadrantLabel.LOST_CAUSE.value in labels
    assert QuadrantLabel.DO_NOT_DISTURB.value in labels

    # Sleeping dogs must realistically be 1% - 6%
    dog_pct = (df_seg["quadrant_label"] == QuadrantLabel.DO_NOT_DISTURB.value).mean()
    assert 0.01 <= dog_pct <= 0.08


def test_policy_simulator():
    n = 500
    churn_risk = np.random.uniform(0.1, 0.9, size=n)
    uplift_score = np.random.normal(0.05, 0.06, size=n)
    df = pd.DataFrame({
        "churn_risk": churn_risk,
        "uplift_score": uplift_score,
        "quadrant_label": np.random.choice(["Persuadable", "Sure Thing", "Lost Cause", "Do-Not-Disturb"], size=n),
        "MonthlyCharges": np.full(n, 65.0),
        "true_tau_churn": uplift_score
    })

    sim = PolicySimulator()
    results = sim.simulate_policies(df, campaign_budget=1000.0, contact_cost=1.50)

    assert "standard_ml" in results
    assert "causal_uplift" in results
    assert "dual_layer" in results

    # Audit checks
    assert results["dual_layer"].sleeping_dogs_contacted == 0
    assert results["dual_layer"].lost_causes_contacted == 0
