"""
Dual-Layer Quadrant Decision Engine.

Segments customers into 4 behavioral quadrants using data-driven quantile distributions
and unit-economic break-even bounds:
1. Persuadables: High positive uplift (Target with 10% discount).
2. Sure Things: Low baseline risk, low/no uplift (Suppress to avoid revenue cannibalization).
3. Lost Causes: High baseline risk, low/no uplift (Suppress to avoid wasted budget).
4. Do-Not-Disturb / Sleeping Dogs: Negative uplift (Strictly suppress; outreach triggers churn).
   Calibrated strictly to a realistic 2% - 4% of customer accounts.
"""

from enum import Enum
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from ..config import CampaignEconomics, QuadrantThresholds


class QuadrantLabel(str, Enum):
    PERSUADABLE = "Persuadable"
    SURE_THING = "Sure Thing"
    LOST_CAUSE = "Lost Cause"
    DO_NOT_DISTURB = "Do-Not-Disturb"


class QuadrantEngine:
    """
    Automated decision engine that maps dual-layer outputs:
    (Baseline Churn Risk, Causal Uplift) -> Behavioral Quadrant & Campaign Action Routing.
    """

    def __init__(
        self,
        thresholds: Optional[QuadrantThresholds] = None,
        economics: Optional[CampaignEconomics] = None
    ):
        self.thresholds = thresholds or QuadrantThresholds()
        self.economics = economics or CampaignEconomics()

    def segment_customers(
        self,
        churn_risk: np.ndarray,
        uplift_score: np.ndarray,
        monthly_charges: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Segments customers into the 4 behavioral quadrants.

        Parameters:
        - churn_risk: 1D array of P(Churn | X) in [0, 1]
        - uplift_score: 1D array of tau(X) (churn reduction uplift: positive = retains user)
        - monthly_charges: optional array of customer monthly bills for economic break-even

        Returns:
        DataFrame with columns: ['churn_risk', 'uplift_score', 'quadrant_label', 'recommended_action', 'is_targetable']
        """
        n = len(churn_risk)
        churn_risk = np.asarray(churn_risk, dtype=float)
        uplift_score = np.asarray(uplift_score, dtype=float)

        # 1. Determine Dynamic Quadrant Thresholds
        if self.thresholds.use_quantiles:
            # Persuadable threshold: top percentile of uplift distribution
            tau_persuadable_cutoff = float(np.percentile(
                uplift_score, 100.0 * self.thresholds.persuadable_quantile
            ))
            # Bound cutoff so it's strictly positive
            tau_persuadable_cutoff = max(0.02, tau_persuadable_cutoff)

            # Sleeping Dog threshold: bottom percentile of uplift distribution
            # Calibrated to capture 2% - 4% negative uplift accounts
            tau_dog_cutoff = float(np.percentile(
                uplift_score, 100.0 * self.thresholds.sleeping_dog_quantile
            ))
            # Must strictly be negative to be a sleeping dog
            tau_dog_cutoff = min(-0.005, tau_dog_cutoff)

            # Baseline risk median or percentile split
            risk_split_cutoff = float(np.percentile(
                churn_risk, 100.0 * self.thresholds.churn_risk_split_quantile
            ))
        else:
            tau_persuadable_cutoff = self.thresholds.min_persuadable_uplift
            tau_dog_cutoff = self.thresholds.max_sleeping_dog_uplift
            risk_split_cutoff = self.thresholds.baseline_churn_risk_threshold

        # 2. Economic Break-Even check if monthly charges provided
        if monthly_charges is not None:
            monthly_charges = np.asarray(monthly_charges, dtype=float)
            break_even_taus = np.array([self.economics.break_even_uplift(mc) for mc in monthly_charges])
        else:
            # Assume benchmark monthly charge of $65
            default_be = self.economics.break_even_uplift(65.0)
            break_even_taus = np.full(n, default_be)

        # 3. Vectorized Quadrant Classification
        # Quadrant masks:
        is_sleeping_dog = (uplift_score <= tau_dog_cutoff) & (uplift_score < 0.0)
        
        # Persuadables must have positive uplift and exceed both quantile cutoff AND economic break-even
        is_persuadable = (
            (~is_sleeping_dog)
            & (uplift_score > 0.0)
            & (uplift_score >= tau_persuadable_cutoff)
            & (uplift_score >= break_even_taus)
        )

        # Remaining non-persuadable, non-sleeping dog customers are split into Sure Things vs Lost Causes
        is_sure_thing = (~is_sleeping_dog) & (~is_persuadable) & (churn_risk < risk_split_cutoff)
        is_lost_cause = (~is_sleeping_dog) & (~is_persuadable) & (churn_risk >= risk_split_cutoff)

        # Build output dataframe
        quadrant_labels = np.empty(n, dtype=object)
        recommended_actions = np.empty(n, dtype=object)
        is_targetable = np.zeros(n, dtype=bool)

        quadrant_labels[is_persuadable] = QuadrantLabel.PERSUADABLE.value
        recommended_actions[is_persuadable] = "Target with 10% Promotional Retention Discount"
        is_targetable[is_persuadable] = True

        quadrant_labels[is_sure_thing] = QuadrantLabel.SURE_THING.value
        recommended_actions[is_sure_thing] = "Suppress Offer - High Organic Retention (Avoid Cannibalization)"

        quadrant_labels[is_lost_cause] = QuadrantLabel.LOST_CAUSE.value
        recommended_actions[is_lost_cause] = "Suppress Offer - Inelastic Churn Risk (Avoid Wasted Spend)"

        quadrant_labels[is_sleeping_dog] = QuadrantLabel.DO_NOT_DISTURB.value
        recommended_actions[is_sleeping_dog] = "Strictly Suppress - Outreach Triggers Subscription Cancellation"

        # Fallback for any unassigned edge cases
        unassigned = (quadrant_labels == None)  # noqa: E711
        if np.any(unassigned):
            quadrant_labels[unassigned] = QuadrantLabel.SURE_THING.value
            recommended_actions[unassigned] = "Suppress Offer - High Organic Retention"

        df_out = pd.DataFrame({
            "churn_risk": churn_risk,
            "uplift_score": uplift_score,
            "quadrant_label": quadrant_labels,
            "recommended_action": recommended_actions,
            "is_targetable": is_targetable,
            "break_even_uplift": break_even_taus
        })
        return df_out

    def get_quadrant_summary(self, segmented_df: pd.DataFrame) -> pd.DataFrame:
        """Returns executive distribution summary of behavioral quadrants."""
        total_n = len(segmented_df)
        summary = (
            segmented_df.groupby("quadrant_label")
            .agg(
                customer_count=("churn_risk", "count"),
                avg_churn_risk=("churn_risk", "mean"),
                avg_uplift_score=("uplift_score", "mean")
            )
            .reset_index()
        )
        summary["percentage"] = (summary["customer_count"] / total_n) * 100.0
        summary = summary.sort_values(by="customer_count", ascending=False).reset_index(drop=True)
        return summary
