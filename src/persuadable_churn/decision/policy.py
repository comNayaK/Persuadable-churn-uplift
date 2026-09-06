"""
Policy Simulation and Campaign ROI Optimization Engine.

Compares retention campaign outcomes under alternative targeting policies with a fixed budget:
- Policy A (Standard ML): Target top k users ranked by predicted churn risk P(Churn | X).
- Policy B (Causal Uplift): Target top k users ranked by predicted uplift tau(X) > 0.
- Policy C (Dual-Layer Engine): Target strictly Persuadables exceeding break-even uplift.

Evaluates incremental customers retained, net revenue saved, cost of offers,
and Campaign ROI lift over baseline (validating PRD target >= 20%).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from ..config import CampaignEconomics


@dataclass
class PolicyResult:
    """Audit metrics for a single campaign targeting policy."""
    policy_name: str
    customers_targeted: int
    outreach_spend: float
    expected_incremental_retained: float
    gross_revenue_saved: float
    net_revenue_saved: float
    roi_percentage: float
    sleeping_dogs_contacted: int
    lost_causes_contacted: int
    sure_things_contacted: int


class PolicySimulator:
    """
    Simulates marketing allocation policies under budget constraints and unit economics.
    """

    def __init__(self, economics: Optional[CampaignEconomics] = None):
        self.economics = economics or CampaignEconomics()

    def simulate_policies(
        self,
        scored_df: pd.DataFrame,
        campaign_budget: float = 5000.0,
        contact_cost: Optional[float] = None
    ) -> Dict[str, PolicyResult]:
        """
        Runs side-by-side simulation of Policy A, Policy B, and Policy C under campaign budget.

        Parameters:
        - scored_df: DataFrame containing:
          ['churn_risk', 'uplift_score', 'quadrant_label', 'MonthlyCharges']
          and optionally ground truth ['true_tau_churn', 'is_sleeping_dog_true']
        - campaign_budget: Total campaign budget in dollars
        - contact_cost: Cost per customer contacted (defaults to config: $1.50)

        Returns:
        Dict mapping policy key ('standard_ml', 'causal_uplift', 'dual_layer') -> PolicyResult
        """
        df = scored_df.copy()
        c_cost = contact_cost if contact_cost is not None else self.economics.outreach_cost
        max_contacts = int(campaign_budget // c_cost)
        max_contacts = min(max_contacts, len(df))

        # Monthly charges fallback
        if "MonthlyCharges" in df.columns:
            monthly_charges = df["MonthlyCharges"].values
        else:
            monthly_charges = np.full(len(df), 65.0)

        # Ground truth uplift if available, otherwise predicted uplift
        tau_eval = df["true_tau_churn"].values if "true_tau_churn" in df.columns else df["uplift_score"].values

        # -------------------------------------------------------------
        # 1. Policy A: Standard ML (Rank by Churn Risk)
        # -------------------------------------------------------------
        # Retention team contacts top highest churn probability accounts
        idx_policy_a = np.argsort(-df["churn_risk"].values)[:max_contacts]
        res_a = self._compute_policy_metrics(
            "Policy A: Standard ML Churn Risk",
            idx_policy_a,
            tau_eval,
            monthly_charges,
            df.get("quadrant_label", None),
            c_cost
        )

        # -------------------------------------------------------------
        # 2. Policy B: Causal Uplift (Rank by Uplift tau(X))
        # -------------------------------------------------------------
        # Contacts accounts with highest positive uplift
        sorted_tau_idx = np.argsort(-df["uplift_score"].values)
        # Filter strictly where tau > 0
        positive_tau_mask = df["uplift_score"].values[sorted_tau_idx] > 0
        idx_policy_b = sorted_tau_idx[positive_tau_mask][:max_contacts]
        res_b = self._compute_policy_metrics(
            "Policy B: Causal Uplift (CATE)",
            idx_policy_b,
            tau_eval,
            monthly_charges,
            df.get("quadrant_label", None),
            c_cost
        )

        # -------------------------------------------------------------
        # 3. Policy C: Dual-Layer Persuadables Engine
        # -------------------------------------------------------------
        # Targets strictly verified Persuadables exceeding break-even
        is_persuadable = (df["quadrant_label"] == "Persuadable").values if "quadrant_label" in df.columns else (df["uplift_score"].values > 0.02)
        persuadable_indices = np.where(is_persuadable)[0]
        # Rank persuadables by uplift score
        persuadables_ranked = persuadable_indices[np.argsort(-df["uplift_score"].values[persuadable_indices])]
        idx_policy_c = persuadables_ranked[:max_contacts]
        res_c = self._compute_policy_metrics(
            "Policy C: Dual-Layer Quadrant Engine",
            idx_policy_c,
            tau_eval,
            monthly_charges,
            df.get("quadrant_label", None),
            c_cost
        )

        return {
            "standard_ml": res_a,
            "causal_uplift": res_b,
            "dual_layer": res_c
        }

    def _compute_policy_metrics(
        self,
        policy_name: str,
        targeted_indices: np.ndarray,
        tau_eval: np.ndarray,
        monthly_charges: np.ndarray,
        quadrant_labels: Optional[pd.Series],
        contact_cost: float
    ) -> PolicyResult:
        k = len(targeted_indices)
        if k == 0:
            return PolicyResult(
                policy_name=policy_name,
                customers_targeted=0,
                outreach_spend=0.0,
                expected_incremental_retained=0.0,
                gross_revenue_saved=0.0,
                net_revenue_saved=0.0,
                roi_percentage=0.0,
                sleeping_dogs_contacted=0,
                lost_causes_contacted=0,
                sure_things_contacted=0
            )

        targeted_taus = tau_eval[targeted_indices]
        targeted_charges = monthly_charges[targeted_indices]

        # Incremental customers retained = sum of treatment effects in targeted cohort
        incremental_retained = float(np.sum(targeted_taus))
        
        # Financial savings:
        # For each saved customer: Monthly Charge * (1 - Discount) * Margin * Months
        clv_factor = (1.0 - self.economics.retention_discount_pct) * self.economics.gross_margin_pct * self.economics.default_clv_months
        gross_revenue_saved = float(np.sum(targeted_taus * targeted_charges * clv_factor))
        
        outreach_spend = float(k * contact_cost)
        net_revenue_saved = gross_revenue_saved - outreach_spend
        
        roi = (net_revenue_saved / outreach_spend * 100.0) if outreach_spend > 0 else 0.0

        # Behavioral count audit
        sleeping_dogs = 0
        lost_causes = 0
        sure_things = 0
        if quadrant_labels is not None:
            targeted_quadrants = quadrant_labels.iloc[targeted_indices].values
            sleeping_dogs = int(np.sum(targeted_quadrants == "Do-Not-Disturb"))
            lost_causes = int(np.sum(targeted_quadrants == "Lost Cause"))
            sure_things = int(np.sum(targeted_quadrants == "Sure Thing"))

        return PolicyResult(
            policy_name=policy_name,
            customers_targeted=k,
            outreach_spend=outreach_spend,
            expected_incremental_retained=incremental_retained,
            gross_revenue_saved=gross_revenue_saved,
            net_revenue_saved=net_revenue_saved,
            roi_percentage=roi,
            sleeping_dogs_contacted=sleeping_dogs,
            lost_causes_contacted=lost_causes,
            sure_things_contacted=sure_things
        )

    def format_comparison_table(self, simulation_results: Dict[str, PolicyResult]) -> pd.DataFrame:
        """Converts policy results into a clean side-by-side comparison table."""
        rows = []
        for key, res in simulation_results.items():
            rows.append({
                "Policy": res.policy_name,
                "Targeted Contacts": f"{res.customers_targeted:,}",
                "Outreach Spend": f"${res.outreach_spend:,.2f}",
                "Incremental Retained": f"{res.expected_incremental_retained:,.1f}",
                "Net Revenue Saved": f"${res.net_revenue_saved:,.2f}",
                "Campaign ROI": f"{res.roi_percentage:,.1f}%",
                "Sleeping Dogs Contacted": f"{res.sleeping_dogs_contacted:,}",
                "Lost Causes Contacted": f"{res.lost_causes_contacted:,}"
            })
        df_table = pd.DataFrame(rows)
        return df_table
