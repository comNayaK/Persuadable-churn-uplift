"""
End-to-End Batch Pipeline Orchestrator.

Executes the full workflow:
1. Ingestion & Preprocessing (Telco Churn with semi-synthetic trial or Hillstrom RCT)
2. Feature Engineering (45+ features)
3. Baseline Classifier Training & Stratified CV (Targeting AUC >= 0.84 - 0.86)
4. Causal Uplift Modeling via X-Learner / T-Learner (Estimating CATE tau(X))
5. Uplift Evaluation (Qini Curve, Qini Score >= 0.10, Decile Lift)
6. Behavioral Quadrant Segmentation (Persuadables, Sure Things, Lost Causes, Sleeping Dogs)
7. Budget-Constrained Policy Simulation & ROI Lift Calculation (Targeting ROI Lift >= 20%)
8. Batch Dispatch Export with Required Schema:
   [customer_id, churn_score, uplift_score, quadrant_label, recommended_action, execution_timestamp]
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import PipelineConfig
from ..data.loader import DataLoader
from ..data.feature_engineering import FeatureEngineer
from ..models.baseline import BaselineChurnModel
from ..models.causal import TLearner, XLearner
from ..decision.quadrant import QuadrantEngine
from ..decision.policy import PolicySimulator
from ..evaluation.metrics import UpliftEvaluator
from ..evaluation.visualization import UpliftVisualizer


class PipelineRunner:
    """Orchestrates end-to-end batch scoring and evaluation."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.data_loader = DataLoader(self.config.data_dir, random_state=self.config.model.random_state)
        self.feature_engineer = FeatureEngineer()
        self.baseline_model = BaselineChurnModel(
            model_type=self.config.model.baseline_model_type,
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            learning_rate=self.config.model.learning_rate,
            random_state=self.config.model.random_state
        )
        if self.config.model.causal_learner_type == "t_learner":
            self.causal_learner = TLearner(
                base_learner_type="lightgbm",
                n_estimators=self.config.model.n_estimators,
                max_depth=self.config.model.max_depth,
                learning_rate=self.config.model.learning_rate,
                random_state=self.config.model.random_state
            )
        else:
            self.causal_learner = XLearner(
                base_learner_type="lightgbm",
                n_estimators=self.config.model.n_estimators,
                max_depth=self.config.model.max_depth,
                learning_rate=self.config.model.learning_rate,
                random_state=self.config.model.random_state
            )

        self.quadrant_engine = QuadrantEngine(self.config.quadrant, self.config.economics)
        self.policy_simulator = PolicySimulator(self.config.economics)
        self.visualizer = UpliftVisualizer()

    def run(
        self,
        dataset_name: str = "telco",
        campaign_budget: float = 5000.0,
        output_csv_path: Optional[str] = None
    ) -> pd.DataFrame:
        print("=" * 75)
        print(f"Starting Persuadable Churn Uplift Pipeline [{dataset_name.upper()}]")
        print("=" * 75)

        # 1. Ingestion
        print("\n[Step 1/7] Ingesting and preparing trial dataset...")
        is_churn = dataset_name.lower() != "hillstrom"
        if not is_churn:
            raw_df = self.data_loader.load_hillstrom()
            raw_df["customerID"] = [f"CUST-{i:05d}" for i in range(len(raw_df))]
            target_col = "target_conversion"
            treatment_col = "treatment"
            id_col = "customerID"
        else:
            raw_df = self.data_loader.load_telco(is_rct=True)
            target_col = "churn"
            treatment_col = "treatment"
            id_col = "customerID"

        print(f" Loaded {len(raw_df):,} records. Treatment rate: {raw_df[treatment_col].mean():.1%}")

        # 2. Train / Holdout Split
        train_df, test_df = train_test_split(
            raw_df,
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=raw_df[target_col]
        )
        print(f" Split: Train={len(train_df):,} | Holdout Test={len(test_df):,}")

        # 3. Feature Engineering
        print("\n[Step 2/7] Running feature engineering pipeline (25+ features)...")
        X_train = self.feature_engineer.fit_transform(train_df)
        X_test = self.feature_engineer.transform(test_df)
        print(f" Extracted {X_train.shape[1]} clean numeric features.")

        y_train = train_df[target_col].values
        w_train = train_df[treatment_col].values
        y_test = test_df[target_col].values
        w_test = test_df[treatment_col].values

        # 4. Step 1 Modeling: Baseline Predictive Classifier
        print("\n[Step 3/7] Training Baseline Classifier with 5-Fold Stratified Cross-Validation...")
        cv_metrics = self.baseline_model.fit_cv(X_train, y_train, cv_folds=self.config.model.cv_folds)
        print(f" Baseline 5-Fold CV AUC: {cv_metrics['cv_auc_mean']:.4f} (+/- {cv_metrics['cv_auc_std']:.4f})")
        print(f" Baseline 5-Fold Brier Score: {cv_metrics['cv_brier_score']:.4f}")

        holdout_eval = self.baseline_model.evaluate(X_test, y_test)
        print(f" Holdout Test AUC-ROC: {holdout_eval['auc_roc']:.4f} (Target: >= 0.84)")

        # Predict baseline risk for entire holdout
        test_churn_risk = self.baseline_model.predict_churn_risk(X_test)

        # 5. Step 2 Modeling: Causal Meta-Learner (HTE Estimation)
        print(f"\n[Step 4/7] Training Causal Meta-Learner [{self.config.model.causal_learner_type.upper()}]...")
        self.causal_learner.fit(X_train, y_train, w_train, target_is_churn=is_churn)
        test_uplift = self.causal_learner.predict_uplift(X_test)
        print(f" Uplift tau(X) Mean: {test_uplift.mean():+.4f} | Min: {test_uplift.min():+.4f} | Max: {test_uplift.max():+.4f}")

        # 6. Step 3 Evaluation: Qini Curve & AUUC
        print("\n[Step 5/7] Evaluating Qini Curve and Decile Uplift...")
        qini_res = UpliftEvaluator.compute_qini_curve(y_test, w_test, test_uplift, target_is_churn=is_churn)
        print(f" Qini Score: {qini_res.qini_score:.4f} (Target: >= 0.10)")
        print(f" AUUC: {qini_res.auuc:.4f}")

        # Save Qini plot to artifacts
        plot_path = self.config.artifacts_dir / "qini_curve.png"
        self.visualizer.plot_qini_curve(qini_res, save_path=plot_path)
        print(f" Saved Qini Curve visual to: {plot_path}")

        # 7. Step 4 Decision: 4-Quadrant Behavioral Segmentation
        print("\n[Step 6/7] Segmenting customers into 4 Behavioral Quadrants...")
        monthly_bills = test_df["MonthlyCharges"].values if "MonthlyCharges" in test_df.columns else None
        segmented_df = self.quadrant_engine.segment_customers(test_churn_risk, test_uplift, monthly_bills)
        
        # Merge metadata
        segmented_df["customer_id"] = test_df[id_col].values
        segmented_df["MonthlyCharges"] = monthly_bills if monthly_bills is not None else 65.0
        if "true_tau_churn" in test_df.columns:
            segmented_df["true_tau_churn"] = test_df["true_tau_churn"].values

        quadrant_summary = self.quadrant_engine.get_quadrant_summary(segmented_df)
        print("\nQuadrant Distribution:")
        for _, row in quadrant_summary.iterrows():
            print(f"  * {row['quadrant_label']:<15}: {row['customer_count']:>5} accounts ({row['percentage']:>5.1f}%) | Avg Churn Risk: {row['avg_churn_risk']:.1%} | Avg Uplift: {row['avg_uplift_score']:+.2%}")

        # Save Quadrant plot
        quad_plot_path = self.config.artifacts_dir / "quadrant_scatter.png"
        self.visualizer.plot_quadrant_scatter(segmented_df, save_path=quad_plot_path)
        print(f" Saved Quadrant Scatter visual to: {quad_plot_path}")

        # 8. Step 5 Policy Simulation: Campaign ROI Comparison
        print(f"\n[Step 7/7] Simulating Campaign Allocation under Budget (${campaign_budget:,.2f})...")
        sim_res = self.policy_simulator.simulate_policies(segmented_df, campaign_budget=campaign_budget)
        
        comp_table = self.policy_simulator.format_comparison_table(sim_res)
        print("\nPolicy Simulation Comparison Table:")
        print(comp_table.to_string(index=False))

        # Calculate ROI Lift of Dual Layer / Causal vs Standard ML
        roi_a = sim_res["standard_ml"].roi_percentage
        roi_b = sim_res["causal_uplift"].roi_percentage
        roi_c = sim_res["dual_layer"].roi_percentage
        
        roi_lift = ((roi_c - roi_a) / max(abs(roi_a), 1e-4)) * 100.0 if roi_a > 0 else (roi_c - roi_a)
        print(f"\n>> CAMPAIGN ROI LIFT OVER STANDARD CHURN TARGETING: {roi_lift:+.1f}% (Target: >= 20.0%)")

        # 9. Format Required Output Schema
        # Schema: [customer_id, churn_score, uplift_score, quadrant_label, recommended_action, execution_timestamp]
        timestamp = datetime.now().isoformat()
        dispatch_df = pd.DataFrame({
            "customer_id": segmented_df["customer_id"],
            "churn_score": segmented_df["churn_risk"].round(4),
            "uplift_score": segmented_df["uplift_score"].round(4),
            "quadrant_label": segmented_df["quadrant_label"],
            "recommended_action": segmented_df["recommended_action"],
            "execution_timestamp": timestamp
        })

        out_path = Path(output_csv_path) if output_csv_path else self.config.artifacts_dir / "persuadables_campaign_dispatch.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        dispatch_df.to_csv(out_path, index=False)
        print(f"\n Exported {len(dispatch_df):,} scored accounts to: {out_path}")

        # Persist summary metrics for dashboard or inspection
        summary_info = {
            "cv_auc": float(cv_metrics["cv_auc_mean"]),
            "holdout_auc": float(holdout_eval["auc_roc"]),
            "qini_score": float(qini_res.qini_score),
            "auuc": float(qini_res.auuc),
            "roi_lift_pct": float(roi_lift),
            "total_scored": len(dispatch_df),
            "persuadables_count": int((dispatch_df["quadrant_label"] == "Persuadable").sum()),
            "sleeping_dogs_count": int((dispatch_df["quadrant_label"] == "Do-Not-Disturb").sum())
        }
        pd.Series(summary_info).to_json(self.config.artifacts_dir / "pipeline_metrics.json")

        print("=" * 75)
        print("Pipeline Execution Completed Successfully!")
        print("=" * 75)

        return dispatch_df


def main():
    parser = argparse.ArgumentParser(description="Persuadable Churn Uplift Decision Engine")
    parser.add_argument("--dataset", type=str, default="telco", choices=["telco", "hillstrom"], help="Dataset to load")
    parser.add_argument("--budget", type=float, default=5000.0, help="Marketing campaign budget in USD")
    parser.add_argument("--learner", type=str, default="x_learner", choices=["x_learner", "t_learner"], help="Causal Meta-Learner")
    parser.add_argument("--output", type=str, default=None, help="Path to output dispatch CSV")

    args = parser.parse_args()
    config = PipelineConfig()
    config.model.causal_learner_type = args.learner

    runner = PipelineRunner(config=config)
    runner.run(dataset_name=args.dataset, campaign_budget=args.budget, output_csv_path=args.output)


if __name__ == "__main__":
    main()
