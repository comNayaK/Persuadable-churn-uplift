"""
Visualization Module for Uplift Curves, Quadrant Distributions, and Policy Comparisons.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .metrics import QiniCurveResult


class UpliftVisualizer:
    """Generates publication-quality charts for Qini curves and quadrant decision plots."""

    def __init__(self, style: str = "seaborn-v0_8-whitegrid"):
        try:
            plt.style.use(style)
        except Exception:
            pass

    def plot_qini_curve(
        self,
        qini_res: QiniCurveResult,
        model_name: str = "X-Learner (CATE)",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """Plots the standard commercial Qini Curve vs Random baseline."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

        # Plot Qini Curve
        ax.plot(
            qini_res.fractions * 100.0,
            qini_res.qini_values,
            label=f"{model_name} (Qini Score = {qini_res.qini_score:.3f})",
            color="#2563eb",
            linewidth=2.5
        )

        # Plot Random Selection line
        ax.plot(
            qini_res.fractions * 100.0,
            qini_res.random_values,
            label="Random Allocation (Baseline)",
            color="#94a3b8",
            linestyle="--",
            linewidth=1.8
        )

        # Fill area between curve and random
        ax.fill_between(
            qini_res.fractions * 100.0,
            qini_res.qini_values,
            qini_res.random_values,
            color="#3b82f6",
            alpha=0.15,
            label="Incremental Causal Lift"
        )

        ax.set_title("Retention Campaign Qini Curve (Cumulative Incremental Retained)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Percentage of Targeted Customer Base (%)", fontsize=11)
        ax.set_ylabel("Incremental Retained Customers ($Q(t)$)", fontsize=11)
        ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
        ax.set_xlim(0, 100)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig

    def plot_quadrant_scatter(
        self,
        segmented_df: pd.DataFrame,
        sample_size: int = 1500,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Plots the 2D Behavioral Quadrant Space:
        X-axis: Baseline Churn Risk P(Churn | X)
        Y-axis: Causal Uplift tau(X)
        """
        fig, ax = plt.subplots(figsize=(9, 7), dpi=120)

        df_plot = segmented_df.sample(min(len(segmented_df), sample_size), random_state=42)

        palette = {
            "Persuadable": "#10b981",    # Emerald Green (Target)
            "Sure Thing": "#3b82f6",     # Blue (Suppress - Organic Loyal)
            "Lost Cause": "#f59e0b",     # Amber/Orange (Suppress - Inelastic Churn)
            "Do-Not-Disturb": "#ef4444"  # Red (Strictly Suppress - Sleeping Dog)
        }

        sns.scatterplot(
            data=df_plot,
            x="churn_risk",
            y="uplift_score",
            hue="quadrant_label",
            palette=palette,
            alpha=0.65,
            s=45,
            ax=ax
        )

        # Draw zero uplift line and risk split
        ax.axhline(0.0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.8)
        
        ax.set_title("Dual-Layer Decision Space: Baseline Churn Risk vs Causal Uplift", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Baseline Churn Risk $P(Churn \\mid X)$", fontsize=11)
        ax.set_ylabel("Predicted Causal Uplift $\\hat{\\tau}(X)$ (Churn Reduction)", fontsize=11)
        ax.legend(title="Behavioral Quadrant", loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")

        return fig
