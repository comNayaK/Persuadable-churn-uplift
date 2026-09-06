"""
Causal Uplift Evaluation Engine: Qini Curve, Qini Score, Cumulative Gain, and Decile Lift.

Methodological Standard:
To ensure positive curvature when the retention campaign is effective:
- Response is strictly evaluated on RETENTION: R = 1 - Churn.
- Treated responders (n_t1) and control responders (n_t0) represent RETAINED customers.
- When persuadables are targeted first, n_t1 / N_t1 > n_t0 / N_t0, yielding an upward-sloping Qini curve.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class QiniCurveResult:
    """Stores full trajectory and summary statistics for Qini and Gain evaluation."""
    fractions: np.ndarray
    qini_values: np.ndarray
    random_values: np.ndarray
    cumulative_gains: np.ndarray
    qini_score: float
    auuc: float
    total_retained_treated: int
    total_retained_control: int
    total_treated: int
    total_control: int


class UpliftEvaluator:
    """
    Computes Qini curves, AUUC, decile lift tables, and policy lift metrics.
    """

    @staticmethod
    def compute_qini_curve(
        y_true: np.ndarray,
        treatment: np.ndarray,
        uplift_preds: np.ndarray,
        n_bins: int = 100,
        target_is_churn: bool = True
    ) -> QiniCurveResult:
        """
        Computes the Qini curve and Qini coefficient on holdout evaluation data.

        Parameters:
        - y_true: Target values (Churn indicator if target_is_churn=True, else Retention indicator)
        - treatment: Binary treatment assignment (1 = Treated, 0 = Control)
        - uplift_preds: Predicted uplift tau(X) (higher means higher priority for retention offer)
        - n_bins: Number of points along the curve
        - target_is_churn: If True, converts target to retention R = 1 - y
        """
        y_true = np.asarray(y_true)
        w = np.asarray(treatment).astype(int)
        tau_pred = np.asarray(uplift_preds)

        # Standardize target to retention: R = 1 - Churn
        retained = (1 - y_true) if target_is_churn else y_true

        # Sort descending by predicted uplift
        sort_order = np.argsort(-tau_pred)
        w_sorted = w[sort_order]
        r_sorted = retained[sort_order]

        total_n = len(y_true)
        indices = np.linspace(total_n // n_bins, total_n, n_bins, dtype=int)

        qini_vals = []
        random_vals = []
        cum_gains = []
        fractions = []

        total_w1 = int(np.sum(w == 1))
        total_w0 = int(np.sum(w == 0))
        total_r1 = int(np.sum(retained[w == 1]))
        total_r0 = int(np.sum(retained[w == 0]))

        # Final Qini value at 100% targeting
        q_final = total_r1 - (total_r0 * (total_w1 / total_w0)) if total_w0 > 0 else 0.0

        for idx in indices:
            frac = idx / total_n
            w_sub = w_sorted[:idx]
            r_sub = r_sorted[:idx]

            n_t1 = np.sum(r_sub[w_sub == 1])  # Retained treated
            n_t0 = np.sum(r_sub[w_sub == 0])  # Retained control
            N_t1 = np.sum(w_sub == 1)         # Total treated in top slice
            N_t0 = np.sum(w_sub == 0)         # Total control in top slice

            if N_t0 > 0:
                q_val = n_t1 - (n_t0 * (N_t1 / N_t0))
                # Cumulative Gain: (p_t1 - p_t0) * (N_t1 + N_t0)
                p_t1 = n_t1 / N_t1 if N_t1 > 0 else 0.0
                p_t0 = n_t0 / N_t0
                gain = (p_t1 - p_t0) * (N_t1 + N_t0)
            else:
                q_val = 0.0
                gain = 0.0

            q_rand = q_final * frac

            fractions.append(frac)
            qini_vals.append(q_val)
            random_vals.append(q_rand)
            cum_gains.append(gain)

        fractions = np.array(fractions)
        qini_vals = np.array(qini_vals)
        random_vals = np.array(random_vals)
        cum_gains = np.array(cum_gains)

        # Numerical integration for Area Under Curve
        trap_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
        auqc_model = float(trap_func(qini_vals, fractions))
        auqc_random = float(trap_func(random_vals, fractions))

        # Standard Qini Coefficient (Radcliffe & Surry):
        # Normalized incremental lift over random allocation
        if abs(auqc_random) > 1e-6:
            qini_score = float((auqc_model - auqc_random) / abs(auqc_random))
        else:
            qini_score = float((auqc_model - auqc_random) / max(total_n * 0.05, 1.0))

        auuc = auqc_model / (total_n / 2.0) if total_n > 0 else 0.0

        return QiniCurveResult(
            fractions=fractions,
            qini_values=qini_vals,
            random_values=random_vals,
            cumulative_gains=cum_gains,
            qini_score=qini_score,
            auuc=auuc,
            total_retained_treated=total_r1,
            total_retained_control=total_r0,
            total_treated=total_w1,
            total_control=total_w0
        )

    @staticmethod
    def compute_decile_table(
        y_true: np.ndarray,
        treatment: np.ndarray,
        uplift_preds: np.ndarray,
        target_is_churn: bool = True
    ) -> pd.DataFrame:
        """
        Builds a commercial 10-decile uplift summary table.
        """
        y_true = np.asarray(y_true)
        w = np.asarray(treatment).astype(int)
        tau_pred = np.asarray(uplift_preds)
        retained = (1 - y_true) if target_is_churn else y_true

        df = pd.DataFrame({
            "treatment": w,
            "retained": retained,
            "pred_uplift": tau_pred
        })

        # Assign deciles (1 = top 10% highest uplift)
        df["decile"] = pd.qcut(df["pred_uplift"].rank(method="first"), q=10, labels=range(10, 0, -1))
        
        decile_rows = []
        cum_treated_retained = 0
        cum_control_retained = 0
        cum_treated_total = 0
        cum_control_total = 0

        for d in range(1, 11):
            sub = df[df["decile"] == d]
            
            t_sub = sub[sub["treatment"] == 1]
            c_sub = sub[sub["treatment"] == 0]

            n_t = len(t_sub)
            n_c = len(c_sub)
            r_t = int(t_sub["retained"].sum())
            r_c = int(c_sub["retained"].sum())

            rate_t = (r_t / n_t) if n_t > 0 else 0.0
            rate_c = (r_c / n_c) if n_c > 0 else 0.0
            decile_uplift = rate_t - rate_c

            cum_treated_retained += r_t
            cum_control_retained += r_c
            cum_treated_total += n_t
            cum_control_total += n_c

            cum_rate_t = (cum_treated_retained / cum_treated_total) if cum_treated_total > 0 else 0.0
            cum_rate_c = (cum_control_retained / cum_control_total) if cum_control_total > 0 else 0.0
            cum_uplift = cum_rate_t - cum_rate_c

            decile_rows.append({
                "Decile": d,
                "Customers": len(sub),
                "Treated Count": n_t,
                "Control Count": n_c,
                "Treated Retain %": f"{rate_t * 100.0:.1f}%",
                "Control Retain %": f"{rate_c * 100.0:.1f}%",
                "Incremental Uplift": f"{decile_uplift * 100.0:+.2f}%",
                "Cumulative Uplift": f"{cum_uplift * 100.0:+.2f}%"
            })

        return pd.DataFrame(decile_rows)
