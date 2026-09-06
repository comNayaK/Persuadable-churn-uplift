"""
Causal Uplift Engine: Meta-Learners for Heterogeneous Treatment Effect Estimation.

Implements:
1. T-Learner (Two-Model Benchmark)
2. X-Learner (Künzel et al., PNAS 2019) with canonical counterfactual imputation

Sign Convention & Target Variable:
- Primary causal target is RETENTION: R = 1 - Churn (where 1 = Retained, 0 = Churned).
- Uplift tau(X) = E[R(1) - R(0) | X] = P(Churn | W=0) - P(Churn | W=1).
- tau(X) > 0  => Persuadable (treatment reduces churn / increases retention).
- tau(X) ~ 0  => Sure Thing or Lost Cause (treatment has negligible effect).
- tau(X) < 0  => Do-Not-Disturb / Sleeping Dog (treatment triggers churn).
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb


class TLearner(BaseEstimator):
    """
    T-Learner (Two-Learner) for Uplift Estimation.
    Trains separate models for treated and control groups:
    mu_1(X) = E[R | X, W=1]
    mu_0(X) = E[R | X, W=0]
    tau(X) = mu_1(X) - mu_0(X)
    """

    def __init__(
        self,
        base_learner_type: str = "lightgbm",
        n_estimators: int = 120,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.base_learner_type = base_learner_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.model_0_: Optional[Any] = None
        self.model_1_: Optional[Any] = None
        self.feature_names_: list = []

    def _init_base_model(self):
        if self.base_learner_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=-1
            )
        elif self.base_learner_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

    def fit(self, X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray, target_is_churn: bool = True):
        """
        Fits T-Learner on features X, target y, and treatment assignment w.
        If target_is_churn=True, converts y to retention (R = 1 - y).
        """
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_np = np.asarray(y)
        w_np = np.asarray(treatment).astype(int)

        # Standardize target to retention: R = 1 - churn
        r_np = (1 - y_np) if target_is_churn else y_np

        mask_0 = (w_np == 0)
        mask_1 = (w_np == 1)

        self.model_0_ = self._init_base_model()
        self.model_0_.fit(X_np[mask_0], r_np[mask_0])

        self.model_1_ = self._init_base_model()
        self.model_1_.fit(X_np[mask_1], r_np[mask_1])

        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimates conditional average treatment effect (CATE):
        tau(X) = mu_1(X) - mu_0(X) = P(Retain|W=1) - P(Retain|W=0)
        """
        if self.model_0_ is None or self.model_1_ is None:
            raise RuntimeError("TLearner has not been fitted.")
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        mu_0 = self.model_0_.predict_proba(X_np)[:, 1]
        mu_1 = self.model_1_.predict_proba(X_np)[:, 1]

        return mu_1 - mu_0


class XLearner(BaseEstimator):
    """
    X-Learner (Künzel et al., PNAS 2019) for Heterogeneous Treatment Effect Estimation.
    
    Canonical 5-Stage Architecture:
    1. First Stage: Train base response models on retention R:
       mu_0(X) = E[R | X, W=0]
       mu_1(X) = E[R | X, W=1]
    2. Counterfactual Imputation:
       D_1 = R_1 - mu_0(X_1)   (Observed treated retention - imputed counterfactual control)
       D_0 = mu_1(X_0) - R_0   (Imputed counterfactual treated - observed control retention)
    3. Second Stage: Train effect models on imputed counterfactual differences:
       tau_1(X) fit on (X_1, D_1)
       tau_0(X) fit on (X_0, D_0)
    4. Propensity Estimation:
       e(X) = P(W=1 | X)
    5. Propensity-Weighted Synthesis:
       tau(X) = e(X)*tau_0(X) + (1 - e(X))*tau_1(X)
    """

    def __init__(
        self,
        base_learner_type: str = "lightgbm",
        n_estimators: int = 120,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        random_state: int = 42
    ):
        self.base_learner_type = base_learner_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state

        self.mu_0_: Optional[Any] = None
        self.mu_1_: Optional[Any] = None
        self.tau_0_: Optional[Any] = None
        self.tau_1_: Optional[Any] = None
        self.propensity_model_: Optional[Any] = None
        self.feature_names_: list = []

    def _init_classifier(self):
        if self.base_learner_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=-1
            )
        elif self.base_learner_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

    def _init_regressor(self):
        """Second stage predicts continuous counterfactual differences D."""
        if self.base_learner_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                verbosity=-1
            )
        elif self.base_learner_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state
            )

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        treatment: np.ndarray,
        propensity_scores: Optional[np.ndarray] = None,
        target_is_churn: bool = True
    ):
        """
        Fits the canonical 5-stage X-Learner.
        y: Binary target. If target_is_churn=True (default), converts to retention R = 1 - y.
        treatment: Binary assignment (1 = Treated, 0 = Control).
        """
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else []
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_np = np.asarray(y)
        w_np = np.asarray(treatment).astype(int)

        # Standardize target to retention: R = 1 - churn
        r_np = (1 - y_np) if target_is_churn else y_np

        mask_0 = (w_np == 0)
        mask_1 = (w_np == 1)

        X_0, r_0 = X_np[mask_0], r_np[mask_0]
        X_1, r_1 = X_np[mask_1], r_np[mask_1]

        # Stage 1: Fit base models mu_0 and mu_1 on observed retention
        self.mu_0_ = self._init_classifier()
        self.mu_0_.fit(X_0, r_0)

        self.mu_1_ = self._init_classifier()
        self.mu_1_.fit(X_1, r_1)

        # Stage 2: Canonical Counterfactual Imputation (Künzel et al.)
        # D_1 = R_1 - mu_0(X_1)
        mu_0_on_treated = self.mu_0_.predict_proba(X_1)[:, 1]
        D_1 = r_1 - mu_0_on_treated

        # D_0 = mu_1(X_0) - R_0
        mu_1_on_control = self.mu_1_.predict_proba(X_0)[:, 1]
        D_0 = mu_1_on_control - r_0

        # Stage 3: Train second-stage models tau_1 and tau_0
        self.tau_1_ = self._init_regressor()
        self.tau_1_.fit(X_1, D_1)

        self.tau_0_ = self._init_regressor()
        self.tau_0_.fit(X_0, D_0)

        # Stage 4: Estimate Propensity Score e(X) = P(W=1 | X)
        if propensity_scores is not None:
            self.propensity_model_ = None
            self._external_propensities = np.asarray(propensity_scores)
        else:
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.propensity_model_ = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=500, random_state=self.random_state)
            )
            self.propensity_model_.fit(X_np, w_np)

        return self

    def predict_uplift(self, X: pd.DataFrame) -> np.ndarray:
        """
        Estimates CATE tau(X) using propensity-weighted combination:
        tau(X) = e(X) * tau_0(X) + (1 - e(X)) * tau_1(X)
        """
        if self.tau_0_ is None or self.tau_1_ is None:
            raise RuntimeError("XLearner has not been fitted yet.")
        X_np = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)

        # Predict second-stage effects
        pred_tau_0 = self.tau_0_.predict(X_np)
        pred_tau_1 = self.tau_1_.predict(X_np)

        # Propensity score e(X)
        if self.propensity_model_ is not None:
            e_x = self.propensity_model_.predict_proba(X_np)[:, 1]
        else:
            e_x = np.full(len(X_np), 0.50)

        # Bound propensity away from 0 and 1 for numerical stability
        e_x = np.clip(e_x, 0.05, 0.95)

        # Canonical X-Learner weighting
        tau = e_x * pred_tau_0 + (1.0 - e_x) * pred_tau_1
        return tau
