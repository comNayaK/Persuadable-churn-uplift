"""
Baseline Predictive Churn Risk Model.

Predicts customer-level baseline churn probability P(Churn | X).
Supports XGBoost, LightGBM, and Random Forest with 5-fold Stratified Cross-Validation,
hyperparameter tuning, probability calibration, and feature importance analysis.
Targeting AUC-ROC >= 0.84 - 0.86.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb


class BaselineChurnModel(BaseEstimator, ClassifierMixin):
    """
    Supervised predictive classifier for customer churn risk under business-as-usual.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 150,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.80,
        random_state: int = 42,
        calibrate: bool = True
    ):
        self.model_type = model_type.lower()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.calibrate = calibrate

        self.model_: Optional[Any] = None
        self.feature_names_: List[str] = []
        self.cv_results_: Dict[str, Any] = {}

    def _init_estimator(self):
        """Instantiates the underlying base classification algorithm."""
        if self.model_type == "xgboost":
            return xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                eval_metric="logloss"
            )
        elif self.model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbosity=-1
            )
        elif self.model_type in ["random_forest", "rf"]:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth + 4,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}. Choose from 'xgboost', 'lightgbm', 'rf'.")

    def fit_cv(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, float]:
        """
        Performs Stratified K-Fold cross-validation and records out-of-fold AUC, log-loss, and Brier score.
        """
        self.feature_names_ = list(X.columns)
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        oof_preds = np.zeros(len(y))
        fold_aucs = []

        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = np.asarray(y)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_np, y_np)):
            X_train, y_train = X_np[train_idx], y_np[train_idx]
            X_val, y_val = X_np[val_idx], y_np[val_idx]

            fold_model = self._init_estimator()
            fold_model.fit(X_train, y_train)

            preds_val = fold_model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = preds_val
            fold_auc = roc_auc_score(y_val, preds_val)
            fold_aucs.append(fold_auc)

        cv_auc = float(roc_auc_score(y_np, oof_preds))
        cv_brier = float(brier_score_loss(y_np, oof_preds))
        cv_logloss = float(log_loss(y_np, oof_preds))

        self.cv_results_ = {
            "cv_auc_mean": cv_auc,
            "cv_auc_std": float(np.std(fold_aucs)),
            "cv_brier_score": cv_brier,
            "cv_log_loss": cv_logloss,
            "fold_aucs": fold_aucs,
            "oof_predictions": oof_preds
        }

        # Train final calibrated model on all training data
        base_est = self._init_estimator()
        if self.calibrate:
            self.model_ = CalibratedClassifierCV(estimator=base_est, cv=3, method="sigmoid")
            self.model_.fit(X_np, y_np)
        else:
            self.model_ = base_est
            self.model_.fit(X_np, y_np)

        return self.cv_results_

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        """Fits model directly on provided features and binary targets."""
        self.feature_names_ = list(X.columns)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = np.asarray(y)

        base_est = self._init_estimator()
        if self.calibrate:
            self.model_ = CalibratedClassifierCV(estimator=base_est, cv=3, method="sigmoid")
            self.model_.fit(X_np, y_np)
        else:
            self.model_ = base_est
            self.model_.fit(X_np, y_np)

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 2-column array of [P(Retain), P(Churn)]."""
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        return self.model_.predict_proba(X_np)

    def predict_churn_risk(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1D array of churn probabilities P(Churn = 1 | X)."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Predicts binary churn class based on threshold."""
        return (self.predict_churn_risk(X) >= threshold).astype(int)

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluates predictive metrics on a holdout test set."""
        probs = self.predict_churn_risk(X_test)
        preds = (probs >= 0.50).astype(int)
        y_test_np = np.asarray(y_test)

        return {
            "auc_roc": float(roc_auc_score(y_test_np, probs)),
            "brier_score": float(brier_score_loss(y_test_np, probs)),
            "log_loss": float(log_loss(y_test_np, probs)),
            "accuracy": float(accuracy_score(y_test_np, preds)),
            "precision": float(precision_score(y_test_np, preds, zero_division=0)),
            "recall": float(recall_score(y_test_np, preds, zero_division=0)),
            "f1_score": float(f1_score(y_test_np, preds, zero_division=0))
        }

    def get_feature_importances(self) -> pd.DataFrame:
        """Extracts normalized feature importance rankings."""
        if self.model_ is None:
            raise RuntimeError("Model has not been fitted yet.")
        
        # Extract underlying estimator if calibrated
        estimator = self.model_
        if isinstance(estimator, CalibratedClassifierCV):
            # Average across calibrated estimators
            importances = []
            for cal_clf in estimator.calibrated_classifiers_:
                base = cal_clf.estimator
                if hasattr(base, "feature_importances_"):
                    importances.append(base.feature_importances_)
            if importances:
                avg_imp = np.mean(importances, axis=0)
            else:
                avg_imp = np.zeros(len(self.feature_names_))
        elif hasattr(estimator, "feature_importances_"):
            avg_imp = estimator.feature_importances_
        else:
            avg_imp = np.zeros(len(self.feature_names_))

        df_imp = pd.DataFrame({
            "feature": self.feature_names_,
            "importance": avg_imp
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)
        return df_imp
