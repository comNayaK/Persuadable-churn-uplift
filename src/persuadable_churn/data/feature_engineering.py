"""
Feature Engineering Pipeline for Churn Risk and Causal Uplift Modeling.

Extracts 25+ engineered domain features:
- Tenure aggregates and buckets
- High-friction interaction terms
- Service breadth & bundle complexity
- Payment and billing friction flags
- Value-to-charge ratios
"""

from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Production feature engineering transformer for Telecom customer churn data.
    Ensures zero leakage across train/test splits.
    """

    ADDON_COLUMNS = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    def __init__(self):
        self.feature_names_: List[str] = []
        self.fitted_columns_: List[str] = []
        self.median_total_charges_: float = 0.0

    def fit(self, X: pd.DataFrame, y=None):
        """Fits preprocessors, records column schema, and learns training medians."""
        X_df = X.copy()
        self.is_hillstrom_ = "history" in X_df.columns and "recency" in X_df.columns
        
        if not self.is_hillstrom_:
            # Learn median TotalCharges on numeric values
            tc = pd.to_numeric(X_df.get("TotalCharges", pd.Series(dtype=float)), errors="coerce")
            self.median_total_charges_ = float(tc.median()) if not tc.isna().all() else 0.0
        
        # Transform dummy frame to capture all one-hot feature columns
        transformed = self._engineer_features(X_df)
        exclude_set = {
            "customerID", "churn", "Churn", "retained", "treatment", "target_conversion",
            "target_visit", "target_spend", "conversion", "visit", "spend", "segment"
        }
        self.fitted_columns_ = [c for c in transformed.columns if c not in exclude_set]
        self.feature_names_ = self.fitted_columns_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame into clean numerical feature matrix."""
        transformed = self._engineer_features(X.copy())
        
        # Realign with fitted schema (filling any missing columns with 0)
        for col in self.fitted_columns_:
            if col not in transformed.columns:
                transformed[col] = 0.0
                
        return transformed[self.fitted_columns_]

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if hasattr(self, "is_hillstrom_") and self.is_hillstrom_ or ("history" in df.columns and "recency" in df.columns):
            return self._engineer_hillstrom(df)
        return self._engineer_telco(df)

    def _engineer_hillstrom(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers domain features for Kevin Hillstrom's retail marketing trial."""
        recency = df["recency"].astype(float)
        history = df["history"].astype(float)
        mens = df["mens"].astype(float)
        womens = df["womens"].astype(float)
        newbie = df["newbie"].astype(float)

        df["log_history"] = np.log1p(history)
        df["history_per_recency"] = history / (recency + 1.0)
        df["is_active_shopper"] = (recency <= 3).astype(float)
        df["is_dormant_shopper"] = (recency >= 9).astype(float)
        df["high_value_customer"] = (history > 300.0).astype(float)

        # Cross category interactions
        df["mens_and_womens"] = mens * womens
        df["mens_history"] = mens * history
        df["womens_history"] = womens * history
        df["newbie_recent"] = newbie * (recency <= 3).astype(float)

        # Categoricals
        zip_dummies = pd.get_dummies(df["zip_code"], prefix="zip", drop_first=False, dtype=float)
        channel_dummies = pd.get_dummies(df["channel"], prefix="chan", drop_first=False, dtype=float)

        result_df = pd.concat([df, zip_dummies, channel_dummies], axis=1)
        drop_cols = ["history_segment", "zip_code", "channel", "segment", "customerID"]
        existing = [c for c in drop_cols if c in result_df.columns]
        result_df = result_df.drop(columns=existing)
        return result_df

    def _engineer_telco(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 1. Numeric Cleanups
        tenure = df["tenure"].astype(float)
        monthly = df["MonthlyCharges"].astype(float)
        
        # Handle TotalCharges with fallback
        total = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(monthly * tenure)
        df["TotalCharges_clean"] = total

        # 2. Tenure Aggregates & Ratios (6 features)
        df["tenure_months"] = tenure
        df["tenure_bucket_0_6m"] = (tenure <= 6).astype(float)
        df["tenure_bucket_6_12m"] = ((tenure > 6) & (tenure <= 12)).astype(float)
        df["tenure_bucket_12_24m"] = ((tenure > 12) & (tenure <= 24)).astype(float)
        df["tenure_bucket_24_48m"] = ((tenure > 24) & (tenure <= 48)).astype(float)
        df["tenure_bucket_48m_plus"] = (tenure > 48).astype(float)

        df["tenure_to_monthly_charge_ratio"] = tenure / (monthly + 1.0)
        df["monthly_to_tenure_burden"] = monthly / (tenure + 1.0)
        df["charge_discrepancy"] = total - (monthly * tenure)
        df["avg_historical_charge_ratio"] = (total / (tenure + 1.0)) / (monthly + 1.0)

        # 3. Service Breadth & Add-on Counts (6 features)
        for col in self.ADDON_COLUMNS:
            if col in df.columns:
                df[f"{col}_active"] = (df[col] == "Yes").astype(float)
            else:
                df[f"{col}_active"] = 0.0

        df["addon_count"] = (
            df["OnlineSecurity_active"]
            + df["OnlineBackup_active"]
            + df["DeviceProtection_active"]
            + df["TechSupport_active"]
            + df["StreamingTV_active"]
            + df["StreamingMovies_active"]
        )
        df["streaming_addon_count"] = df["StreamingTV_active"] + df["StreamingMovies_active"]
        df["security_backup_addon_count"] = (
            df["OnlineSecurity_active"] + df["OnlineBackup_active"] + df["DeviceProtection_active"]
        )
        df["has_no_addons"] = (df["addon_count"] == 0).astype(float)
        df["addon_density"] = df["addon_count"] / (monthly / 20.0 + 1.0)

        # 4. High-Friction Interaction Terms (6 features)
        is_fiber = (df["InternetService"] == "Fiber optic").astype(float)
        no_tech = (df["TechSupport"] == "No").astype(float)
        no_security = (df["OnlineSecurity"] == "No").astype(float)
        is_m2m = (df["Contract"] == "Month-to-month").astype(float)
        is_two_yr = (df["Contract"] == "Two year").astype(float)
        is_echeck = (df["PaymentMethod"] == "Electronic check").astype(float)
        is_paperless = (df["PaperlessBilling"] == "Yes").astype(float)

        df["fiber_no_techsupport"] = is_fiber * no_tech
        df["fiber_no_security"] = is_fiber * no_security
        df["month_to_month_high_charges"] = is_m2m * (monthly > 70.0).astype(float)
        df["contract_m2m_x_monthly_charges"] = is_m2m * monthly
        df["contract_two_yr_x_monthly_charges"] = is_two_yr * monthly
        df["echeck_x_monthly_charges"] = is_echeck * monthly
        df["m2m_paperless_friction"] = is_m2m * is_paperless

        # 5. Billing Friction & Payment Flags (5 features)
        is_auto_bank = (df["PaymentMethod"] == "Bank transfer (automatic)").astype(float)
        is_auto_cc = (df["PaymentMethod"] == "Credit card (automatic)").astype(float)
        df["is_electronic_check"] = is_echeck
        df["is_automatic_payment"] = is_auto_bank + is_auto_cc
        df["paperless_billing_active"] = is_paperless
        df["echeck_and_paperless"] = is_echeck * is_paperless

        # 6. Demographics & Household Structure (5 features)
        is_senior = (df["SeniorCitizen"] == 1).astype(float) if "SeniorCitizen" in df.columns else 0.0
        has_partner = (df["Partner"] == "Yes").astype(float) if "Partner" in df.columns else 0.0
        has_dependents = (df["Dependents"] == "Yes").astype(float) if "Dependents" in df.columns else 0.0

        df["is_senior"] = is_senior
        df["family_support_index"] = has_partner + has_dependents
        df["senior_living_alone"] = is_senior * (1.0 - has_partner) * (1.0 - has_dependents)
        df["has_multiple_lines"] = (df["MultipleLines"] == "Yes").astype(float) if "MultipleLines" in df.columns else 0.0
        df["phone_service_active"] = (df["PhoneService"] == "Yes").astype(float) if "PhoneService" in df.columns else 1.0

        # Categorical Dummies for Contract and InternetService
        contract_dummies = pd.get_dummies(df["Contract"], prefix="contract", drop_first=False, dtype=float)
        internet_dummies = pd.get_dummies(df["InternetService"], prefix="internet", drop_first=False, dtype=float)
        payment_dummies = pd.get_dummies(df["PaymentMethod"], prefix="payment", drop_first=False, dtype=float)

        result_df = pd.concat([
            df,
            contract_dummies,
            internet_dummies,
            payment_dummies
        ], axis=1)

        # Drop raw unencoded text columns
        raw_cols_to_drop = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges", "TotalCharges_clean",
            "true_p_churn_control", "true_p_churn_treated", "true_tau_churn", "is_sleeping_dog_true",
            "propensity_score", "Churn", "churn", "retained", "treatment", "customerID"
        ]
        existing_drop = [c for c in raw_cols_to_drop if c in result_df.columns]
        result_df = result_df.drop(columns=existing_drop)

        return result_df
