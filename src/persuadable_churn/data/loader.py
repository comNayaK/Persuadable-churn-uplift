"""
Data Ingestion and Semi-Synthetic Trial Generation Engine.

Addresses Key Methodological Considerations:
1. Clear separation between Telecom Churn (negative event to suppress) and 
   Hillstrom E-commerce trials (positive event to increase).
2. Advanced semi-synthetic causal DGP (Data Generating Process) for Telco Churn:
   - Non-linear interactions & link functions (sigmoid).
   - Unobserved latent confounder / noise shock (epsilon ~ N(0, sigma^2)) preventing circular reverse-engineering.
   - Calibrated realistic Do-Not-Disturb rate (2% - 4%), not unrealistic double digits.
   - Supports both randomized trial (RCT: W ~ Bernoulli(0.5)) and observational trial with selection bias.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import requests


class SyntheticTrialGenerator:
    """
    Generates realistic, non-linear semi-synthetic treatment assignments and counterfactual 
    potential outcomes on the IBM Telco Churn dataset.
    
    Inspired by ACIC (Atlantic Causal Inference Conference) and Hill (2011) benchmarks.
    Ensures that ML models cannot trivially reverse-engineer the outcome via linear shortcuts.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate_trial(
        self,
        df: pd.DataFrame,
        is_rct: bool = True,
        treatment_propensity: float = 0.50,
        sleeping_dog_rate: float = 0.03,
        persuadable_rate: float = 0.16
    ) -> pd.DataFrame:
        """
        Augments customer behavioral features with treatment assignment W, potential outcomes
        Y(0), Y(1), ground-truth uplift tau*, observed churn Y, and retained indicator R.

        Conventions:
        - Y: Churn indicator (1 = Churned, 0 = Retained)
        - R: Retention indicator (1 = Retained, 0 = Churned) -> R = 1 - Y
        - tau_churn*: True CATE on churn reduction: P(Churn|W=0) - P(Churn|W=1)
        """
        df = df.copy()
        n = len(df)

        # Standardize key numeric fields for latent scoring
        tenure = df["tenure"].astype(float).values
        monthly_charges = df["MonthlyCharges"].astype(float).values
        
        # Clean TotalCharges
        if "TotalCharges" in df.columns:
            fallback = pd.Series(monthly_charges * tenure, index=df.index)
            total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(fallback).values
        else:
            total_charges = monthly_charges * tenure

        # Binary feature indicators for latent utility computation
        is_month_to_month = (df["Contract"] == "Month-to-month").astype(float).values
        is_two_year = (df["Contract"] == "Two year").astype(float).values
        has_fiber = (df["InternetService"] == "Fiber optic").astype(float).values
        no_tech_support = (df["TechSupport"] == "No").astype(float).values
        electronic_check = (df["PaymentMethod"] == "Electronic check").astype(float).values
        paperless = (df["PaperlessBilling"] == "Yes").astype(float).values
        senior = df["SeniorCitizen"].astype(float).values

        # 1. Baseline Churn Risk Latent Utility (Control group: W=0, Business-As-Usual)
        # Higher score = higher probability of churning under control
        z_control = (
            -1.4
            + 1.35 * is_month_to_month
            - 1.80 * is_two_year
            + 0.015 * (monthly_charges - 65.0)
            - 0.035 * (tenure - 30.0)
            + 0.50 * has_fiber * no_tech_support  # High-friction service interaction
            + 0.45 * electronic_check
            + 0.30 * paperless
            + 0.25 * senior
            # Non-linear ratio term: charge-burden per tenure month
            + 0.08 * (monthly_charges / (np.maximum(tenure, 1.0) + 5.0))
        )

        # 2. Add unobserved latent shock epsilon ~ N(0, 0.4^2) to break circular validation
        unobserved_shock = self.rng.normal(0, 0.40, size=n)
        z_control += unobserved_shock

        # Baseline churn probability P(Churn | W=0) via logistic sigmoid
        p_churn_control = 1.0 / (1.0 + np.exp(-z_control))
        # Clip to realistic probability range
        p_churn_control = np.clip(p_churn_control, 0.03, 0.95)

        # 3. Heterogeneous Treatment Effect (Outreach with 10% promotional retention discount)
        # tau_churn = P(Churn|W=0) - P(Churn|W=1) = reduction in churn probability
        # Positive tau = Persuadables (discount rescues customer)
        # Negative tau = Sleeping Dogs (outreach triggers cancellation of forgotten idle subscription)
        # Zero tau = Sure Things (stay anyway) or Lost Causes (leave anyway)

        # A. Latent propensity to be a Persuadable (price-sensitive, month-to-month, high monthly charges)
        persuadable_score = (
            1.2 * is_month_to_month
            + 0.02 * (monthly_charges - 60.0)
            + 0.4 * has_fiber
            - 0.8 * is_two_year
            - 0.02 * tenure
            + self.rng.normal(0, 0.35, size=n)
        )
        
        # B. Latent propensity to be a Sleeping Dog (low tenure, automated payment, paperless, low engagement)
        # Calibrated strictly to ~2% to 4% of the population
        sleeping_dog_score = (
            -0.03 * tenure
            + 0.6 * (1.0 - electronic_check) # automatic payments
            + 0.5 * paperless
            - 0.01 * monthly_charges
            + 0.4 * (1.0 - has_fiber) # low bandwidth/infrequent user
            + self.rng.normal(0, 0.40, size=n)
        )

        # Assign heterogeneous uplift delta:
        # Base churn reduction for most users is modest (~1-4%)
        churn_reduction = 0.03 + 0.14 * (1.0 / (1.0 + np.exp(-persuadable_score)))
        
        # Identify Sleeping Dogs (top sleeping_dog_rate percentile of sleeping_dog_score where control churn was low)
        dog_cutoff = np.percentile(sleeping_dog_score, 100.0 * (1.0 - sleeping_dog_rate))
        is_sleeping_dog = (sleeping_dog_score >= dog_cutoff) & (p_churn_control < 0.30)
        
        # For sleeping dogs, outreach increases churn (negative churn reduction)
        churn_reduction[is_sleeping_dog] = -self.rng.uniform(0.08, 0.22, size=np.sum(is_sleeping_dog))

        # Dampen effect for extreme cases (Sure Things & Lost Causes)
        # If customer has < 8% control churn risk, uplift cannot exceed their base risk
        churn_reduction = np.where(
            p_churn_control < 0.10,
            np.minimum(churn_reduction, p_churn_control * 0.4),
            churn_reduction
        )
        # If customer has > 85% control churn risk (Lost Cause), 10% discount barely moves them
        churn_reduction = np.where(
            p_churn_control > 0.85,
            churn_reduction * 0.15,
            churn_reduction
        )

        p_churn_treated = np.clip(p_churn_control - churn_reduction, 0.02, 0.96)
        true_tau_churn = p_churn_control - p_churn_treated  # positive means treatment reduces churn

        # 4. Treatment Assignment W
        if is_rct:
            # Pure Randomized Controlled Trial (A/B Test)
            w = self.rng.binomial(1, treatment_propensity, size=n)
            propensity = np.full(n, treatment_propensity)
        else:
            # Observational setting with selection bias: marketers slightly more likely to contact high-value month-to-month
            confounding_score = 0.5 * is_month_to_month + 0.01 * (monthly_charges - 60.0) + self.rng.normal(0, 0.2, size=n)
            propensity = 1.0 / (1.0 + np.exp(-confounding_score))
            propensity = np.clip(propensity, 0.15, 0.85)
            w = self.rng.binomial(1, propensity, size=n)

        # 5. Realize Counterfactuals and Observed Outcomes
        # Potential outcomes: Y(0) ~ Bern(p_control), Y(1) ~ Bern(p_treated)
        # Use common random numbers for latent resistance to preserve realistic coupling
        latent_threshold = self.rng.uniform(0, 1, size=n)
        y0 = (latent_threshold < p_churn_control).astype(int)
        y1 = (latent_threshold < p_churn_treated).astype(int)

        # Observed churn outcome Y = W * Y(1) + (1 - W) * Y(0)
        y_observed = np.where(w == 1, y1, y0)
        # Observed retention outcome R = 1 - Y
        r_observed = 1 - y_observed

        # Attach trial columns
        df["treatment"] = w
        df["propensity_score"] = propensity
        df["churn"] = y_observed
        df["retained"] = r_observed
        # Store true unobserved counterfactuals for synthetic benchmark validation
        df["true_p_churn_control"] = p_churn_control
        df["true_p_churn_treated"] = p_churn_treated
        df["true_tau_churn"] = true_tau_churn
        df["is_sleeping_dog_true"] = is_sleeping_dog.astype(int)

        return df


class DataLoader:
    """
    Ingests and caches datasets for both Telecom Churn and E-commerce trials.
    """

    TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    HILLSTROM_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"

    def __init__(self, data_dir: Path, random_state: int = 42):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.generator = SyntheticTrialGenerator(random_state=random_state)
        self.random_state = random_state

    def load_telco(
        self,
        is_rct: bool = True,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Loads the Telco Customer Churn dataset and enriches it with the calibrated semi-synthetic
        trial mechanism. If offline or download fails, synthesizes a representative dataset.
        """
        raw_path = self.data_dir / "telco_raw.csv"
        df = None

        if raw_path.exists() and not force_download:
            try:
                df = pd.read_csv(raw_path)
            except Exception:
                df = None

        if df is None:
            try:
                resp = requests.get(self.TELCO_URL, timeout=15)
                if resp.status_code == 200:
                    with open(raw_path, "wb") as f:
                        f.write(resp.content)
                    df = pd.read_csv(raw_path)
            except Exception:
                # Synthesize high-fidelity Telco dataset if network is unavailable
                df = self._synthesize_telco_base(n_samples=7043)
                df.to_csv(raw_path, index=False)

        # Generate trial columns
        trial_df = self.generator.generate_trial(df, is_rct=is_rct)
        return trial_df

    def load_hillstrom(self, force_download: bool = False) -> pd.DataFrame:
        """
        Loads Kevin Hillstrom's MineThatData E-Mail Analytics Trial dataset.
        Target: E-commerce conversion / spend (positive response to increase).
        Treatment: Mens E-Mail / Womens E-Mail vs No E-Mail.
        """
        raw_path = self.data_dir / "hillstrom_raw.csv"
        df = None

        if raw_path.exists() and not force_download:
            try:
                df = pd.read_csv(raw_path)
            except Exception:
                df = None

        if df is None:
            try:
                resp = requests.get(self.HILLSTROM_URL, timeout=15)
                if resp.status_code == 200:
                    with open(raw_path, "wb") as f:
                        f.write(resp.content)
                    df = pd.read_csv(raw_path)
            except Exception:
                df = self._synthesize_hillstrom_base(n_samples=64000)
                df.to_csv(raw_path, index=False)

        # Normalize Hillstrom columns
        # segment: 'Mens E-Mail', 'Womens E-Mail', 'No E-Mail'
        df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
        df["target_conversion"] = df["conversion"].astype(int)
        df["target_visit"] = df["visit"].astype(int)
        df["target_spend"] = df["spend"].astype(float)
        return df

    def _synthesize_telco_base(self, n_samples: int = 7043) -> pd.DataFrame:
        """Generates realistic base Telco dataset if remote source is unreachable."""
        rng = np.random.RandomState(self.random_state)
        customer_ids = [f"{rng.randint(1000, 9999)}-{rng.choice(['A','B','C','D'])}{rng.randint(100, 999)}" for _ in range(n_samples)]
        
        genders = rng.choice(["Male", "Female"], size=n_samples)
        senior = rng.choice([0, 1], p=[0.83, 0.17], size=n_samples)
        partner = rng.choice(["Yes", "No"], p=[0.48, 0.52], size=n_samples)
        dependents = rng.choice(["Yes", "No"], p=[0.30, 0.70], size=n_samples)
        tenure = rng.randint(1, 73, size=n_samples)
        phone = rng.choice(["Yes", "No"], p=[0.90, 0.10], size=n_samples)
        multiple = np.where(phone == "Yes", rng.choice(["Yes", "No"], p=[0.45, 0.55], size=n_samples), "No phone service")
        internet = rng.choice(["Fiber optic", "DSL", "No"], p=[0.44, 0.34, 0.22], size=n_samples)

        def pick_addon():
            return np.where(internet == "No", "No internet service", rng.choice(["Yes", "No"], p=[0.35, 0.65], size=n_samples))

        security = pick_addon()
        backup = pick_addon()
        device_prot = pick_addon()
        tech_support = pick_addon()
        streaming_tv = pick_addon()
        streaming_mov = pick_addon()

        contract = rng.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.21, 0.24], size=n_samples)
        paperless = rng.choice(["Yes", "No"], p=[0.59, 0.41], size=n_samples)
        payment = rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            p=[0.34, 0.23, 0.22, 0.21],
            size=n_samples
        )

        monthly = np.where(
            internet == "Fiber optic", rng.uniform(70, 118, size=n_samples),
            np.where(internet == "DSL", rng.uniform(40, 85, size=n_samples), rng.uniform(18, 25, size=n_samples))
        ).round(2)
        total = (monthly * tenure + rng.normal(0, 15, size=n_samples)).clip(min=18.0).round(2)

        df = pd.DataFrame({
            "customerID": customer_ids,
            "gender": genders,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone,
            "MultipleLines": multiple,
            "InternetService": internet,
            "OnlineSecurity": security,
            "OnlineBackup": backup,
            "DeviceProtection": device_prot,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_mov,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": "No"
        })
        return df

    def _synthesize_hillstrom_base(self, n_samples: int = 64000) -> pd.DataFrame:
        """Generates realistic base Hillstrom dataset if remote source is unreachable."""
        rng = np.random.RandomState(self.random_state)
        recency = rng.randint(1, 13, size=n_samples)
        history = rng.exponential(scale=240, size=n_samples).clip(29.99, 3345.0).round(2)
        mens = rng.choice([0, 1], p=[0.45, 0.55], size=n_samples)
        womens = rng.choice([0, 1], p=[0.45, 0.55], size=n_samples)
        zip_code = rng.choice(["Urban", "Suburban", "Rural"], p=[0.40, 0.45, 0.15], size=n_samples)
        newbie = rng.choice([0, 1], p=[0.50, 0.50], size=n_samples)
        channel = rng.choice(["Web", "Phone", "Multichannel"], p=[0.44, 0.44, 0.12], size=n_samples)
        segment = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], p=[0.333, 0.333, 0.334], size=n_samples)
        
        is_treated = segment != "No E-Mail"
        visit_prob = np.where(is_treated, 0.15 + (history / 5000.0), 0.10 + (history / 5000.0)).clip(0, 0.9)
        visit = rng.binomial(1, visit_prob)
        conv_prob = np.where(visit == 1, 0.08, 0.0)
        conversion = rng.binomial(1, conv_prob)
        spend = np.where(conversion == 1, rng.exponential(scale=110, size=n_samples).round(2), 0.0)

        df = pd.DataFrame({
            "recency": recency,
            "history_segment": "2) $100 - $200",
            "history": history,
            "mens": mens,
            "womens": womens,
            "zip_code": zip_code,
            "newbie": newbie,
            "channel": channel,
            "segment": segment,
            "visit": visit,
            "conversion": conversion,
            "spend": spend
        })
        return df
