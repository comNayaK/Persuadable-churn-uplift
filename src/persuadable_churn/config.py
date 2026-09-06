from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CampaignEconomics:
    """Economic parameters for retention campaigns."""
    retention_discount_pct: float = 0.10  # 10% discount offer on monthly bill
    outreach_cost: float = 1.50           # Contact cost per user ($1.50 direct mail/SMS/call)
    default_clv_months: float = 12.0      # Expected customer lifetime in months post-retention
    gross_margin_pct: float = 0.70        # Gross margin percentage for telecom services
    
    def break_even_uplift(self, monthly_charge: float) -> float:
        """
        Calculate the minimal retention uplift required to break even on outreach + discount.
        Net Margin = (monthly_charge * (1 - discount) * gross_margin * months) * tau - outreach_cost
        Break-even tau = outreach_cost / (monthly_charge * (1 - discount) * gross_margin * months)
        """
        margin_per_retained = monthly_charge * (1.0 - self.retention_discount_pct) * self.gross_margin_pct * self.default_clv_months
        if margin_per_retained <= 0:
            return 1.0
        return self.outreach_cost / margin_per_retained


@dataclass
class QuadrantThresholds:
    """
    Data-driven & quantile-based thresholds for segmenting customers into 4 behavioral quadrants.
    Avoids arbitrary static floats that fail on real-world clustered CATE distributions.
    """
    # Use quantile-based boundaries by default
    use_quantiles: bool = True
    
    # Quantile cutoffs:
    # Persuadables: top quantile of positive uplift (e.g. top 15% uplift)
    persuadable_quantile: float = 0.85
    # Sleeping Dogs / Do-Not-Disturb: bottom 2% - 5% of uplift distribution where tau < 0
    sleeping_dog_quantile: float = 0.03
    # Churn risk split separating Sure Things from Lost Causes (median or 50th percentile)
    churn_risk_split_quantile: float = 0.50

    # Fallback absolute thresholds (used if use_quantiles is False)
    min_persuadable_uplift: float = 0.03
    max_sleeping_dog_uplift: float = -0.01
    baseline_churn_risk_threshold: float = 0.40


@dataclass
class ModelConfig:
    """Machine learning and causal estimation configuration."""
    random_state: int = 42
    test_size: float = 0.20
    cv_folds: int = 5
    baseline_model_type: str = "xgboost"  # "xgboost", "lightgbm", "random_forest"
    causal_learner_type: str = "x_learner"  # "t_learner", "x_learner"
    n_estimators: int = 150
    max_depth: int = 4
    learning_rate: float = 0.05


@dataclass
class PipelineConfig:
    """Paths and runtime settings."""
    workspace_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    data_dir: Path = field(init=False)
    artifacts_dir: Path = field(init=False)
    economics: CampaignEconomics = field(default_factory=CampaignEconomics)
    quadrant: QuadrantThresholds = field(default_factory=QuadrantThresholds)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self):
        self.data_dir = self.workspace_root / "data"
        self.artifacts_dir = self.workspace_root / "artifacts"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
