"""Central configuration.

Every constant that comes from the reference paper (Ostberg & Peterson, CS229
Spring 2020) is defined here once, with a comment pointing at its origin, so
the rest of the codebase never hard-codes a magic number.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    results_dir: Path = PROJECT_ROOT / "results"
    figures_dir: Path = PROJECT_ROOT / "results" / "figures"
    models_dir: Path = PROJECT_ROOT / "results" / "models"
    cache_dir: Path = PROJECT_ROOT / ".cache"

    def ensure(self) -> "Paths":
        for p in (self.raw_dir, self.processed_dir, self.results_dir,
                  self.figures_dir, self.models_dir, self.cache_dir):
            p.mkdir(parents=True, exist_ok=True)
        return self


@dataclass(frozen=True)
class CohortConfig:
    """Parameters of the synthetic cohort (sized like the paper's cohort)."""
    n_patients: int = 1252                 # target cohort size (paper: 1,252)
    seed: int = 229
    n_labs: int = 49                       # distinct lab tests (12 informative + filler)
    n_diagnoses: int = 320                 # raw dx codes; ~55% survive the 2.5% filter
    n_procedures: int = 340                # raw procedure codes; ~45% survive the filter
    n_med_classes: int = 121               # paper: 121 medication classes
    min_cohort_date: str = "2013-01-01"    # paper: diagnosis after Jan 1 2013


@dataclass(frozen=True)
class FeatureConfig:
    lookback_days: int = 180               # paper: drop data > 6 months before chemo (except dx)
    decay_days: float = 365.0              # paper: weight = exp(-t / 365)
    min_prevalence: float = 0.025          # paper: drop dx/labs/procs in < 2.5% of cohort
    knn_neighbors: int = 5                 # KNN lab imputation
    n_lab_bins: int = 3                    # paper: tertiles + missing indicator


@dataclass(frozen=True)
class LabelConfig:
    drop_points: float = 10.0              # paper: decline >= 10 pts (1 SD) = "drop"
    threshold_score: float = 40.0          # paper: on-chemo GPH <= 40 = "threshold"
    on_treatment_min_days: int = 30        # first on-treatment survey >= 30 d after start
    on_treatment_max_days: int = 180
    baseline_max_days_before: int = 90     # baseline survey within 90 d before start


@dataclass(frozen=True)
class TrainConfig:
    test_size: float = 0.20
    cv_folds: int = 10                     # paper: 10-fold CV
    scoring: str = "roc_auc"               # paper: AUROC
    seed: int = 229
    n_jobs: int = -1
    n_bootstrap: int = 1000                # for 95% CIs on the test set


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    cohort: CohortConfig = field(default_factory=CohortConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    labels: LabelConfig = field(default_factory=LabelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


OUTCOMES = ("drop", "threshold")
LAB_STRATEGIES = ("knn", "binned")

# Feature-group prefixes. Every engineered column is named "<group>__<name>",
# which lets preprocessing and ablation select groups without bookkeeping.
FEATURE_GROUPS = ("vital", "lab", "proc", "dx", "med", "util", "demo", "promis")

CONFIG = Config()
