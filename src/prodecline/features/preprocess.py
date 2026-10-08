"""Leakage-safe preprocessing, fitted on training folds only.

Everything that learns from data across patients is an sklearn transformer
inside the model ``Pipeline``. Consequently, during 10-fold cross-validation
the prevalence filter, lab imputation/binning, scaling and one-hot categories
are re-fitted on each training fold and never see validation or test rows.

Two lab strategies from the paper:
  * ``knn``    - standardise, then K-nearest-neighbour imputation (continuous)
  * ``binned`` - tertile bins learned on the training data + explicit
                 "missing" indicator, one-hot encoded
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from ..config import FeatureConfig


class PrevalenceFilter(BaseEstimator, TransformerMixin):
    """Keep columns that are present (non-missing and non-zero) in at least
    ``min_prevalence`` of training rows (paper: 2.5%)."""

    def __init__(self, min_prevalence: float = 0.025):
        self.min_prevalence = min_prevalence

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        present = X.notna() & (X != 0)
        self.prevalence_ = present.mean(axis=0).to_numpy()
        self.support_ = self.prevalence_ >= self.min_prevalence
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X = pd.DataFrame(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Column count differs from fit.")
        return X.loc[:, self.support_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "support_")
        return self.feature_names_in_[self.support_]


class QuantileBinner(BaseEstimator, TransformerMixin):
    """Bin each column into ``n_bins`` quantile bins learned on training data,
    plus a separate 'missing' indicator; output is one-hot (dense)."""

    def __init__(self, n_bins: int = 3):
        self.n_bins = n_bins

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = [np.unique(np.nanquantile(X[c].to_numpy(dtype=float), qs))
                       if X[c].notna().any() else np.array([])
                       for c in X.columns]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "edges_")
        X = pd.DataFrame(X)
        out = []
        for j, c in enumerate(X.columns):
            v = X[c].to_numpy(dtype=float)
            missing = np.isnan(v)
            idx = np.searchsorted(self.edges_[j], v, side="right")  # 0..n_bins-1
            block = np.zeros((len(v), self.n_bins + 1), dtype=float)
            block[~missing, idx[~missing]] = 1.0
            block[missing, self.n_bins] = 1.0
            out.append(block)
        return np.hstack(out) if out else np.empty((len(X), 0))

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "edges_")
        return np.array([f"{c}__{s}" for c in self.feature_names_in_
                         for s in [f"q{i + 1}" for i in range(self.n_bins)] + ["missing"]],
                        dtype=object)


def _sel(group: str):
    return make_column_selector(pattern=f"^{group}__")


def build_preprocessor(lab_strategy: str, cfg: FeatureConfig) -> Pipeline:
    if lab_strategy == "knn":
        lab_pipe = Pipeline([
            ("prevalence", PrevalenceFilter(cfg.min_prevalence)),
            ("scale", StandardScaler()),                 # KNN distances need scaling
            ("impute", KNNImputer(n_neighbors=cfg.knn_neighbors)),
        ])
    elif lab_strategy == "binned":
        lab_pipe = Pipeline([
            ("prevalence", PrevalenceFilter(cfg.min_prevalence)),
            ("bin", QuantileBinner(cfg.n_lab_bins)),
        ])
    else:
        raise ValueError(f"Unknown lab strategy {lab_strategy!r}")

    numeric = Pipeline([("impute", SimpleImputer(strategy="median"))])
    demo_cat = make_column_selector(pattern="^demo__(?!age$)")

    columns = ColumnTransformer(
        transformers=[
            ("lab", lab_pipe, _sel("lab")),
            ("dx", PrevalenceFilter(cfg.min_prevalence), _sel("dx")),
            ("proc", PrevalenceFilter(cfg.min_prevalence), _sel("proc")),
            ("med", "passthrough", _sel("med")),
            ("vital", numeric, _sel("vital")),
            ("util", numeric, _sel("util")),
            ("promis", numeric, _sel("promis")),
            ("demo_num", numeric, make_column_selector(pattern="^demo__age$")),
            ("demo_cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             demo_cat),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([
        ("columns", columns),
        ("drop_constant", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
    ])
