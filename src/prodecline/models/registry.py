"""Model zoo with hyper-parameter grids (paper Section 5).

Grids follow the paper; where the paper's grid is computationally excessive
for a laptop the reduction is noted inline. ``fast=True`` gives a much smaller
grid for quick demos and CI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import sklearn
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

_SK_GE_18 = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)


def logistic(kind: str, seed: int) -> LogisticRegression:
    """Version-safe penalised logistic regression.

    scikit-learn 1.8 deprecated ``penalty=`` in favour of ``l1_ratio``; older
    versions *silently ignore* ``l1_ratio`` unless ``penalty='elasticnet'``.
    This helper produces the right estimator on either side of that change.
    """
    common = dict(max_iter=5000, random_state=seed)
    if kind == "lasso":
        return (LogisticRegression(l1_ratio=1.0, solver="liblinear", **common) if _SK_GE_18
                else LogisticRegression(penalty="l1", solver="liblinear", **common))
    if kind == "ridge":
        return (LogisticRegression(l1_ratio=0.0, solver="lbfgs", **common) if _SK_GE_18
                else LogisticRegression(penalty="l2", solver="lbfgs", **common))
    if kind == "elasticnet":
        return (LogisticRegression(l1_ratio=0.5, solver="saga", **common) if _SK_GE_18
                else LogisticRegression(penalty="elasticnet", l1_ratio=0.5,
                                        solver="saga", **common))
    raise ValueError(kind)


@dataclass
class ModelSpec:
    name: str
    family: str            # "logistic" | "tree" | "svm" | "neural" | "neighbors"
    estimator: object
    grid: dict = field(default_factory=dict)


def _c_grid(lambdas: np.ndarray) -> list[float]:
    """Paper parameterises by lambda; sklearn uses C = 1 / lambda."""
    return sorted(float(1.0 / lam) for lam in lambdas)


def get_model_specs(seed: int, fast: bool = False) -> list[ModelSpec]:
    # paper: lambda in 10^-2 ... 10^8 (11 values)
    lam_full = 10.0 ** np.arange(-2, 9)
    lam = 10.0 ** np.array([-1, 1, 3]) if fast else lam_full

    specs = [
        ModelSpec("lasso", "logistic", logistic("lasso", seed), {"C": _c_grid(lam)}),
        ModelSpec("ridge", "logistic", logistic("ridge", seed), {"C": _c_grid(lam)}),
        ModelSpec(
            "elasticnet", "logistic", logistic("elasticnet", seed),
            # paper: 11 lambdas x 9 L1-ratios (99 fits/fold). saga is slow, so we
            # keep the informative range of lambda (10^-1..10^3) x 5 ratios.
            {"C": _c_grid(10.0 ** np.array([1, 2]) if fast else 10.0 ** np.arange(-1, 4)),
             "l1_ratio": [0.5] if fast else [0.1, 0.3, 0.5, 0.7, 0.9]},
        ),
        ModelSpec(
            "random_forest", "tree",
            # paper: 1000 trees, max_features in {all, sqrt}. "all" with ~600
            # features x 1000 trees x 10 folds is impractical on a laptop, so we
            # use 500 trees and max_features in {sqrt, 30%}.
            RandomForestClassifier(n_estimators=200 if fast else 500, n_jobs=1,
                                   random_state=seed),
            {"min_samples_split": [10, 100] if fast else [2, 5, 10, 100],
             "max_features": ["sqrt"] if fast else ["sqrt", 0.3]},
        ),
        ModelSpec(
            "lightgbm", "tree",
            # paper tuned learning rate & leaves with early stopping *on the test
            # set* (a leak the authors acknowledge). We tune n_estimators in CV.
            LGBMClassifier(subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
                           min_child_samples=20, reg_lambda=1.0, random_state=seed,
                           n_jobs=1, verbose=-1),
            {"learning_rate": [0.05] if fast else [0.01, 0.05, 0.1],
             "num_leaves": [4, 15] if fast else [4, 8, 16, 31],
             "n_estimators": [200] if fast else [100, 300]},
        ),
        ModelSpec(
            "mlp", "neural",
            # paper: 3 hidden layers (100-1000, 50-500, 10), Adam, early stopping
            # after 5 epochs w/o improvement. Early stopping uses an internal
            # validation split carved from the *training fold* only.
            MLPClassifier(solver="adam", early_stopping=True, validation_fraction=0.15,
                          n_iter_no_change=5, max_iter=500, random_state=seed),
            {"alpha": [1e-2, 1e1] if fast else [1e-4, 1e-2, 1e0, 1e1, 1e2],
             "hidden_layer_sizes": [(100, 50, 10)] if fast
             else [(100, 50, 10), (300, 100, 10), (1000, 500, 10)]},
        ),
        ModelSpec(
            "svm", "svm",
            # AUROC uses decision_function during search; Platt scaling is added
            # (CalibratedClassifierCV) only when the SVM joins the ensemble.
            SVC(max_iter=200_000, cache_size=500, random_state=seed),
            {"C": [1e-2, 1e0] if fast else [float(c) for c in 10.0 ** np.arange(-4, 5)],
             "kernel": ["linear", "rbf"] if fast else ["linear", "poly", "rbf", "sigmoid"]},
        ),
        ModelSpec(
            "knn", "neighbors", KNeighborsClassifier(),
            # paper: k from 1 to 100 on a log scale, uniform or distance weights
            {"n_neighbors": [5, 50] if fast else [1, 2, 5, 10, 20, 50, 100],
             "weights": ["uniform", "distance"]},
        ),
    ]
    return specs


# Which tuned models are averaged in the soft-voting ensemble (paper: the best
# logistic models, the two best tree models and the best SVM).
ENSEMBLE_MEMBERS = ("lasso", "ridge", "elasticnet", "random_forest", "lightgbm", "svm")
