import numpy as np
import pandas as pd
import pytest

from prodecline.config import CONFIG
from prodecline.features.preprocess import QuantileBinner, build_preprocessor
from prodecline.models.registry import ENSEMBLE_MEMBERS, get_model_specs, logistic
from prodecline.models.training import ensemble_from, make_pipeline


@pytest.mark.parametrize("strategy", ["knn", "binned"])
def test_preprocessor_output_is_finite(features, strategy):
    Z = build_preprocessor(strategy, CONFIG.features).fit_transform(features)
    assert np.isfinite(Z).all()
    assert Z.shape[0] == len(features)


def test_preprocessor_handles_missing_group(features):
    """Ablation removes whole groups; the pipeline must still work."""
    X = features.drop(columns=features.filter(regex="^lab__").columns)
    Z = build_preprocessor("knn", CONFIG.features).fit_transform(X)
    assert np.isfinite(Z).all()


def test_quantile_binner_one_hot():
    X = pd.DataFrame({"a": [1, 2, 3, 4, 5, 6, np.nan, np.nan, 7]})
    Z = QuantileBinner(3).fit(X).transform(X)
    assert Z.shape == (9, 4)
    assert (Z.sum(axis=1) == 1).all()             # exactly one bin per row
    assert Z[6, 3] == 1 and Z[7, 3] == 1          # missing indicator


def test_lasso_is_really_l1():
    """Guards against sklearn versions silently ignoring the L1 penalty."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 40))
    y = (X[:, 0] + rng.normal(size=300) > 0).astype(int)
    lasso = logistic("lasso", 0).set_params(C=0.05).fit(X, y)
    ridge = logistic("ridge", 0).set_params(C=0.05).fit(X, y)
    assert (lasso.coef_ == 0).sum() > 20
    assert (ridge.coef_ == 0).sum() == 0


def test_every_model_and_ensemble_fit(features, cohort):
    y = cohort["drop"]
    best = {}
    for spec in get_model_specs(seed=0, fast=True):
        params = {f"clf__{k}": v[0] for k, v in spec.grid.items()}
        pipe = make_pipeline(spec.estimator, "knn", CONFIG).set_params(**params)
        best[spec.name] = pipe.fit(features, y)
    ens = ensemble_from(best).fit(features, y)
    p = ens.predict_proba(features)[:, 1]
    assert p.shape == (len(features),) and ((p >= 0) & (p <= 1)).all()
    assert [n for n, _ in ens.estimators] == list(ENSEMBLE_MEMBERS)
