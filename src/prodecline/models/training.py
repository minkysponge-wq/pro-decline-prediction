"""Model fitting: train/test split, CV grid search and the voting ensemble.

Protocol (paper Section 5, with the paper's test-set early-stopping leak fixed)
  1. Single stratified 80/20 patient-level train/test split, saved to disk.
  2. For each model family: grid search with stratified 10-fold CV on the
     training set, scored by AUROC. Preprocessing is inside the pipeline, so it
     is re-fitted on every training fold.
  3. Best configuration is refitted on the full training set and evaluated
     *once* on the held-out test set.
  4. Ensemble = soft vote (mean predicted probability) of the tuned
     ``ENSEMBLE_MEMBERS``.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from ..config import Config
from ..features.preprocess import build_preprocessor
from .registry import ENSEMBLE_MEMBERS, ModelSpec

log = logging.getLogger(__name__)


def split_patients(cohort: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return a Series patient_id -> 'train' | 'test', stratified on both labels."""
    strata = cohort["drop"].astype(str) + "_" + cohort["threshold"].astype(str)
    train_ids, test_ids = train_test_split(
        cohort.index.to_numpy(), test_size=cfg.train.test_size,
        stratify=strata, random_state=cfg.train.seed)
    split = pd.Series("train", index=cohort.index, name="split")
    split.loc[test_ids] = "test"
    assert set(train_ids).isdisjoint(test_ids)
    return split


def make_pipeline(estimator, lab_strategy: str, cfg: Config,
                  memory: Memory | None = None) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(lab_strategy, cfg.features)),
                     ("clf", clone(estimator))], memory=memory)


def cv_splitter(cfg: Config) -> StratifiedKFold:
    return StratifiedKFold(n_splits=cfg.train.cv_folds, shuffle=True,
                           random_state=cfg.train.seed)


def tune(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, lab_strategy: str,
         cfg: Config, memory: Memory | None = None) -> tuple[GridSearchCV, float]:
    grid = {f"clf__{k}": v for k, v in spec.grid.items()}
    search = GridSearchCV(
        make_pipeline(spec.estimator, lab_strategy, cfg, memory),
        param_grid=grid, scoring=cfg.train.scoring, cv=cv_splitter(cfg),
        n_jobs=cfg.train.n_jobs, refit=True, error_score="raise")
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0
    n_cfg = int(np.prod([len(v) for v in spec.grid.values()]))
    log.info("  %-14s %3d configs  CV AUROC %.3f  (%.0fs)  best=%s", spec.name, n_cfg,
             search.best_score_, elapsed,
             {k.replace("clf__", ""): v for k, v in search.best_params_.items()})
    return search, elapsed


def ensemble_from(best: dict[str, Pipeline]) -> VotingClassifier:
    """Soft-voting ensemble of the tuned member pipelines (unfitted clones)."""
    missing = [m for m in ENSEMBLE_MEMBERS if m not in best]
    if missing:
        raise KeyError(f"Ensemble members not trained: {missing}")
    members = []
    for name in ENSEMBLE_MEMBERS:
        pipe = clone(best[name])
        if name == "svm":
            # Platt scaling so the SVM can contribute a probability to the vote.
            # (Version-safe replacement for the deprecated SVC(probability=True).)
            pipe.set_params(clf=CalibratedClassifierCV(pipe.named_steps["clf"],
                                                       method="sigmoid", cv=5,
                                                       ensemble=False))
        pipe.set_params(memory=None)                  # no cache inside the ensemble
        members.append((name, pipe))
    return VotingClassifier(members, voting="soft", n_jobs=1)


def cv_auroc(estimator, X, y, cfg: Config) -> tuple[float, float]:
    scores = cross_val_score(estimator, X, y, scoring=cfg.train.scoring,
                             cv=cv_splitter(cfg), n_jobs=cfg.train.n_jobs)
    return float(scores.mean()), float(scores.std())
