"""Extension beyond the paper: push classification accuracy as far as is honest.

Two changes, both decided on TRAINING data only (the test set is used once):

1. **Burden features** (``features/burden.py``): patient-level totals that let
   the models pick up the diffuse comorbidity signal.
2. **Decision threshold chosen on out-of-fold training predictions.** A
   classifier's default cut-off of 0.5 is arbitrary for an imbalanced problem
   (~20% positives). We pick the cut-off that maximises accuracy on 10-fold
   out-of-fold predictions of the training set, then apply it unchanged to test.

Run with ``python -m prodecline improve``. Writes ``results/improved/``.
The paper-replication results in ``results/metrics.csv`` are not touched.
"""
from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict

from .config import OUTCOMES, Config
from .data.simulate import RawEHR
from .evaluation.metrics import bootstrap_ci
from .features.burden import build_burden_features
from .models.registry import ENSEMBLE_MEMBERS, get_model_specs
from .models.training import cv_splitter, ensemble_from, tune
from .pipeline import load_dataset

log = logging.getLogger(__name__)
LAB = "knn"
THRESHOLD_GRID = np.round(np.arange(0.05, 0.951, 0.005), 3)


def best_accuracy_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Cut-off maximising accuracy; ties broken towards 0.5 (most conservative)."""
    acc = np.array([accuracy_score(y, p >= t) for t in THRESHOLD_GRID])
    best = np.flatnonzero(acc == acc.max())
    return float(THRESHOLD_GRID[best[np.argmin(np.abs(THRESHOLD_GRID[best] - 0.5))]])


def classification_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    acc, lo, hi = bootstrap_ci(y, yh, metric=accuracy_score, n_boot=1000, seed=0)
    return {"threshold": thr, "accuracy": acc, "accuracy_ci_low": lo, "accuracy_ci_high": hi,
            "balanced_accuracy": balanced_accuracy_score(y, yh),
            "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0),
            "f1": f1_score(y, yh, zero_division=0),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def _evaluate_ensemble(name: str, ens, X: pd.DataFrame, y: pd.Series, tr, te,
                       cfg: Config) -> tuple[list[dict], np.ndarray]:
    """OOF threshold on train -> fit on train -> score test at 0.5 and tuned cut-off."""
    oof = cross_val_predict(clone(ens), X[tr], y[tr], cv=cv_splitter(cfg),
                            method="predict_proba", n_jobs=cfg.train.n_jobs)[:, 1]
    thr = best_accuracy_threshold(y[tr].to_numpy(), oof)
    fitted = clone(ens).fit(X[tr], y[tr])
    p = fitted.predict_proba(X[te])[:, 1]
    yt = y[te].to_numpy()
    auc = roc_auc_score(yt, p)
    rows = []
    for rule, t in (("default 0.5", 0.5), ("tuned on train OOF", thr)):
        rows.append({"variant": name, "cutoff_rule": rule, "test_auroc": auc,
                     "train_oof_auroc": roc_auc_score(y[tr], oof),
                     "train_oof_accuracy": accuracy_score(y[tr], oof >= t),
                     **classification_metrics(yt, p, t)})
    return rows, fitted


def stage_improve(cfg: Config, fast: bool = False) -> pd.DataFrame:
    paths = cfg.paths.ensure()
    out_dir = paths.results_dir / "improved"
    out_dir.mkdir(exist_ok=True)
    X, cohort, split = load_dataset(cfg)
    tr, te = split.eq("train").to_numpy(), split.eq("test").to_numpy()

    cohort_dates = cohort.assign(index_date=pd.to_datetime(cohort["index_date"]))
    burden = build_burden_features(RawEHR.load(paths.raw_dir), cohort_dates, X, cfg.features)
    X_plus = pd.concat([X, burden], axis=1)
    burden.to_csv(paths.processed_dir / "burden_features.csv")
    memory = Memory(paths.cache_dir / "pipelines_improved", verbose=0)

    rows, thresholds = [], {}
    for outcome in OUTCOMES:
        y = cohort[outcome]
        log.info("== improve: outcome=%s", outcome)

        # Baseline: the paper ensemble exactly as trained in `train` (same params).
        base = joblib.load(paths.models_dir / f"{outcome}__{LAB}__ensemble.joblib")
        r, _ = _evaluate_ensemble("paper ensemble", base, X, y, tr, te, cfg)
        rows += [{"outcome": outcome, **x} for x in r]

        # Improved: re-tune the members on paper features + burden features.
        best = {}
        for spec in get_model_specs(cfg.train.seed, fast=fast):
            if spec.name in ENSEMBLE_MEMBERS:
                search, _ = tune(spec, X_plus[tr], y[tr], LAB, cfg, memory)
                best[spec.name] = search.best_estimator_.set_params(memory=None)
        r, fitted = _evaluate_ensemble("ensemble + burden features", ensemble_from(best),
                                       X_plus, y, tr, te, cfg)
        rows += [{"outcome": outcome, **x} for x in r]
        thresholds[outcome] = r[-1]["threshold"]
        joblib.dump(fitted, out_dir / f"{outcome}__improved_ensemble.joblib")
        for x in rows[-4:]:
            log.info("  %-28s %-20s acc %.3f  AUROC %.3f  recall %.3f  (cut-off %.3f)",
                     x["variant"], x["cutoff_rule"], x["accuracy"], x["test_auroc"],
                     x["recall"], x["threshold"])

    df = pd.DataFrame(rows)
    df["majority_class_accuracy"] = df["outcome"].map(
        {o: 1 - cohort.loc[te, o].mean() for o in OUTCOMES})
    df.to_csv(out_dir / "metrics.csv", index=False)
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    log.info("Saved %s", out_dir / "metrics.csv")
    return df


def summary(cfg: Config) -> None:
    df = pd.read_csv(cfg.paths.results_dir / "improved" / "metrics.csv")
    for o, d in df.groupby("outcome", sort=False):
        print(f"\n### {o}   (always-predict-0 accuracy: {d.majority_class_accuracy.iloc[0]:.3f})")
        d = d.assign(accuracy=d.apply(lambda r: f"{r.accuracy:.3f} ({r.accuracy_ci_low:.3f}-"
                                                f"{r.accuracy_ci_high:.3f})", axis=1))
        print(d[["variant", "cutoff_rule", "threshold", "accuracy", "balanced_accuracy",
                 "recall", "precision", "f1", "test_auroc"]].round(3).to_string(index=False))
