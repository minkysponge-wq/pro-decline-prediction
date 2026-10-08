"""Evaluation metrics with bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def predict_score(estimator, X) -> tuple[np.ndarray, bool]:
    """Positive-class score and whether it is a calibrated-scale probability.

    Falls back to ``decision_function`` (e.g. SVC without Platt scaling); AUROC
    only needs a ranking, but Brier score needs probabilities.
    """
    try:
        return estimator.predict_proba(X)[:, 1], True
    except AttributeError:
        return estimator.decision_function(X), False


def bootstrap_ci(y_true, y_score, metric=roc_auc_score, n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap (1 - alpha) CI."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.default_rng(seed)
    stats = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():   # need both classes
            continue
        stats.append(metric(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y_true, y_score)), float(lo), float(hi)


def evaluate(y_true, y_score, n_boot: int, seed: int, is_probability: bool) -> dict:
    auc, lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot, seed=seed)
    out = {"test_auroc": auc, "test_auroc_ci_low": lo, "test_auroc_ci_high": hi,
           "test_auprc": float(average_precision_score(y_true, y_score)),
           "test_prevalence": float(np.mean(y_true))}
    out["test_brier"] = (float(brier_score_loss(y_true, y_score))
                         if is_probability else np.nan)
    return out
