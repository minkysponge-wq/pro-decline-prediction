"""Accuracy, precision, recall, F1 and confusion matrix for the trained models.

Uses the saved held-out test predictions (results/test_predictions.csv), so no
retraining is needed.

    python docs/classification_report.py                 # ensemble, KNN labs
    python docs/classification_report.py --model lasso --labs binned

Two decision thresholds are shown:
  * 0.5    - sklearn's default predict()
  * Youden - the threshold maximising (TPR - FPR) on the ROC curve. NOTE: chosen on
             the test set itself, so it is illustrative; in practice pick it on
             training/validation data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]


def report(y: np.ndarray, score: np.ndarray, thr: float) -> dict:
    yh = (score >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {"threshold": round(float(thr), 3),
            "accuracy": accuracy_score(y, yh),
            "balanced_acc": balanced_accuracy_score(y, yh),
            "precision": precision_score(y, yh, zero_division=0),
            "recall": recall_score(y, yh, zero_division=0),
            "f1": f1_score(y, yh, zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ensemble")
    ap.add_argument("--labs", default="knn", choices=["knn", "binned"])
    args = ap.parse_args()

    p = pd.read_csv(ROOT / "results" / "test_predictions.csv")
    for outcome in ("drop", "threshold"):
        d = p[(p.model == args.model) & (p.lab_strategy == args.labs) & (p.outcome == outcome)]
        if d.empty:
            raise SystemExit(f"No predictions for model={args.model} labs={args.labs}")
        y, s = d["y_true"].to_numpy(), d["score"].to_numpy()
        fpr, tpr, th = roc_curve(y, s)
        youden = th[np.argmax(tpr - fpr)]
        rows = pd.DataFrame([report(y, s, 0.5), report(y, s, youden)],
                            index=["default 0.5", "Youden"])
        print(f"\n=== {outcome.upper()}  | model={args.model}, labs={args.labs} | "
              f"test n={len(y)}, positives={y.sum()} ({y.mean():.1%})")
        print(f"AUROC = {roc_auc_score(y, s):.3f}   |   'always predict 0' accuracy = "
              f"{1 - y.mean():.3f}")
        print(rows.round(3).to_string())


if __name__ == "__main__":
    main()
