"""Best accuracy ANY model could reach on this data (the Bayes-optimal ceiling).

The simulator knows each patient's hidden frailty and the exact formula for the
change in GPH; only the N(0, 5.5) noise term is unpredictable. A model that knew
all hidden variables would predict P(label = 1) = Phi(...) and pick the more likely
class. Its accuracy is the ceiling: no real model, which only sees noisy EHR
proxies, can beat it except by luck.

    python docs/accuracy_ceiling.py

(Approximate: ignores the clipping of T-scores to their 16.2-67.7 range.)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from prodecline.config import CONFIG
from prodecline.data.simulate import EHRSimulator

ROOT = Path(__file__).resolve().parents[1]
NOISE_SD = 5.5          # must match simulate.EHRSimulator._outcome


def main() -> None:
    cfg = CONFIG.cohort
    sim = EHRSimulator(cfg)
    n = int(round(cfg.n_patients * 1.37))         # same draws, same order as generate()
    pts = sim._patients(n)
    levels = sim._lab_levels(pts)
    albumin = next(lv[-1] for lv in levels if lv[0] == "albumin")
    out = sim._outcome(pts, albumin)

    cohort = pd.read_csv(ROOT / "data/processed/cohort.csv", index_col="patient_id")
    test = pd.read_csv(ROOT / "data/processed/split.csv", index_col="patient_id")["split"].eq("test")
    hidden = pd.concat([pts.set_index("patient_id"),
                        out.set_index(pts["patient_id"]),
                        pd.Series(albumin, index=pts["patient_id"], name="alb")], axis=1)
    hidden = hidden.loc[cohort.index]
    assert np.allclose(hidden["_gph_baseline"], cohort["gph_baseline"]), "data mismatch"

    b = hidden["_gph_baseline"]
    low_alb_old = ((hidden["alb"] < 3.5) & (hidden["_age"] > 70)).astype(float)
    mu = (-3.0 - 3.5 * hidden["_frailty"] - 2.0 * hidden["_cancer_effect"]
          - 0.60 * (b - 50) - 3.0 * low_alb_old)          # expected change in GPH
    prob = {"drop": norm.cdf((-10 - mu) / NOISE_SD),
            "threshold": norm.cdf((40 - (b + mu)) / NOISE_SD)}

    for o, p in prob.items():
        y = cohort[o]
        acc = ((p >= 0.5).astype(int) == y)[test].mean()
        print(f"{o:9s}  ceiling test accuracy = {acc:.3f}   ceiling test AUROC = "
              f"{roc_auc_score(y[test], p[test]):.3f}   (always-predict-0 = {1 - y[test].mean():.3f})")


if __name__ == "__main__":
    main()
