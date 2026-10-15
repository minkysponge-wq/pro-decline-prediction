"""Extension features (not in the paper): comorbidity / care burden summaries.

The latent health state shows up as many weak, sparse signals spread across
hundreds of diagnosis, medication and procedure columns. Linear models with a
strong penalty struggle to add up many tiny effects, so we hand them simple
patient-level totals - the same idea as a comorbidity index:

* ``burden__n_dx`` / ``n_med`` / ``n_proc``   number of distinct codes present
* ``burden__n_lab_tests`` / ``n_distinct_labs`` how intensively the patient was tested
* ``burden__n_encounters``                    total visits in the look-back window
* ``burden__abnormal_labs``                   count of abnormal prognostic labs using
                                              *fixed clinical cut-offs* (modified Glasgow
                                              Prognostic Score: CRP, albumin; plus NLR,
                                              haemoglobin, LDH - all standard oncology
                                              prognostic markers)

Everything here is per-patient, label-free, uses only events on/before the
index date, and uses fixed constants (no statistics learned from the cohort),
so it is safe to compute before the train/test split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..data.simulate import RawEHR
from .build import _windowed

# (lab, direction, clinical cut-off): direction +1 means "above is abnormal"
ABNORMAL_LAB_RULES = (
    ("c_reactive_protein", +1, 10.0),      # mg/L   (mGPS)
    ("albumin", -1, 3.5),                  # g/dL   (mGPS)
    ("neutrophil_lymph_ratio", +1, 3.0),   # NLR
    ("hemoglobin", -1, 12.0),              # g/dL   (anaemia)
    ("lactate_dehydrogenase", +1, 250.0),  # U/L
)


def build_burden_features(ehr: RawEHR, cohort: pd.DataFrame, X: pd.DataFrame,
                          cfg: FeatureConfig) -> pd.DataFrame:
    """Return ``burden__*`` columns aligned to ``X`` (the paper feature matrix)."""
    out = pd.DataFrame(index=X.index)
    for grp in ("dx", "med", "proc"):
        cols = [c for c in X.columns if c.startswith(f"{grp}__")]
        out[f"n_{grp}"] = (X[cols] > 0).sum(axis=1)

    labs = _windowed(ehr.labs, cohort, cfg.lookback_days)
    out["n_lab_tests"] = labs.groupby("patient_id").size()
    out["n_distinct_labs"] = labs.groupby("patient_id")["lab"].nunique()
    out["n_encounters"] = X[[c for c in X.columns if c.startswith("util__")]].sum(axis=1)

    abnormal = np.zeros(len(X), dtype=int)
    for lab, direction, cut in ABNORMAL_LAB_RULES:
        v = X[f"lab__{lab}"]                       # NaN (not measured) counts as normal
        abnormal += ((v > cut) if direction > 0 else (v < cut)).fillna(False).to_numpy(int)
    out["abnormal_labs"] = abnormal

    out = out.fillna(0).astype(float)
    out.columns = [f"burden__{c}" for c in out.columns]
    return out
