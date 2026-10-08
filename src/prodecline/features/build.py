"""Turn raw event tables into one row per patient (paper Section 4).

This step is *label-free and per-patient*: every feature of a patient depends
only on that patient's own events dated on/before their index date. Nothing is
fitted across patients here, so it is safe to run on the whole cohort before
the train/test split. Everything that *is* fitted across patients (prevalence
filtering, imputation, binning, scaling, one-hot categories) lives in
``features.preprocess`` and is fitted on training folds only.

Column naming convention: ``<group>__<feature>`` where group is one of
``config.FEATURE_GROUPS``.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from ..data.simulate import RawEHR
from ..data.vocab import CHEMO_MED_CLASS, VITALS

log = logging.getLogger(__name__)

VITAL_BOUNDS = {name: bounds for name, *_, bounds in VITALS}
DEMO_CATEGORICAL = ["sex", "race", "ethnicity", "insurance", "language",
                    "marital_status", "cancer_type"]


def time_weight(days_before: np.ndarray | pd.Series, decay_days: float) -> np.ndarray:
    """Paper's exponential decay weight ``exp(-t / 365)``; t = days before index."""
    t = np.asarray(days_before, dtype=float)
    if (t < 0).any():
        raise ValueError("time_weight received post-index events (negative t).")
    return np.exp(-t / decay_days)


def _windowed(events: pd.DataFrame, cohort: pd.DataFrame,
              lookback_days: int | None) -> pd.DataFrame:
    """Attach index date, keep events on/before index (and within look-back)."""
    ev = events.merge(cohort[["index_date"]], left_on="patient_id", right_index=True)
    ev["days_before"] = (ev["index_date"] - ev["date"]).dt.days
    keep = ev["days_before"] >= 0                      # on or before chemo start
    if lookback_days is not None:
        keep &= ev["days_before"] <= lookback_days
    return ev.loc[keep]


def _weighted_mean(ev: pd.DataFrame, key: str, value: str, decay: float) -> pd.DataFrame:
    ev = ev.assign(w=time_weight(ev["days_before"], decay))
    ev = ev.assign(wv=ev["w"] * ev[value])
    g = ev.groupby(["patient_id", key])[["wv", "w"]].sum()
    return (g["wv"] / g["w"]).unstack(key)


def _weighted_count(ev: pd.DataFrame, key: str, decay: float) -> pd.DataFrame:
    ev = ev.assign(w=time_weight(ev["days_before"], decay))
    return ev.groupby(["patient_id", key])["w"].sum().unstack(key)


def _binary(ev: pd.DataFrame, key: str) -> pd.DataFrame:
    return (ev.assign(v=1).groupby(["patient_id", key])["v"].max()
              .unstack(key))


def _prefix(df: pd.DataFrame, group: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = [f"{group}__{c}" for c in df.columns]
    df.columns.name = None
    return df


def clean_vitals(vitals: pd.DataFrame) -> pd.DataFrame:
    """Drop physiologically impossible readings (paper: clinician-defined bounds)."""
    lo = vitals["vital"].map(lambda v: VITAL_BOUNDS[v][0])
    hi = vitals["vital"].map(lambda v: VITAL_BOUNDS[v][1])
    ok = vitals["value"].between(lo, hi)
    log.info("Removed %d impossible vital readings (%.2f%%)", (~ok).sum(), 100 * (~ok).mean())
    return vitals.loc[ok]


def build_features(ehr: RawEHR, cohort: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Return the patient x feature matrix (index = patient_id, same order as cohort)."""
    lb, decay = cfg.lookback_days, cfg.decay_days
    blocks = []

    # vitals: exponentially weighted mean of cleaned readings (last 6 months)
    v = _windowed(clean_vitals(ehr.vitals), cohort, lb)
    blocks.append(_prefix(_weighted_mean(v, "vital", "value", decay), "vital"))

    # labs: exponentially weighted mean; NaN = never measured (handled later)
    lab = _windowed(ehr.labs, cohort, lb)
    blocks.append(_prefix(_weighted_mean(lab, "lab", "value", decay), "lab"))

    # procedures: exponentially weighted count (last 6 months)
    pr = _windowed(ehr.procedures, cohort, lb)
    blocks.append(_prefix(_weighted_count(pr, "proc_code", decay), "proc").fillna(0.0))

    # diagnoses: binary, *full history* (paper keeps old diagnoses)
    dx = _windowed(ehr.diagnoses, cohort, None)
    blocks.append(_prefix(_binary(dx, "dx_code"), "dx").fillna(0).astype(np.int8))

    # medication classes: binary in last 6 months; chemo itself defines the index
    md = _windowed(ehr.medications.loc[ehr.medications["med_class"] != CHEMO_MED_CLASS],
                   cohort, lb)
    blocks.append(_prefix(_binary(md, "med_class"), "med").fillna(0).astype(np.int8))

    # healthcare utilisation: encounter counts by type in last 6 months
    en = _windowed(ehr.encounters, cohort, lb)
    util = en.groupby(["patient_id", "encounter_type"]).size().unstack("encounter_type")
    blocks.append(_prefix(util, "util").fillna(0).astype(float))

    # demographics: age at index + raw categoricals (one-hot encoded in-pipeline)
    pts = ehr.patients.set_index("patient_id").loc[cohort.index]
    demo = pd.DataFrame(index=cohort.index)
    demo["age"] = ((cohort["index_date"] - pts["birth_date"]).dt.days / 365.25).round(1)
    for col in DEMO_CATEGORICAL:
        demo[col] = pts[col].astype(str)
    blocks.append(_prefix(demo, "demo"))

    # baseline PROMIS (6 features)
    blocks.append(_promis_features(ehr, cohort))

    X = pd.concat(blocks, axis=1).reindex(cohort.index)
    # patients with zero events in a count/binary group get 0, not NaN
    for grp in ("proc", "dx", "med", "util"):
        cols = [c for c in X.columns if c.startswith(f"{grp}__")]
        X[cols] = X[cols].fillna(0)
    X.index.name = "patient_id"
    log.info("Feature matrix: %d patients x %d raw features", *X.shape)
    return X


def _promis_features(ehr: RawEHR, cohort: pd.DataFrame) -> pd.DataFrame:
    sv = _windowed(ehr.promis, cohort, None)
    sv = sv.sort_values(["patient_id", "days_before"])          # most recent first
    rank = sv.groupby("patient_id").cumcount()                  # 0 = most recent
    latest = sv.loc[rank == 0].set_index("patient_id")
    prior = sv.loc[rank > 0].groupby("patient_id")["gph_tscore"].mean()
    out = pd.DataFrame(index=cohort.index)
    out["gph_baseline"] = latest["gph_tscore"]
    out["gmh_baseline"] = latest["gmh_tscore"]
    out["days_since_baseline"] = latest["days_before"].astype(float)
    out["n_prior_surveys"] = sv.groupby("patient_id").size().reindex(cohort.index) - 1
    out["gph_prior_mean"] = prior
    out["gph_recent_change"] = out["gph_baseline"] - out["gph_prior_mean"]
    # sanity: the baseline feature must equal the cohort's baseline used for labels
    assert np.allclose(out["gph_baseline"], cohort["gph_baseline"]), "baseline mismatch"
    return _prefix(out, "promis")
