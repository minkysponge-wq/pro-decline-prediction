"""Cohort construction and outcome labelling (paper Section 3).

Inclusion criteria
  1. Cancer diagnosis on/after 2013-01-01.
  2. Received chemotherapy (index date = first chemotherapy administration).
  3. A pre-treatment PROMIS survey within ``baseline_max_days_before`` days
     before (or on) the index date.
  4. An on-treatment PROMIS survey ``on_treatment_min_days`` to
     ``on_treatment_max_days`` days after the index date.

Labels (PROMIS Global Physical Health T-score, population mean 50, SD 10)
  * drop      : on-treatment GPH - baseline GPH <= -10   (decline of >= 1 SD)
  * threshold : on-treatment GPH <= 40                  (>= 1 SD below mean)
"""
from __future__ import annotations

import logging

import pandas as pd

from ..config import CohortConfig, LabelConfig
from .simulate import RawEHR
from .vocab import CHEMO_MED_CLASS

log = logging.getLogger(__name__)


def index_dates(meds: pd.DataFrame) -> pd.Series:
    """First chemotherapy administration per patient."""
    chemo = meds.loc[meds["med_class"] == CHEMO_MED_CLASS]
    return chemo.groupby("patient_id")["date"].min().rename("index_date")


def build_cohort(ehr: RawEHR, cohort_cfg: CohortConfig,
                 label_cfg: LabelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(cohort, flow)``.

    ``cohort`` has one row per included patient with the index date, baseline
    and on-treatment scores and both binary labels. ``flow`` is a CONSORT-style
    table of how many patients each criterion removed.
    """
    flow = []
    pts = ehr.patients[["patient_id", "cancer_dx_date"]].copy()
    flow.append(("All patients in extract", len(pts)))

    pts = pts.loc[pts["cancer_dx_date"] >= pd.Timestamp(cohort_cfg.min_cohort_date)]
    flow.append((f"Cancer diagnosis on/after {cohort_cfg.min_cohort_date}", len(pts)))

    pts = pts.merge(index_dates(ehr.medications), on="patient_id", how="inner")
    flow.append(("Received chemotherapy", len(pts)))

    sv = ehr.promis.merge(pts[["patient_id", "index_date"]], on="patient_id")
    sv["rel_day"] = (sv["date"] - sv["index_date"]).dt.days

    base = sv.loc[sv["rel_day"].between(-label_cfg.baseline_max_days_before, 0)]
    base = (base.sort_values(["patient_id", "rel_day"])
                .groupby("patient_id").tail(1)              # most recent before chemo
                .set_index("patient_id"))
    pts = pts.loc[pts["patient_id"].isin(base.index)]
    flow.append(("Baseline PROMIS survey <= 90 d before chemo", len(pts)))

    on = sv.loc[sv["rel_day"].between(label_cfg.on_treatment_min_days,
                                      label_cfg.on_treatment_max_days)]
    on = (on.sort_values(["patient_id", "rel_day"])
            .groupby("patient_id").head(1)                  # first on-treatment survey
            .set_index("patient_id"))
    pts = pts.loc[pts["patient_id"].isin(on.index)]
    flow.append(("On-treatment PROMIS survey 30-180 d after chemo start", len(pts)))

    cohort = pts.set_index("patient_id")[["index_date"]].copy()
    cohort["baseline_date"] = base.loc[cohort.index, "date"]
    cohort["gph_baseline"] = base.loc[cohort.index, "gph_tscore"]
    cohort["outcome_date"] = on.loc[cohort.index, "date"]
    cohort["gph_on_treatment"] = on.loc[cohort.index, "gph_tscore"]
    change = cohort["gph_on_treatment"] - cohort["gph_baseline"]
    cohort["drop"] = (change <= -label_cfg.drop_points).astype(int)
    cohort["threshold"] = (cohort["gph_on_treatment"] <= label_cfg.threshold_score).astype(int)

    # Defensive invariants: labels must come strictly from after the index date
    # and the baseline strictly from on/before it.
    assert (cohort["baseline_date"] <= cohort["index_date"]).all()
    assert (cohort["outcome_date"] > cohort["index_date"]).all()

    flow_df = pd.DataFrame(flow, columns=["step", "n_patients"])
    log.info("Cohort: %d patients | drop prevalence %.1f%% | threshold prevalence %.1f%%",
             len(cohort), 100 * cohort["drop"].mean(), 100 * cohort["threshold"].mean())
    return cohort.sort_index(), flow_df
