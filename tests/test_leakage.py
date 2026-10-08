"""The most important tests: no information from after the index date (or from
the test set) may reach the model."""
import dataclasses

import numpy as np
import pandas as pd

from prodecline.config import CONFIG
from prodecline.data.simulate import RawEHR
from prodecline.features.build import build_features
from prodecline.features.preprocess import PrevalenceFilter


def _drop_post_index(ehr: RawEHR, cohort: pd.DataFrame) -> RawEHR:
    idx = cohort["index_date"]
    out = {}
    for name in RawEHR.TABLES:
        df = getattr(ehr, name)
        if "date" in df.columns:
            d = idx.reindex(df["patient_id"]).to_numpy()
            df = df.loc[~(df["date"].to_numpy() > d)]
        out[name] = df
    return RawEHR(**out)


def test_simulator_does_create_post_index_events(ehr, cohort):
    """Guard: the leakage test below is only meaningful if traps exist."""
    labs = ehr.labs.merge(cohort[["index_date"]], left_on="patient_id", right_index=True)
    assert (labs["date"] > labs["index_date"]).sum() > 1000


def test_features_identical_without_post_index_events(ehr, cohort, features):
    rebuilt = build_features(_drop_post_index(ehr, cohort), cohort, CONFIG.features)
    pd.testing.assert_frame_equal(features, rebuilt)


def test_label_columns_not_in_features(features):
    banned = {"drop", "threshold", "gph_on_treatment", "outcome_date"}
    assert not banned & {c.split("__")[-1] for c in features.columns}
    assert all("__" in c for c in features.columns)


def test_prevalence_filter_uses_only_training_rows():
    X = pd.DataFrame({"a": [0] * 99 + [1], "b": [1] * 50 + [0] * 50})
    # 'a' is 1% prevalent in the training rows -> dropped, even though it is
    # common in the "test" rows we transform afterwards
    f = PrevalenceFilter(0.025).fit(X)
    X_test = pd.DataFrame({"a": [1] * 10, "b": [1] * 10})
    assert list(f.transform(X_test).columns) == ["b"]


def test_baseline_survey_precedes_index(cohort):
    assert (cohort["baseline_date"] <= cohort["index_date"]).all()
    assert (cohort["outcome_date"] > cohort["index_date"]).all()


def test_lookback_window_respected(ehr, cohort):
    """Labs older than 180 days must not influence features; diagnoses may."""
    cfg = dataclasses.replace(CONFIG.features)
    old = ehr.labs.merge(cohort[["index_date"]], left_on="patient_id", right_index=True)
    old = old.loc[(old["index_date"] - old["date"]).dt.days > cfg.lookback_days]
    assert len(old) > 0
    ehr2 = dataclasses.replace(ehr, labs=ehr.labs.drop(index=old.index.intersection(ehr.labs.index)))
    f1 = build_features(ehr, cohort, cfg)
    f2 = build_features(ehr2, cohort, cfg)
    lab_cols = [c for c in f1 if c.startswith("lab__")]
    np.testing.assert_allclose(f1[lab_cols].to_numpy(dtype=float),
                               f2[lab_cols].to_numpy(dtype=float), equal_nan=True)
