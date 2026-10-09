import numpy as np
import pandas as pd
import pytest

from prodecline.data.simulate import EHRSimulator
from prodecline.features.build import clean_vitals, time_weight
from tests.conftest import SMALL


def test_simulation_is_deterministic():
    a = EHRSimulator(SMALL).generate()
    b = EHRSimulator(SMALL).generate()
    for name in a.TABLES:
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))


def test_labels_follow_paper_definitions(cohort):
    change = cohort["gph_on_treatment"] - cohort["gph_baseline"]
    assert (cohort["drop"] == (change <= -10).astype(int)).all()
    assert (cohort["threshold"] == (cohort["gph_on_treatment"] <= 40).astype(int)).all()
    assert 0.05 < cohort["drop"].mean() < 0.5
    assert 0.05 < cohort["threshold"].mean() < 0.5


def test_inclusion_criteria(cohort):
    rel_on = (cohort["outcome_date"] - cohort["index_date"]).dt.days
    rel_base = (cohort["index_date"] - cohort["baseline_date"]).dt.days
    assert rel_on.between(30, 180).all()
    assert rel_base.between(0, 90).all()


def test_time_weight():
    w = time_weight(np.array([0, 365, 730]), 365.0)
    np.testing.assert_allclose(w, [1.0, np.exp(-1), np.exp(-2)])
    with pytest.raises(ValueError):
        time_weight(np.array([-1]), 365.0)


def test_clean_vitals_removes_impossible_values():
    df = pd.DataFrame({"vital": ["heart_rate", "heart_rate", "spo2", "spo2"],
                       "value": [80, 0, 97, 970]})
    assert clean_vitals(df)["value"].tolist() == [80, 97]


def test_feature_matrix_shape_and_types(features, cohort):
    assert features.index.equals(cohort.index)
    groups = {c.split("__")[0] for c in features.columns}
    assert groups == {"vital", "lab", "proc", "dx", "med", "util", "demo", "promis"}
    # count/binary groups never missing; labs may be missing (imputed later)
    for g in ("proc", "dx", "med", "util"):
        assert not features.filter(regex=f"^{g}__").isna().any().any()
    assert features.filter(regex="^lab__").isna().any().any()
    assert set(np.unique(features.filter(regex="^dx__").to_numpy())) <= {0, 1}
    assert "med__antineoplastic_chemotherapy" not in features.columns


def test_promis_baseline_matches_label_baseline(features, cohort):
    np.testing.assert_allclose(features["promis__gph_baseline"], cohort["gph_baseline"])
