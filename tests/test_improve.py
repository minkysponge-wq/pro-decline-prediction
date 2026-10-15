"""Tests for the accuracy extension (burden features + tuned cut-off)."""
import numpy as np
import pandas as pd

from prodecline.config import CONFIG
from prodecline.features.burden import ABNORMAL_LAB_RULES, build_burden_features
from prodecline.improve import best_accuracy_threshold
from tests.test_leakage import _drop_post_index


def test_burden_features_ignore_post_index_events(ehr, cohort, features):
    a = build_burden_features(ehr, cohort, features, CONFIG.features)
    clean = _drop_post_index(ehr, cohort)
    b = build_burden_features(clean, cohort, features, CONFIG.features)
    pd.testing.assert_frame_equal(a, b)


def test_burden_features_shape_and_ranges(ehr, cohort, features):
    b = build_burden_features(ehr, cohort, features, CONFIG.features)
    assert b.index.equals(features.index)
    assert all(c.startswith("burden__") for c in b.columns)
    assert not b.isna().any().any() and (b >= 0).all().all()
    assert b["burden__abnormal_labs"].between(0, len(ABNORMAL_LAB_RULES)).all()
    assert b["burden__n_dx"].max() > 0


def test_best_accuracy_threshold():
    y = np.array([0] * 80 + [1] * 20)
    p = np.r_[np.linspace(0.0, 0.3, 80), np.linspace(0.35, 0.6, 20)]   # separable at ~0.32
    t = best_accuracy_threshold(y, p)
    assert 0.3 < t <= 0.35
    assert ((p >= t) == y).all()
