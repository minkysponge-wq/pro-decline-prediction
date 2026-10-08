import dataclasses

import pytest

from prodecline.config import CONFIG
from prodecline.data.cohort import build_cohort
from prodecline.data.simulate import EHRSimulator
from prodecline.features.build import build_features

SMALL = dataclasses.replace(CONFIG.cohort, n_patients=300, seed=7)


@pytest.fixture(scope="session")
def ehr():
    return EHRSimulator(SMALL).generate()


@pytest.fixture(scope="session")
def cohort(ehr):
    return build_cohort(ehr, SMALL, CONFIG.labels)[0]


@pytest.fixture(scope="session")
def features(ehr, cohort):
    return build_features(ehr, cohort, CONFIG.features)
