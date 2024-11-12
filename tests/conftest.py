from pathlib import Path

import pytest


@pytest.fixture
def covariance_data_path():
    return Path("tests/data/covariance.pickle")


@pytest.fixture
def gpy_data_path():
    return "tests/data/gpy_result.pickle"


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate",
        action="store_true",
        default=False,
        help="Regenerate the reference data for the regression tests.",
    )
