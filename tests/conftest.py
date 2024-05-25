import pickle
from pathlib import Path

import pytest


@pytest.fixture
def covariance_data_path():
    return Path("tests/data/covariance.pickle")


@pytest.fixture
def gpy_data_path():
    return "tests/data/gpy_result.pickle"


@pytest.fixture
def load_data(data_path):
    if not data_path.exists():
        pytest.skip(f"Data file not found: {data_path}")
    with open(data_path, "rb") as f:
        return pickle.load(f)


def pytest_addoption(parser):
    parser.addoption(
        "--regenerate",
        action="store_true",
        default=False,
        help="Regenerate the reference data for the regression tests.",
    )
