import pytest

from gpy_test import config


@pytest.fixture
def test_config():
    return config()
