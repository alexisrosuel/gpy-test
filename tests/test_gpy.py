import pickle

import numpy as np
from gpy_test.config.covariance import CovarianceConfig
from gpy_test.gpy import GPY

np.random.seed(42)


def test_gpy(gpy_data_path, request):
    fs = [lambda x: x, lambda x: x**2]
    y = np.random.randn(10, 4)
    covariance_config = CovarianceConfig(
        **{
            "integral_config": {
                "epsabs": 1e-2,
                "epsrel": 1e-2,
            },
            "contour_pair_config": {
                "contours": [
                    {"real_slack": 1, "type_": "circle"},
                    {"real_slack": 2, "type_": "circle"},
                ]
            },
            "admissible_imag": 1e-3,
            "admissible_negative": 1e2,
            "n_jobs": 1,
            "verbose": False,
        }
    )
    result = GPY(y, tuple(fs), covariance_config)

    if request.config.getoption("--regenerate"):
        with open(gpy_data_path, "wb") as f:
            pickle.dump(result, f)

    with open(gpy_data_path, "rb") as f:
        expected_output = pickle.load(f)

    assert (
        result == expected_output
    ), "Computed GPY result does not match the expected result"
