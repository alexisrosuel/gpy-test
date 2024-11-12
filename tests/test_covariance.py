import pickle

import numpy as np
import pytest
from gpy_test.config.covariance import CovarianceConfig
from gpy_test.covariance import (
    _check_real_definite_non_negative,
    _integrand,
    compute_covariance,
)


@pytest.mark.parametrize(
    "c, f_pair, eigenvalues, z1, z2, expected",
    [
        (
            0.5,
            (lambda x: x, lambda x: x**2),
            [0.5, 1, 1.5, 2],
            10 + 0j,
            9 + 1j,
            4.556622800115634 - 20.264128681041736j,
        )
    ],
)
def test__integrand(c, f_pair, eigenvalues, z1, z2, expected):
    integrand_function = _integrand(c, f_pair, np.array(eigenvalues))
    assert integrand_function(z1, z2) == expected


@pytest.mark.parametrize(
    "matrix, tolerance_negative, tolerance_imag, expected_exception",
    [
        # pytest.param(
        #     np.array([[2, 0], [0, 3]]),
        #     1e-9,
        #     1e-9,
        #     None,
        #     id="valid case: positive definite matrix",
        # ),
        # pytest.param(
        #     np.array([[1, 1], [1, 1]]),
        #     1e-9,
        #     1e-9,
        #     None,
        #     id="valid case: eigenvalues are 2 and 0 (positive and real)",
        # ),
        pytest.param(
            np.array([[1, 0], [0, -1]]),
            1e-9,
            1e-9,
            ValueError,
            id="invalid case: negative eigenvalue",
        ),
        pytest.param(
            np.array([[0, 1], [1, 0]]),
            1e-9,
            1e-9,
            ValueError,
            id="invalid case: zero eigenvalue",
        ),
        pytest.param(
            np.array([[1, 1e-10], [1e-10, 1]]),
            1e-9,
            1e-9,
            None,
            id="valid case: very small off-diagonal values",
        ),
        pytest.param(
            np.array([[1, 0], [0, 1]]),
            1e-9,
            None,
            None,
            id="valid case: tolerance_imag is None, check only for negative eigenvalues",
        ),
        pytest.param(
            np.array([[1, 0], [0, 1]]),
            None,
            1e-9,
            None,
            id="valid case: tolerance_negative is None, check only for real eigenvalues",
        ),
        pytest.param(
            np.array([[1j, 0], [0, 1j]]),
            None,
            1e-9,
            ValueError,
            id="invalid case: purely imaginary eigenvalues",
        ),
    ],
)
def test_check_real_definite_non_negative(
    matrix, tolerance_negative, tolerance_imag, expected_exception
):
    if expected_exception:
        with pytest.raises(expected_exception):
            _check_real_definite_non_negative(
                matrix, tolerance_negative, tolerance_imag
            )
    else:
        _check_real_definite_non_negative(matrix, tolerance_negative, tolerance_imag)


def test_compute_covariance(covariance_data_path, request):
    fs = [lambda x: x, lambda x: x**2]
    XTX_eigenvalues = np.linspace(0, 2, 10)
    c = 0.5
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
    result = compute_covariance(fs, XTX_eigenvalues, c, covariance_config)

    if request.config.getoption("--regenerate"):
        with open(covariance_data_path, "wb") as f:
            pickle.dump(result, f)

    with open(covariance_data_path, "rb") as f:
        expected_output = pickle.load(f)

    assert np.allclose(
        result, expected_output, rtol=1e-5, atol=1e-8
    ), "Computed covariance does not match the expected result"
