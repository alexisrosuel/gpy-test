from math import floor
from typing import Optional

import numpy as np
from scipy.stats import chi2

from gpy_test.config.covariance import CovarianceConfig
from gpy_test.covariance import compute_covariance
from gpy_test.result import GPYResult
from gpy_test.types import complex_2d, f64_1d, f64_2d, real_function


def _validate(y: complex_2d | f64_2d, fs: tuple[real_function, ...]) -> None:
    assert len(y.shape) == 2, "y must be a 2D array"
    assert len(fs) > 0, "fs must be a non-empty list of real functions"


def _apply_functions_to_eigenvalues(
    eigenvalues: f64_1d, fs: list[real_function]
) -> f64_2d:
    results = []
    for f in fs:
        f_eigs = f(eigenvalues)
        results.append(f_eigs)
    return np.array(results)


def _LSSs(y: f64_2d, fs: list[real_function]) -> f64_1d:
    T, M = y.shape
    half_covariance = y @ y.conj().T / M
    eigenvalues = np.linalg.eigvalsh(half_covariance)
    f_eigs = _apply_functions_to_eigenvalues(eigenvalues, fs)
    LSSs = np.sum(f_eigs, axis=1)
    assert len(LSSs) == len(fs)
    return LSSs


def quadratic_form(x: f64_1d, A: f64_2d) -> float:
    return ((x.T @ A) @ x).item()


def GPY(
    y: complex_2d | f64_2d,
    fs: tuple[real_function, ...],
    covariance_config: Optional[CovarianceConfig] = None,
    is_complex_gaussian: bool = False,
    recenter: bool = False,
) -> GPYResult:
    """
    Compute the GPY statistic for a given time series y. The time series dimension order is assumed to
    be (time, dimension). The GPY statistic is a test for the null hypothesis that the time series is
    dimensions are independent.
    """
    # start by validating the input
    _validate(y, fs)

    if recenter:
        y -= np.mean(y, axis=0)

    # N is the number of time samples, M is the number of dimensions
    T, M = y.shape

    # GPY method of splitting the time series in two along the dimension axis
    # this allow to remove the recentering terms
    half_M = floor(M / 2)
    sub_ys = [y[:, :half_M], y[:, half_M : 2 * half_M]]
    LSSs_diff = _LSSs(sub_ys[0], fs) - _LSSs(sub_ys[1], fs)

    # Limit covariance of the LSSs
    Cov = compute_covariance(
        fs,
        # XTX_eigenvalues = np.linalg.eigvalsh(y @ y.conj().T / M),
        XTX_eigenvalues=(
            np.linalg.eigvalsh(sub_ys[0] @ sub_ys[0].conj().T * 2 / M),
            np.linalg.eigvalsh(sub_ys[1] @ sub_ys[1].conj().T * 2 / M),
        ),
        c=T / M,
        covariance_config=covariance_config,
    )

    # since we take the difference of the same random variable distribution,
    # mean is zero
    # cov is scaled by 2
    mean = 0
    Cov *= 2

    # for complex gaussian data, the covariance matrix should be divided by 2
    complex_gaussian_factor = 1 / 2 if is_complex_gaussian else 1
    Cov *= complex_gaussian_factor

    # limiting distribution of the lss statistics is gaussian with zero mean and known covariance
    test_statistic = quadratic_form(LSSs_diff - mean, np.linalg.inv(Cov))
    dof = len(LSSs_diff)
    p_value = 1 - chi2.cdf(test_statistic, dof)

    return GPYResult(
        p_value,
        test_statistic,
        T,
        M,
        LSSs_diff,
        Cov,
    )
