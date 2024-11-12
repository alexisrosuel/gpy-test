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


def _apply_functions(eigenvalues: f64_1d, fs: list[real_function]) -> f64_2d:
    results = []
    for f in fs:
        f_eigs = f(eigenvalues)
        results.append(f_eigs)
    return np.array(results)


def _LSSs(eigenvalues: f64_1d, fs: list[real_function]) -> f64_1d:
    f_eigs = _apply_functions(eigenvalues, fs)
    LSSs = np.sum(f_eigs, axis=1)
    assert len(LSSs) == len(fs)
    return LSSs


def _quadratic_form(x: f64_1d, A: f64_2d) -> float:
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

    # T is the number of time samples, M is the number of dimensions
    T, M = y.shape

    # GPY method of splits the time series in two along the dimension axis,
    # this allow to remove the recentering terms in the limiting distribution
    # under H0.
    half_M = floor(M / 2)
    sub_ys = [y[:, :half_M], y[:, half_M : 2 * half_M]]

    # compute all the eigenvalues of y @ y.T and y.T @ y (they differ only by
    # some 0s). Use the fastest way depending on the relative values of T and M
    if T >= half_M:
        svs = [np.linalg.svd(sub_y, full_matrices=False)[1] for sub_y in sub_ys]
        eigenvalues = [sv**2 for sv in svs]
        assert len(eigenvalues[0]) == half_M

        yty_eigenvalues = eigenvalues
        yyt_eigenvalues = [np.pad(eigs, (0, T - half_M)) for eigs in eigenvalues]
    else:
        svs = [np.linalg.svd(sub_y.T, full_matrices=False)[1] for sub_y in sub_ys]
        eigenvalues = [sv**2 for sv in svs]
        assert len(eigenvalues[0]) == T

        yty_eigenvalues = [np.pad(eigs, (0, half_M - T)) for eigs in eigenvalues]
        yyt_eigenvalues = eigenvalues

    assert len(yty_eigenvalues[0]) == half_M
    assert len(yyt_eigenvalues[0]) == T

    LSSs_diff = _LSSs(yyt_eigenvalues[0] / half_M, fs) - _LSSs(
        yyt_eigenvalues[1] / half_M, fs
    )
    Cov = compute_covariance(
        fs,
        XTX_eigenvalues=yty_eigenvalues[0] / half_M,
        c=T / half_M,
        covariance_config=covariance_config,
    )

    # since we take the difference of the same random variable distribution,
    # mean is zero and cov is scaled by 2
    mean = 0
    Cov *= 2

    # for complex gaussian data, the covariance matrix should be divided by 2
    complex_gaussian_factor = 1 / 2 if is_complex_gaussian else 1
    Cov *= complex_gaussian_factor

    # limiting distribution of the lss statistics is gaussian with zero mean and known covariance
    test_statistic = _quadratic_form(LSSs_diff - mean, np.linalg.inv(Cov))
    dof = len(fs)
    p_value = 1 - chi2.cdf(test_statistic, dof)

    return GPYResult(
        p_value,
        test_statistic,
        T,
        M,
        LSSs_diff,
        Cov,
    )
