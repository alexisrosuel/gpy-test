from math import floor
from typing import Optional

import numpy as np
from scipy.stats import chi2

from gpy_test.config.covariance import CovarianceConfig
from gpy_test.covariance import covariance
from gpy_test.result import GPYResult
from gpy_test.types import complex_2d, f64_1d, f64_2d, real_function


def _validate(
    y: complex_2d,
    fs: tuple[real_function, ...],
    Cov: Optional[f64_2d],
    covariance_config: Optional[CovarianceConfig],
    sd: Optional[real_function | tuple[f64_1d, f64_1d]],
) -> None:
    assert len(y.shape) == 2, "y must be a 2D array"
    assert len(fs) > 0, "fs must be a non-empty list of real functions"
    assert Cov is not None or (
        covariance_config is not None and sd is not None
    ), "Either covariance or (covariance_config, sd) must be provided"
    if covariance_config is not None and sd is not None:
        if isinstance(tuple, sd):
            frequency_grid, sd_values = sd
            assert len(frequency_grid) == len(
                sd_values
            ), "frequency_grid and sd_values must have the same length"


def GPY(
    y: complex_2d,
    fs: tuple[real_function, ...],
    Cov: Optional[f64_2d] = None,
    covariance_config: Optional[CovarianceConfig] = None,
    sd: Optional[real_function | tuple[f64_1d, f64_1d]] = None,
) -> GPYResult:
    """
    Compute the GPY statistic for a given time series y. The time series dimension order is assumed to
    be (time, dimension). The GPY statistic is a test for the null hypothesis that the time series is
    dimensions are independent.

    Possible to call this function with either:
    - a known covariance matrix covariance
    - a pair of covariance_config and sd, which will be used to estimate the covariance matrix covariance

    Warning: the computation of the covariance matrix covariance is the most time-consuming part of the test.
    """
    # start by validating the input
    _validate(y, fs, Cov, covariance_config, sd)

    # follow GPY method of splitting the time series in two along the dimension axis (not the time one!)
    N, M = y.shape
    half_M = floor(M / 2)
    y_1, y_2 = y[:, :half_M], y[:, half_M:2*half_M]
    S_1 = y_1.conj() @ y_1.T / half_M
    S_2 = y_2.conj() @ y_2.T / (M-half_M)
    eigs_1 = np.linalg.eigvalsh(S_1)
    eigs_2 = np.linalg.eigvalsh(S_2)

    # comnpute the lss 
    lsss = [] 
    for f in fs:
        f1 = np.array([f(eig) for eig in eigs_1])
        f2 = np.array([f(eig) for eig in eigs_2])
        lss = N * np.mean(f2 - f1)
        lsss.append(lss)
    lsss = np.array(lsss)

    # limiting distribution of the lss statistics is gaussian with known mean and covariance
    mean = np.zeros(len(fs))
    if Cov is None:
        Cov = covariance(
            covariance_config,
            fs,
            sd,
            (min(eigs_1), max(eigs_1)),
            c=N / M,
        )

    test_statistic = (lsss - mean).reshape(1, -1) @ np.linalg.inv(Cov) @ (lsss - mean)
    test_statistic = test_statistic[0]
    p_value = 1 - chi2.cdf(test_statistic, len(lsss))
    return GPYResult(p_value, test_statistic, N, M, lsss, mean, Cov, eigs_1, eigs_2)
