from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from scipy.integrate import quad, simpson
from scipy.optimize import fixed_point
from tqdm import tqdm

from gpy_test.config.covariance import CovarianceConfig
from gpy_test.config.fixed_point import FixedPointConfig
from gpy_test.contour import contour
from gpy_test.integration import complex_dblquad, dblsimpson
from gpy_test.types import f64_1d, f64_2d, real_function
from gpy_test.utils import derivative


def _A(
    sd: real_function | tuple[f64_1d, f64_1d], c: float, z: complex, tolerance: float
) -> complex:
    # possible to provide either the spectral density as a callable, or as a pair of frequencies and values
    if isinstance(sd, tuple):
        freqs, values = sd
        return simpson(
            y=1 / (c * z + 1 / values),
            x=freqs,
        )
    else:
        return quad(
            lambda x: 1 / (c * z + 1 / sd(x)),
            0,
            1,
            epsrel=tolerance,
            complex_func=True,
            limit=50,
        )[0]


def _m(
    z: complex,
    sd: real_function | tuple[f64_1d, f64_1d],
    c: float,
    fixed_point_config: FixedPointConfig,
) -> float:
    """Fixed point iteration to solve the m equation."""
    # set tolerance one order of magnitude less than the tolerance in the fixed point iteration
    func = lambda m: 1 / (-z + _A(sd, c, m, fixed_point_config.tolerance / 10))
    return fixed_point(
        func,
        fixed_point_config.init_m,
        args=(),
        xtol=fixed_point_config.tolerance,
        maxiter=fixed_point_config.max_steps,
        method="del2",
    )


def _m_bar(
    z: complex,
    sd: real_function | tuple[f64_1d, f64_1d],
    c: float,
    fixed_point: FixedPointConfig,
) -> float:
    return -(1 - c) / z + c * _m(z, sd, c, fixed_point)


def _integrand(
    f1: real_function,
    f2: real_function,
    sd: real_function | tuple[f64_1d, f64_1d],
    c: float,
    derivate_epsilon: float,
    fixed_point_config: FixedPointConfig,
) -> Callable[[complex, complex], complex]:
    """Integrand of the covariance integral as defined in the paper."""
    m_bar = lambda z: _m_bar(z, sd, c, fixed_point_config)

    def dm_bar(z: complex) -> complex:
        return derivative(m_bar, derivate_epsilon)(z)

    def integrd(z1: complex, z2: complex) -> complex:
        return (
            -(1 / np.pi**2)
            * f1(z1)
            * f2(z2)
            * dm_bar(z1)
            * dm_bar(z2)
            / (m_bar(z1) - m_bar(z2)) ** 2
        )

    return integrd


def _omega_ij(
    covariance_config: CovarianceConfig,
    f1: real_function,
    f2: real_function,
    i1: int,
    i2: int,
    sd: real_function | tuple[f64_1d, f64_1d],
    eig_range: tuple[float, float],
    c: float,
) -> tuple[float, tuple[real_function, real_function]]:

    # define integrand parametrized by the contour
    integrand = _integrand(
        f1,
        f2,
        sd,
        c,
        covariance_config.derivative_epsilon,
        covariance_config.fixed_point_config,
    )
    z1, dz1 = contour(covariance_config.contour_config_pair[0], eig_range)
    z2, dz2 = contour(covariance_config.contour_config_pair[1], eig_range)
    integrand_reparametrized = (
        lambda t1, t2: integrand(z1(t1), z2(t2)) * dz1(t1) * dz2(t2)
    )

    # compute the integral
    integral_config = covariance_config.integral_config
    if integral_config.type_ == "dblquad":
        omega_ij = complex_dblquad(
            integrand_reparametrized,
            0,
            1,
            lambda t: 0,
            lambda t: 1,
            (),
            integral_config.epsabs,
            integral_config.epsrel,
        )
    elif integral_config.type_ == "dblsimpson":
        t_range = np.linspace(0, 1, integral_config.n_points)
        values = np.array(
            [[integrand_reparametrized(t1, t2) for t1 in t_range] for t2 in t_range]
        )
        omega_ij = dblsimpson(values, t_range, t_range)

    # check that the noise in the imaginary part of covariance is small
    if np.imag(omega_ij) > covariance_config.admissible_imag:
        msg = f"Convergence issue: one entry of the limit covariance matrix is not real: {omega_ij=}"
        raise ValueError(msg)

    # return also the indices of the pair of functions (required for parallel computation of covariance entries)
    return np.real(omega_ij), (i1, i2)


def _is_definite_non_negative(matrix: f64_2d) -> bool:
    """Check if a matrix is definite non-negative."""
    return np.all(np.linalg.eigvals(matrix) >= 0)


def covariance(
    covariance_config: CovarianceConfig,
    fs: tuple[real_function, ...],
    sd: real_function | tuple[f64_1d, f64_1d],
    eigs_range: tuple[float, float],
    c: float,
) -> f64_2d:
    tasks = []
    for i1, f1 in enumerate(fs):
        for i2, f2 in enumerate(fs):
            # covariance is symmetric, so we only need to compute the upper triangular part
            if i1 > i2:
                continue

            task = (f1, f2, i1, i2)
            tasks.append(task)

    # compute in parallel the entries of covariance
    if covariance_config.verbose:
        tasks = tqdm(tasks)

    results = Parallel(n_jobs=covariance_config.n_jobs)(
        delayed(_omega_ij)(covariance_config, f1, f2, i1, i2, sd, eigs_range, c)
        for f1, f2, i1, i2 in tasks
    )

    # assemble the results into a matrix
    n_fs = len(fs)
    covariance = np.zeros((n_fs, n_fs), dtype=np.float64)
    for omega_ij, (i1, i2) in results:
        covariance[i1, i2] = omega_ij
        covariance[i2, i1] = omega_ij

    # check that the matrix is definite non-negative
    if not _is_definite_non_negative(covariance):
        msg = f"Convergence issue: the limit covariance matrix is not definite non-negative: {covariance=}"
        raise ValueError(msg)

    return covariance
