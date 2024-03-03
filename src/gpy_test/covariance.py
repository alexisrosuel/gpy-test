from pathlib import Path
from typing import Callable, Optional

import numpy as np
from joblib import Parallel, delayed
from ruamel.yaml import YAML
from scipy.integrate import dblquad
from tqdm import tqdm

from gpy_test.config.covariance import CovarianceConfig
from gpy_test.contour import Contour, create_contour
from gpy_test.types import f64_1d, f64_2d, real_function


def _integrand(
    c: float,
    f_pair: tuple[real_function, real_function],
    eigenvalues: tuple[f64_1d, f64_1d],
) -> Callable[[complex, complex], complex]:
    """Integrand of the covariance integral as defined in the paper.

    Note that the eigenvalues of both "half data matrices" are passed. z1 will be applued only to the
    first half, while z2 will be applied to the second half."""
    f1, f2 = f_pair
    CONSTANT = -(1 / (2 * np.pi**2))  # compute upfront the constant

    def integrd(z1: complex, z2: complex) -> complex:
        # the eigenvalues of (XTX-zI)^(-1) are simply the 1/(lambda_i -z)
        eigs_minus_z1 = 1 / (eigenvalues[0] - z1)
        eigs_minus_z2 = 1 / (eigenvalues[1] - z2)

        # compute the Stieljes transforms
        m_z1 = np.mean(eigs_minus_z1)
        m_z2 = np.mean(eigs_minus_z2)

        # compute the derivatives with respect to z
        dm_z1 = np.mean(eigs_minus_z1**2)
        dm_z2 = np.mean(eigs_minus_z2**2)

        # finally get the Stieltes transforms of the transpose matrices
        m_bar_z1 = -(1 - c) / z1 + c * m_z1
        m_bar_z2 = -(1 - c) / z2 + c * m_z2
        dm_bar_z1 = (1 - c) / z1**2 + c * dm_z1
        dm_bar_z2 = (1 - c) / z2**2 + c * dm_z2

        return (
            CONSTANT
            * 4  # I don't know why we need this factor. It is in the code provided by the author though
            * f1(z1)
            * f2(z2)
            * dm_bar_z1
            * dm_bar_z2
            / (m_bar_z1 - m_bar_z2) ** 2
        )

    return integrd


def _omega_ij(
    c: float,
    f_pair: tuple[real_function, real_function],
    XTX_eigenvalues: tuple[f64_1d, f64_1d],
    contour_pair: tuple[Contour, Contour],
    covariance_config: CovarianceConfig,
) -> complex:
    # define integrand parametrized by the contour
    integrand = _integrand(c, f_pair, XTX_eigenvalues)

    # define the double contour complex integrand
    contour_1, contour_2 = contour_pair
    integrand_reparametrized = (
        lambda t1, t2: integrand(contour_1.z(t1), contour_2.z(t2))
        * contour_1.dz(t1)
        * contour_2.dz(t2)
    )

    omega_ij_real = dblquad(
        lambda t1, t2: np.real(integrand_reparametrized(t1, t2)),
        contour_1.t_range[0],
        contour_1.t_range[1],
        lambda t: contour_2.t_range[0],
        lambda t: contour_2.t_range[1],
        (),
        covariance_config.integral_config.epsabs,
        covariance_config.integral_config.epsrel,
    )[0]
    # omega_ij_imag = dblquad(
    #    lambda t1, t2: np.imag(integrand_reparametrized(t1, t2)),
    #    contour_1.t_range[0],
    #    contour_1.t_range[1],
    #    lambda t: contour_2.t_range[0],
    #    lambda t: contour_2.t_range[1],
    #    (),
    #    covariance_config.integral_config.epsabs,
    #    covariance_config.integral_config.epsrel,
    # )[0]

    return omega_ij_real  # + 1j * omega_ij_imag


def _check_definite_non_negative(
    matrix: f64_2d, tolerance_imag: float, tolerance_negative: float
) -> None:
    """Check if a matrix is definite non-negative."""
    eigs = np.linalg.eigvals(matrix)

    # check all eigs are reals
    if np.any(np.abs(np.imag(eigs)) > tolerance_imag):
        msg = (
            "Convergence issue: the eigenvalues of the limit covariance matrix "
            f"are not real: {matrix=}, {eigs=}"
        )
        raise ValueError(msg)

    # check all eigs are positive
    eigs = np.real(eigs)  # keep only the real part
    if np.any(eigs < -tolerance_negative):
        msg = f"Convergence issue: the limit covariance matrix is not definite non-negative: {matrix=}, {eigs=}"
        raise ValueError(msg)


def _load_default_covariance_config() -> CovarianceConfig:
    yaml = YAML(typ="safe")
    current_path = Path(__file__).parent
    with open(current_path / "config" / "config.yaml") as f:
        config = yaml.load(f)
    return CovarianceConfig(**config)


def compute_covariance(
    fs: tuple[real_function, ...],
    XTX_eigenvalues: tuple[f64_1d, f64_1d] | f64_1d,
    c: float,
    covariance_config: Optional[CovarianceConfig] = None,
) -> f64_2d:
    # if covariance config is not defined, use the default one
    covariance_config = covariance_config or _load_default_covariance_config()

    # Get the contours. They must circle around the support of the distribution of
    # the eigvenvalues of XTX, with a good margin to ensure numerical stability
    # in the integral computation.
    eig_range = (np.min(XTX_eigenvalues), np.max(XTX_eigenvalues))
    contour_1 = create_contour(covariance_config.contour_config_pair[0], eig_range)
    contour_2 = create_contour(covariance_config.contour_config_pair[1], eig_range)

    tasks = []
    for i1, f1 in enumerate(fs):
        for i2, f2 in enumerate(fs):
            # covariance is symmetric, so we only need to compute the upper triangular part
            if i1 > i2:
                continue

            # the author suggests to split the eigenvalues per half covariance
            if isinstance(XTX_eigenvalues, tuple):
                if i1 == 0 and i2 == 0:
                    effective_eigs = (XTX_eigenvalues[0], XTX_eigenvalues[0])
                elif i1 == 1 and i2 == 1:
                    effective_eigs = (XTX_eigenvalues[1], XTX_eigenvalues[1])
                else:
                    effective_eigs = XTX_eigenvalues
            else:
                effective_eigs = (XTX_eigenvalues, XTX_eigenvalues)

            task = (
                i1,
                i2,
                c,
                (f1, f2),
                effective_eigs,
                (contour_1, contour_2),
                covariance_config,
            )
            tasks.append(task)

    if covariance_config.verbose:
        tasks = tqdm(tasks)

    # define the function to be computed in parallel, which returns also the indices of the pair of functions
    def _compute_omega_ij_with_indices(i1, i2, *args):
        cov = _omega_ij(*args)
        return cov, (i1, i2)

    results = Parallel(n_jobs=covariance_config.n_jobs)(
        delayed(_compute_omega_ij_with_indices)(*task) for task in tasks
    )

    # assemble the results into a matrix
    n_fs = len(fs)
    covariance = np.zeros((n_fs, n_fs), dtype=np.complex128)
    for omega_ij, (i1, i2) in results:
        covariance[i1, i2] = omega_ij
        covariance[i2, i1] = omega_ij

    _check_definite_non_negative(
        covariance,
        covariance_config.admissible_imag,
        covariance_config.admissible_negative,
    )

    return np.real(covariance)
