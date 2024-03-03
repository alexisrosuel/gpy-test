import numpy as np
from scipy.linalg import sqrtm
from statsmodels.tsa.arima_process import arma_generate_sample

from gpy_test.types import complex_2d, f64_1d, f64_2d


def white_arma_sample(
    N: int, M: int, burn: int, ar: float, ma: float, is_complex_gaussian: bool = False
) -> f64_2d | complex_2d:
    """Generate a sample of M independent ARMA(ar, ma) processes, of length N."""
    real = arma_generate_sample(
        ar=[1, -ar],
        ma=[1, ma],
        nsample=(N, M),
        scale=1 / np.sqrt(2),
        burnin=burn,
        axis=0,
    )
    if is_complex_gaussian:
        imag = arma_generate_sample(
            ar=[1, -ar],
            ma=[1, ma],
            nsample=(N, M),
            scale=1 / np.sqrt(2),
            burnin=burn,
            axis=0,
        )
        y = real + 1j * imag
    else:
        y = real * np.sqrt(2)

    return y


def arma_covariance(M: int, ar: float, ma: float) -> f64_2d:
    """Generate the covariance matrix of an ARMA(ar, ma) process with M
    dimensions."""
    C = np.zeros((M, M))
    for k in range(M):
        for h in range(M):
            if k == h:
                C[k, h] = 1 + (ar + ma) ** 2 / (1 - ar**2)
            elif abs(k - h) == 1:
                C[k, h] = ar + ma + (ar + ma) ** 2 * ar / (1 - ar**2)
            else:
                C[k, h] = ar ** (abs(k - h) - 1) * (
                    ar + ma + (ar + ma) ** 2 * ar / (1 - ar**2)
                )
    return C


def non_linear_sample(N: int, M: int, burn: int) -> f64_2d:
    Z = np.random.normal(0, 1, (N + burn, M + 2))
    return Z[:, :-2] * Z[:, 1:-1] * (1 + Z[:, 2:] + Z[:, :-2])


def factor_model_sample(
    N: int, M: int, burn: int, K: int, time_ar: float, time_ma: float
) -> f64_2d:
    epsilon = np.random.normal(0, 1, (N, M))
    f = arma_generate_sample(
        [1, -time_ar], [1, time_ma], (K, N), scale=0.05, burnin=burn, axis=0
    )
    lambdas = np.random.normal(4, 1, (1, K))
    return (lambdas @ f).T + epsilon


def arch_sample(N: int, M: int, burn: int, alpha_0: float, alpha_1: float) -> f64_2d:
    Z = np.random.normal(0, 1, (N + burn, M + 1))
    X = np.zeros((N + burn, M + 1))
    X[:, 0] = Z[:, 0]
    for m in range(1, M + 1):
        X[:, m] = Z[:, m] * np.sqrt(alpha_0 + alpha_1 * X[:, m - 1] ** 2)
    return X


def rank_one(
    N: int, M: int, burn: int, time_ar: float, time_ma: float, u: f64_1d
) -> f64_2d:
    Z = white_arma_sample(N, M, burn, time_ar, time_ma)
    T = np.identity(M) + u[:, np.newaxis] @ u[np.nexaxis, :]
    T_half = sqrtm(T)
    return Z @ T
