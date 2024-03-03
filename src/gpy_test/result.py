from dataclasses import dataclass

import numpy as np

from gpy_test.types import f64_1d, f64_2d


@dataclass(frozen=True)
class GPYResult:
    p_value: float
    test_statistic: float
    N: int
    M: int
    lss: tuple[float, ...]
    mean: f64_1d
    covariance: f64_2d
    # diagnostics
    eigs_S_1: np.ndarray
    eigs_S_2: np.ndarray

    @property
    def c(self) -> float:
        return self.M / self.N
