from dataclasses import dataclass

import numpy as np

from gpy_test.types import f64_1d, f64_2d


@dataclass(frozen=True)
class GPYResult:
    p_value: float
    test_statistic: float
    N: int
    M: int
    LSSs_diff: f64_1d
    covariance: f64_2d

    @property
    def c(self) -> float:
        return self.M / self.N

    def __eq__(self, other):
        if not isinstance(other, GPYResult):
            return NotImplemented
        return (
            np.isclose(self.p_value, other.p_value, rtol=1e-5, atol=1e-8)
            and np.isclose(
                self.test_statistic, other.test_statistic, rtol=1e-5, atol=1e-8
            )
            and self.N == other.N
            and self.M == other.M
            and np.allclose(self.LSSs_diff, other.LSSs_diff, rtol=1e-5, atol=1e-8)
            and np.allclose(self.covariance, other.covariance, rtol=1e-5, atol=1e-8)
        )
