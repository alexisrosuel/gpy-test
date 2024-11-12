from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.stats import chi2

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
    def df(self) -> int:
        return len(self.LSSs_diff)

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

    def is_positive(
        self, level: float, alternative: Literal["left", "right", "double"] = "double"
    ) -> bool:
        if alternative == "left":
            critical_value = chi2.ppf(level, self.df)
            return self.test_statistic < critical_value
        elif alternative == "right":
            critical_value = chi2.ppf(1 - level, self.df)
            return self.test_statistic > critical_value
        elif alternative == "double":
            lower = chi2.ppf(level / 2, self.df)
            upper = chi2.ppf(1 - level / 2, self.df)
            return (self.test_statistic < lower) | (self.test_statistic > upper)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
