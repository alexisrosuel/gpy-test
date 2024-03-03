from dataclasses import dataclass

from gpy_test.types import f64_1d, f64_2d


@dataclass(frozen=True)
class GPYResult:
    p_value: float
    test_statistic: float
    N: int
    M: int
    LSSs_diff: f64_1d
    # limit statistics
    covariance: f64_2d
    # diagnostics
    # half_covariances: tuple[f64_2d, f64_2d]
    # eigenvalues_half_covariance: tuple[f64_1d, f64_1d]
    # LSSs_halfs: tuple[f64_2d, f64_2d]

    @property
    def c(self) -> float:
        return self.M / self.N
