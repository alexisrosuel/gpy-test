from typing import Optional

from gpy_test.config.contour import ContourPairConfig
from gpy_test.config.integral import IntegralConfig
from pydantic import BaseModel, PositiveFloat


class CovarianceConfig(BaseModel):
    integral_config: IntegralConfig
    contour_pair_config: ContourPairConfig
    admissible_negative: PositiveFloat
    admissible_imag: Optional[PositiveFloat] = None  # if None, no check
    n_jobs: int = 1
    verbose: bool = False
