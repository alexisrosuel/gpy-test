import numpy as np
from pydantic import BaseModel, PositiveFloat, PositiveInt

f64_1d = np.ndarray
f64_2d = np.ndarray
complex_1d = np.ndarray


class FixedPointConfig(BaseModel):
    init_m_real: float
    init_m_imag: float
    max_steps: PositiveInt
    tolerance: PositiveFloat

    @property
    def init_m(self) -> complex:
        return complex(self.init_m_real, self.init_m_imag)
