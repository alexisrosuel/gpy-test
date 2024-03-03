from typing import Callable

import numpy as np
from scipy.integrate import dblquad


def complex_dblquad(
    func: Callable[[float, float], complex],
    a: float,
    b: float,
    gfun: Callable[[float], float],
    hfun: Callable[[float], float],
    args: tuple,
    epsabs: float,
    epsrel: float,
) -> complex:
    return (
        dblquad(
            lambda x, y: np.real(func(x, y)),
            a,
            b,
            gfun,
            hfun,
            args,
            epsabs,
            epsrel,
        )[0]
        + 1j
        * dblquad(
            lambda x, y: np.imag(func(x, y)),
            a,
            b,
            gfun,
            hfun,
            args,
            epsabs,
            epsrel,
        )[0]
    )
