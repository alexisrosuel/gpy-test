from typing import Callable

import numpy as np
from scipy.integrate import dblquad, simpson

from gpy_test.types import complex_2d, f64_1d


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


def dblsimpson(func_values: complex_2d, x_values: f64_1d, y_values: f64_1d) -> complex:
    return simpson(y=simpson(y=func_values, x=x_values, axis=0), x=y_values, axis=0)
