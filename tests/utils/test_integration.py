import numpy as np
import pytest
from scipy.integrate import dblquad

from gpy_test.utils.integration import dblsimpson


@pytest.mark.parametrize(
    "f", [lambda x, y: x**2 + y**4, lambda x, y: np.exp(-1j * (x**2 + y**2))]
)
def test_dblsimpson(f):
    x_range = np.linspace(0, 2, 100)
    y_range = np.linspace(0, 1, 50)
    func_values = np.array([[f(x, y) for y in y_range] for x in x_range])

    dblsimpson_value = dblsimpson(func_values, x_range, y_range)

    # check the value estimated via the simpson rule against the value estimated via dblquad
    dblquad_real_value = dblquad(
        lambda x, y: np.real(f(x, y)), 0, 1, lambda x: 0, lambda x: 2
    )  # warning: dlquad computes the integral of f(y, x) instead of f(x, y) as in dblsimpson!!
    dblquad_imag_value = dblquad(
        lambda x, y: np.imag(f(x, y)), 0, 1, lambda x: 0, lambda x: 2
    )

    assert np.isclose(
        dblsimpson_value, dblquad_real_value[0] + 1j * dblquad_imag_value[0]
    )
