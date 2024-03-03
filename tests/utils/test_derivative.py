import numpy as np
import pytest

from gpy_test.utils.derivative import derivative


@pytest.mark.parametrize(
    "f, true_f",
    [
        (lambda x: x**2, lambda x: 2 * x),
        (lambda t: 1 + 1 * np.exp(1j * t), lambda t: 1j * np.exp(1j * t)),
        (lambda z: z**3, lambda z: 3 * z**2),
    ],
)
@pytest.mark.parametrize("x", [1, 2, 3, 1 + 1j])
def test_derivative(f, true_f, x):
    assert abs(derivative(f)(x) - true_f(x)) < 1e-6
