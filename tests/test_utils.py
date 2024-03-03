import numpy as np
import pytest

from gpy_test.utils import complex_derivative, derivative


@pytest.mark.parametrize(
    "f, x, expected",
    [
        (lambda x: x**2, 1, 2),
        (lambda x: x**2, 2, 4),
        (lambda t: 1 + 1 * np.exp(1j * t), 0, 1j),
        (lambda t: 1 + 1 * np.exp(1j * t), np.pi / 2, -1),
    ],
)
def test_derivative(f, x, expected):
    precision = 1e-8
    assert abs(derivative(f)(x) - expected) < precision


@pytest.mark.parametrize(
    "f, true_f",
    [
        (lambda z: z**2, lambda z: 2 * z),
        (lambda z: z**3, lambda z: 3 * z**2),
    ],
)
@pytest.mark.parametrize("z", [1 + 1j, 2 + 1j, 3 + 2j])
def test_complex_derivative(f, true_f, z):
    assert abs(complex_derivative(f)(z) - true_f(z)) < 1e-6
