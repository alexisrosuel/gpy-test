from typing import Callable

import numpy as np

from gpy_test.config.contour import ContourConfig
from gpy_test.types import complex_1d, f64_1d


def _ellipse(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> tuple[Callable[[float], complex], Callable[[float], complex]]:
    """Complex a ellipse countour, with edge points at min_x + min_y i, max_x + max_y i, max_x - max_y i, min_x - min_y i.
    Oriented counter-clockwise.
    """
    x = lambda t: (min_x + max_x) / 2 + (max_x - min_x) / 2 * np.cos(t * 2 * np.pi)
    dx = lambda t: -(max_x - min_x) / 2 * np.sin(t * 2 * np.pi) * 2 * np.pi
    y = lambda t: (min_y + max_y) / 2 + (max_y - min_y) / 2 * np.sin(t * 2 * np.pi)
    dy = lambda t: (max_y - min_y) / 2 * np.cos(t * 2 * np.pi) * 2 * np.pi
    z = lambda t: x(t) + 1j * y(t)
    dz = lambda t: dx(t) + 1j * dy(t)
    return z, dz


def _rectangle(
    min_x: float, max_x: float, min_y: float, max_y: float
) -> tuple[Callable[[float], complex], Callable[[float], complex]]:
    """Complex a rectangle countour, with edge points at min_x + min_y i, min_x + max_y i, max_x + min_y i, min_x + min_y i.
    Oriented counter-clockwise.
    """
    total_length = 2 * (max_x - min_x + max_y - min_y)
    half_x_ratio = (max_x - min_x) / total_length
    half_y_ratio = (max_y - min_y) / total_length

    def x(t):
        if t <= half_x_ratio:
            return min_x + t * (max_x - min_x) / half_x_ratio
        elif t <= half_y_ratio + half_x_ratio:
            return max_x
        elif t <= 2 * half_x_ratio + half_y_ratio:
            return (
                max_x
                - (t - half_x_ratio - half_y_ratio) * (max_x - min_x) / half_x_ratio
            )
        else:
            return min_x

    def dx(t):
        if t <= half_x_ratio:
            return (max_x - min_x) / half_x_ratio
        elif t <= half_y_ratio + half_x_ratio:
            return 0
        elif t <= 2 * half_x_ratio + half_y_ratio:
            return -(max_x - min_x) / half_x_ratio
        else:
            return 0

    def y(t):
        if t <= half_x_ratio:
            return min_y
        elif t <= half_y_ratio + half_x_ratio:
            return min_y + (t - half_x_ratio) * (max_y - min_y) / half_y_ratio
        elif t <= 2 * half_x_ratio + half_y_ratio:
            return max_y
        else:
            return (
                max_y
                - (t - 2 * half_x_ratio - half_y_ratio) * (max_y - min_y) / half_y_ratio
            )

    def dy(t):
        if t <= half_x_ratio:
            return 0
        elif t <= half_y_ratio + half_x_ratio:
            return (max_y - min_y) / half_y_ratio
        elif t <= 2 * half_x_ratio + half_y_ratio:
            return 0
        else:
            return -(max_y - min_y) / half_y_ratio

    z = lambda t: x(t) + 1j * y(t)
    dz = lambda t: dx(t) + 1j * dy(t)
    return z, dz


def _circle(
    min_x: float, max_x: float
) -> tuple[Callable[[float], complex], Callable[[float], complex]]:
    """Complex a circle countour, with edge points at min_x + min_y i, min_x + max_y i, max_x + min_y i, min_x + min_y i.
    Oriented counter-clockwise.
    """
    z = lambda t: min_x + (max_x - min_x) * np.exp(1j * t * 2 * np.pi)
    dz = lambda t: 1j * (max_x - min_x) * np.exp(1j * t * 2 * np.pi) * 2 * np.pi
    return z, dz


def contour(
    contour_config: ContourConfig, eig_range: tuple[float, float]
) -> tuple[Callable[[float], complex], Callable[[float], complex]]:
    # determine the edge of the contour (make sure it is circling the limit eigenvalues distribution support)
    min_eig, max_eig = eig_range
    eig_diameter = max_eig - min_eig
    a = min_eig - contour_config.real_slack * eig_diameter
    b = max_eig + contour_config.real_slack * eig_diameter

    if contour_config.type_ == "ellipse":
        z, dz = _ellipse(a, b, -contour_config.imag_height, contour_config.imag_height)
    elif contour_config.type_ == "rectangle":
        z, dz = _rectangle(
            a, b, -contour_config.imag_height, contour_config.imag_height
        )
    elif contour_config.type_ == "circle":
        z, dz = _circle(a, b)
    else:
        raise ValueError(f"Unknown contour type: {contour_config.type_}")

    return z, dz
