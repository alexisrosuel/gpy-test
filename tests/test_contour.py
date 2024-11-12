import numpy as np
import pytest
from gpy_test.contour import Contour


@pytest.mark.parametrize(
    "center, radius, t, expected_z, expected_dz",
    [
        (0, 1, 0, 1 + 0j, 0 - 2j * np.pi),  # t = 0
        (0, 1, 0.5, -1 + 0j, 0 + 2j * np.pi),  # t = 0.5
        (1, 1, 0, 2 + 0j, 0 - 2j * np.pi),  # t = 0
        (1, 2, 0.25, 1 - 2j, -4 * np.pi + 0j),  # t = 0.25
    ],
)
def test_contour_from_circle_parameters(center, radius, t, expected_z, expected_dz):
    contour = Contour.from_circle_parameters(center, radius)

    assert contour.t_range == (0, 1)
    assert contour.z(t) == pytest.approx(expected_z, rel=1e-9)
    assert contour.dz(t) == pytest.approx(expected_dz, rel=1e-9)
