import pytest

from gpy_test.gpy import _A, _m


@pytest.mark.parametrize(
    "m, sd, c, expected",
    [
        (1 + 1j, lambda x: 1, 1, 0.4000000000000001 - 0.20000000000000004j),
    ],
)
def test__A(m, sd, c, expected):
    assert _A(m, sd, c) == expected


@pytest.mark.parametrize(
    "z, sd, c, expected",
    [(1 + 1j, lambda x: 1, 1, -0.10692431068530435 + 0.6360098231439529j)],
)
def test__m(z, sd, c, expected):
    assert _m(z, sd, c) == expected


# @pytest.mark.parametrize(
#     "lrv_config, expected_lrv_result",
#     [
#         (
#             GPYConfig(L=2, f=lambda x: (x - 1) ** 2, n_points_density=3),
#             None,
#         )
#     ],
# )
# def test_GPY(lrv_config, expected_lrv_result):
#     y = np.arange(10 * 2).reshape(10, 2)
#     assert GPY(y, lrv_config) == expected_lrv_result
