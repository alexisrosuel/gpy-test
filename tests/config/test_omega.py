import pytest

from gpy_test.config.covariance import CovarianceConfig


@pytest.mark.parametrize(
    "contour_1, contour_2, is_valid",
    [
        (
            {"imag_height": 1, "real_slack": 1},
            {"imag_height": 2, "real_slack": 2},
            True,
        ),
        (
            {"imag_height": 2, "real_slack": 2},
            {"imag_height": 1, "real_slack": 1},
            True,
        ),
        (
            {"imag_height": 1, "real_slack": 2},
            {"imag_height": 2, "real_slack": 1},
            False,
        ),
        (
            {"imag_height": 2, "real_slack": 1},
            {"imag_height": 1, "real_slack": 2},
            False,
        ),
    ],
)
def test_contour_pair_validator(test_config, contour_1, contour_2, is_valid):
    d = test_config.to_dict()

    # update contour_1 and contour_2 in d
    d["contour_config_pair"][0]["imag_height"] = contour_1["imag_height"]
    d["contour_config_pair"][0]["real_slack"] = contour_1["real_slack"]
    d["contour_config_pair"][1]["imag_height"] = contour_2["imag_height"]
    d["contour_config_pair"][1]["real_slack"] = contour_2["real_slack"]

    if is_valid:
        CovarianceConfig(**d)
    else:
        with pytest.raises(ValueError):
            CovarianceConfig(**d)
