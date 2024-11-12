import pytest
from gpy_test.config.contour import ContourConfig, ContourPairConfig


@pytest.mark.parametrize(
    "real_slack, type_, imag_height, is_valid",
    [
        (0.0, "circle", None, True),
        (1.5, "circle", None, True),
        (0.0, "circle", 1.0, False),
        (0.0, "circle", -1.0, False),
        (1.0, "circle", None, True),
        (0.0, "circle", 0.5, False),
        (2.0, "circle", None, True),
    ],
)
def test_contour_config(real_slack, type_, imag_height, is_valid):
    if not is_valid:
        with pytest.raises(ValueError):
            ContourConfig(real_slack=real_slack, type_=type_, imag_height=imag_height)
    else:
        config = ContourConfig(
            real_slack=real_slack, type_=type_, imag_height=imag_height
        )
        assert config.real_slack == real_slack
        assert config.type_ == type_
        assert config.imag_height == imag_height


@pytest.mark.parametrize(
    "contour_1, contour_2, is_valid",
    [
        (
            {"real_slack": 1, "type_": "circle"},
            {"real_slack": 2, "type_": "circle"},
            True,
        ),
        (
            {"real_slack": 2, "type_": "circle"},
            {"real_slack": 1, "type_": "circle"},
            True,
        ),
        (
            {"real_slack": 1, "type_": "circle"},
            {"real_slack": 1, "type_": "circle"},
            False,
        ),
    ],
)
def test_contour_pair_validator(contour_1, contour_2, is_valid):
    d = {"contours": (contour_1, contour_2)}

    if is_valid:
        ContourPairConfig(**d)
    else:
        with pytest.raises(ValueError):
            ContourPairConfig(**d)
