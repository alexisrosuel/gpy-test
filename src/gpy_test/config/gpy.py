from pydantic import BaseModel, Field, PositiveFloat, validator

from gpy_test.config.contour import ContourConfig
from gpy_test.config.covariance import CovarianceConfig
from gpy_test.config.fixed_point import FixedPointConfig


class GPYConfig(BaseModel):
    covariance_config: CovarianceConfig
    spectral_density_estimate_config: SpectralDensityEstimateConfig

    # validate that ct1 is strictly inside ct2 (or the other way around)
    @validator("contour_config_pair")
    def validate_contour_config_pair(cls, v):
        ct1, ct2 = v
        is_ct1_inside_ct2 = (
            ct1.imag_height < ct2.imag_height and ct1.real_slack < ct2.real_slack
        )
        is_ct2_inside_ct1 = (
            ct2.imag_height < ct1.imag_height and ct2.real_slack < ct1.real_slack
        )
        if is_ct1_inside_ct2 or is_ct2_inside_ct1:
            return v
        else:
            raise ValueError("One contour must be strictly inside the other")
