from typing import Literal, Optional

from pydantic import BaseModel, PositiveFloat, PositiveInt, validator


class IntegralConfig(BaseModel):
    type_: Literal["dblquad", "dblsimpson"]
    n_points: Optional[PositiveInt] = None
    epsabs: Optional[PositiveFloat] = None
    epsrel: Optional[PositiveFloat] = None

    # validate that n_points is provided only if type_ is dblsimpson
    @validator("n_points")
    def validate_n_points_integration(cls, v, values):
        if values["type_"] == "dblsimpson" and v is None:
            raise ValueError("n_points must be provided if type_ is dblsimpson")
        return v

    # validate that epsabs and epsrel are provided only if type_ is dblquad
    @validator("epsabs", "epsrel")
    def validate_eps(cls, v, values):
        if values["type_"] == "dblquad" and v is None:
            raise ValueError("epsabs and epsrel must be provided if type_ is dblquad")
        return v
