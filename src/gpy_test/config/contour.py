from typing import Literal, Optional

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat, validator


class ContourConfig(BaseModel):
    """
    The contour is a rectangle, going around the eigenvalues of the sample covariance matrix. To make sure
    the full limit spectral distribution is captured, the contour is extended by a slack in the real. The height
    of the rectangle is given by the imag_height parameter.
    """

    real_slack: NonNegativeFloat  # slack to add around the estimated eigenvalues
    type_: Literal[
        "rectangle", "ellipse", "circle"
    ]  # issues of convergence with "rectangle" as it is not a differentiable contour.
    imag_height: Optional[PositiveFloat] = (
        None  # height of the contour in the imaginary axis, not defined in case of circle
    )

    # add a validator that checks that real_slack is None if type_ is circle
    @validator("imag_height")
    def check_imag_height(cls, v, values):
        if values["type_"] == "circle":
            if v is not None:
                raise ValueError("imag_height should be None if type_ is circle")
        return v
