from typing import Literal

from pydantic import BaseModel, NonNegativeFloat, PositiveFloat


class ContourConfig(BaseModel):
    """
    The contour is a rectangle, going around the eigenvalues of the sample covariance matrix. To make sure
    the full limit spectral distribution is captured, the contour is extended by a slack in the real. The height
    of the rectangle is given by the imag_height parameter.
    """

    imag_height: PositiveFloat  # height of the contour in the imaginary axis
    real_slack: NonNegativeFloat  # slack to add around the estimated eigenvalues
    type_: Literal["rectangle", "ellipse", "circle"]
