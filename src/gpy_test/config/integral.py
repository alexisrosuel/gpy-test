from pydantic import BaseModel, PositiveFloat


class IntegralConfig(BaseModel):
    epsabs: PositiveFloat
    epsrel: PositiveFloat
