import pydantic
from pydantic import BaseModel


class nyc(BaseModel):
    first_feat : float
    second_feat : float
    third_feat : float


