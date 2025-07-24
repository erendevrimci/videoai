from pydantic import BaseModel


class ShotResponse(BaseModel):
    success: bool
    message: str




