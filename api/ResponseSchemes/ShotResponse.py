from pydantic import BaseModel
from api.ResponseSchemes.StoryboardResponse import Shot

class ShotResponse(BaseModel):
    success: bool
    message: str




