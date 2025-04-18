from pydantic import BaseModel

class VideoEditResponse(BaseModel):
    success: bool
    message: str


