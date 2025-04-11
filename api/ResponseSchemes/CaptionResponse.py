from pydantic import BaseModel

class CaptionResponse(BaseModel):
    success: bool
    message: str


