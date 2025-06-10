from pydantic import BaseModel
from typing import Optional

class ImageResponse(BaseModel):
    id: str
    url: str

class UploadImageResponse(BaseModel):
    success: bool
    message: str
    image_response: Optional[ImageResponse] = None
