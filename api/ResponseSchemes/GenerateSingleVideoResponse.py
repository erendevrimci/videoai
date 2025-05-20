from pydantic import BaseModel
from typing import Optional

class GenerateSingleVideoResponse(BaseModel):
    success: bool
    message: str
    video: Optional[str] = None
