from pydantic import BaseModel
from typing import Optional

class GenerateSingleVideoRequest(BaseModel):
    model: str
    prompt: str
    negativePrompt: Optional[str] = None
    duration: int
    cfgScale: Optional[float] = None
    aspectRatio: str
    startImage: Optional[str] = None
    endImage: Optional[str] = None
    startImageId: Optional[str] = None
    endImageId: Optional[str] = None