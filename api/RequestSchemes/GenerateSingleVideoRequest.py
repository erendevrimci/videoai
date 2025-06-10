from pydantic import BaseModel
from typing import Optional

class GenerateSingleVideoRequest(BaseModel):
    model: str
    prompt: str
    negativePrompt: Optional[str] = None
    duration: int
    cfgScale: Optional[float] = None
    aspectRatio: str
    startImage: str
    endImage: Optional[str] = None
    startImageId: str
    endImageId: Optional[str] = None