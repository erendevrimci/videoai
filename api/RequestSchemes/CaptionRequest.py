from pydantic import BaseModel
from typing import Optional
class CaptionRequest(BaseModel):
    voice_over_id: int
    channel_number: int
    use_timeline: Optional[bool] = False

