from pydantic import BaseModel
from typing import Optional
class CaptionRequest(BaseModel):
    project_id: int
    voice_over_id: int
    channel_number: Optional[int] = 1
    use_timeline: Optional[bool] = False

