from pydantic import BaseModel
from typing import Optional


class VideoEditRequest(BaseModel):
    project_id: int
    caption_id: int
    use_timeline: Optional[bool] = False

