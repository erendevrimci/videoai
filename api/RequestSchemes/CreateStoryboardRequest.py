from pydantic import BaseModel
from typing import Optional


class CreateStoryboardRequest(BaseModel):
    project_id: int
    caption_id: int
    name: str
    shot_index_size: Optional[int] = None
