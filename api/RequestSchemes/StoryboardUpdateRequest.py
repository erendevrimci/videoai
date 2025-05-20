from pydantic import BaseModel
from typing import Optional

class StoryboardUpdateRequest(BaseModel):
    name: Optional[str] = None
    story_board: Optional[list[dict]] = None
    initial_images_created: Optional[bool] = None

