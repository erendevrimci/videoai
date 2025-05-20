from pydantic import BaseModel
from typing import Optional



class Storyboard(BaseModel):
    id: int
    project_id: int
    name: str
    story_board: Optional[list[dict]] = None
    initial_images_created: Optional[bool] = False
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class StoryboardResponse(BaseModel):
    success: bool
    message: str
    storyboards: list[Storyboard]

