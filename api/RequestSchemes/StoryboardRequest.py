from pydantic import BaseModel
from typing import Optional
class StoryboardRequest(BaseModel):
    name: str
    project_id: int
    story_board: list[dict]
    initial_images_created: Optional[bool] = False
    
