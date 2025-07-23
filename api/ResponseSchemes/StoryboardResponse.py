from pydantic import BaseModel
from typing import Optional
from typing import List



class StoryboardResponse(BaseModel):
    success: bool
    message: str
    task_id: Optional[str] = None

