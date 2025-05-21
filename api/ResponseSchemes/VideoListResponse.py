from pydantic import BaseModel, ConfigDict
from typing import List, Optional

class VideoURLItem(BaseModel):
    created_at: Optional[str] = None
    url: Optional[str] = None
    id: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)

class VideoListResponse(BaseModel):
    success: bool
    message: str
    videos: Optional[List[VideoURLItem]] = None 