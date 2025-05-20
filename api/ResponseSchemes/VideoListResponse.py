from pydantic import BaseModel
from typing import List, Optional

class VideoURLItem(BaseModel):
    created_at: Optional[str] = None
    url: Optional[str] = None
    id: Optional[int] = None
    class Config:
        orm_mode = True

class VideoListResponse(BaseModel):
    success: bool
    message: str
    videos: Optional[List[VideoURLItem]] = None 