from pydantic import BaseModel
from typing import Optional
class RefreshSignedUrlResponse(BaseModel):
    success: bool
    message: str
    signed_url: Optional[str] = None
