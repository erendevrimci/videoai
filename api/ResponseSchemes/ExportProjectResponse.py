from pydantic import BaseModel
from typing import Optional

class ExportProjectResponse(BaseModel):
    success: bool
    message: str
    export_url: Optional[str] = None