from pydantic import BaseModel

class ShotRequest(BaseModel):
    is_approved: bool