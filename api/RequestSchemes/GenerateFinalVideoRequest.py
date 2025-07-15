from pydantic import BaseModel

class GenerateFinalVideoRequest(BaseModel):
    project_id: int
    caption_id: int 