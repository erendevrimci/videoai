from pydantic import BaseModel

class GenerateFinalVideoFromStoryboardRequest(BaseModel):
    storyboard_id: int 