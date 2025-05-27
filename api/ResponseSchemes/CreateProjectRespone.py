from pydantic import BaseModel


class CreateProject(BaseModel):
    id: int
    name: str
    user_id: str
    created_at: str

class CreateProjectResponse(BaseModel):
    success: bool
    message: str
    project: CreateProject
