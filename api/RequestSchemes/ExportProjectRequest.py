from pydantic import BaseModel


class ExportProjectRequest(BaseModel):
    project_id: int