from pydantic import BaseModel


class HealthResponse(BaseModel):

    status: str


class RootResponse(BaseModel):

    service: str

    status: str

    version: str