from pydantic import BaseModel
from typing import List

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    sample_rate: int
    supported_languages: List[str]