from pydantic import BaseModel, Field
from typing import List


class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=1000)
    valence: float = Field(..., description="Valence value (0.0 to 1.0)", ge=0.0, le=1.0)
    arousal: float = Field(..., description="Arousal value (0.0 to 1.0)", ge=0.0, le=1.0)
    language: str = Field(default="en", description="Language code")
    temperature: float = Field(default=0.75, ge=0.1, le=2.0, description="Sampling temperature")
    length_penalty: float = Field(default=1.0, ge=0.1, le=3.0, description="Length penalty")
    repetition_penalty: float = Field(default=2.0, ge=1.0, le=5.0, description="Repetition penalty")
    top_p: float = Field(default=0.8, ge=0.1, le=1.0, description="Top-p sampling")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_available: bool
    sample_rate: int
    supported_languages: List[str]