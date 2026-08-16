from typing import Literal, Optional
from pydantic import BaseModel, Field


class Detection(BaseModel):
    label: Literal['person', 'smoke', 'fire']
    box: list[int] = Field(description="[x1, y1, x2, y2] absolute pixels, 640x480, origin top-left")
    confidence: float = Field(ge=0.0, le=1.0)
    description: Optional[str] = Field(
        default=None,
        description="Visual evidence for this call: features, colour/texture, surrounding context",
    )


class DetectionResult(BaseModel):
    detections: list[Detection]
