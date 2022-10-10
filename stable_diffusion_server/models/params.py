from typing import Optional, Union

import pydantic
from pydantic import conlist, Field

from stable_diffusion_server.models.model import ModelId


class Token(pydantic.BaseModel):
    text: str
    alt_text: Optional[str] = None
    emphasis: int = 0
    percentage_divider: Optional[float] = None


class Params(pydantic.BaseModel):
    model_id: ModelId
    prompt: list[Token] = Field(..., min_items=1, max_items=75)
    negative_prompt: Optional[list[Token]] = Field(min_items=1, max_items=75, default=None)
    step_count: int = 20
    width: int = 512
    height: int = 512
