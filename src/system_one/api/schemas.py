"""OpenAI-compatible request schemas for the gateway."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MessageRole = Literal["assistant", "developer", "system", "tool", "user"]


class ChatMessage(BaseModel):
    """A text chat message accepted by the initial gateway contract."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    """The supported non-routing chat completion request."""

    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False
    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    n: int | None = Field(default=None, gt=0)
    user: str | None = None
