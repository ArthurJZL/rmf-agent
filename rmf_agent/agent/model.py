"""Model clients. The agent loop talks to a small ``ModelClient`` protocol; ``CohereModel`` is the
real implementation on the Cohere Python SDK v2 (``cohere.ClientV2.chat`` with ``tools``).

Messages are Cohere v2 chat messages as plain dicts (system / user / assistant with
``tool_plan`` + ``tool_calls`` / tool with ``tool_call_id``), which the SDK accepts directly.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurn:
    """One model response, normalized."""

    text: str = ""
    tool_plan: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "COMPLETE"
    input_tokens: int | None = None
    output_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelTurn:
        calls = [ToolCall(**c) for c in data.get("tool_calls", [])]
        return cls(**{**data, "tool_calls": calls})

    def as_message(self) -> Message:
        """The assistant message to append to the conversation."""
        if self.tool_calls:
            return {
                "role": "assistant",
                "tool_plan": self.tool_plan or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in self.tool_calls
                ],
            }
        return {"role": "assistant", "content": self.text}


class ModelClient(Protocol):
    name: str
    is_fake: bool

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
        timeout_s: float | None = None,
    ) -> ModelTurn: ...


class CohereModel:
    """Cohere chat via ``cohere.ClientV2``. Reads the API key from ``CO_API_KEY``."""

    is_fake = False

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any = None,
    ) -> None:
        import cohere

        key = api_key or os.environ.get("CO_API_KEY")
        if client is None and not key:
            raise RuntimeError("CO_API_KEY is not set. Use --offline to run without a key.")
        self.name = model
        self.temperature = temperature
        self.client = client or cohere.ClientV2(api_key=key, client_name="rmf-agent")

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
        timeout_s: float | None = None,
    ) -> ModelTurn:
        kwargs: dict[str, Any] = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
        if timeout_s:
            kwargs["request_options"] = {"timeout_in_seconds": max(1, int(timeout_s))}
        response = self.client.chat(**kwargs)
        message = response.message
        text = "".join(
            item.text for item in (message.content or []) if getattr(item, "type", "") == "text"
        )
        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments or "{}"),
            )
            for call in (message.tool_calls or [])
        ]
        tokens = response.usage.tokens if response.usage else None
        return ModelTurn(
            text=text,
            tool_plan=message.tool_plan,
            tool_calls=calls,
            finish_reason=str(response.finish_reason),
            input_tokens=_int(tokens.input_tokens) if tokens else None,
            output_tokens=_int(tokens.output_tokens) if tokens else None,
        )


def _int(value: float | None) -> int | None:
    return int(value) if value is not None else None
