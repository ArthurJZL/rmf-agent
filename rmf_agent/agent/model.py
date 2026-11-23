"""Model clients. The agent loop talks to a small ``ModelClient`` protocol; ``CohereModel`` is the
real implementation on the Cohere Python SDK v2 (``cohere.ClientV2.chat`` with ``tools``).

Messages are Cohere v2 chat messages as plain dicts (system / user / assistant with
``tool_plan`` + ``tool_calls`` / tool with ``tool_call_id``), which the SDK accepts directly.
"""

from __future__ import annotations

import json
import os
import threading
import time
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


def sanitize_messages(messages: list[Message]) -> list[Message]:
    """Drop ``tool_plan`` from replayed assistant messages.

    Newer Command models (e.g. ``command-a-plus-05-2026``) reject a ``tool_plan`` field in the
    request history with HTTP 400 ("tool plan cannot be used with this model"). The plan is
    still kept in the trace; it just isn't sent back to the API.
    """
    clean: list[Message] = []
    for m in messages:
        if m.get("role") == "assistant" and "tool_plan" in m:
            m = {k: v for k, v in m.items() if k != "tool_plan"}
        clean.append(m)
    return clean


class _RateLimiter:
    """Process-wide minimum spacing between API calls (trial keys allow 20 calls/min)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self, min_interval_s: float) -> None:
        if min_interval_s <= 0:
            return
        with self._lock:
            delay = self._last + min_interval_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


_LIMITER = _RateLimiter()
CALL_COUNT = {"calls": 0, "retries": 0}


def _is_rate_limit(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    return status == 429 or type(exc).__name__ in {"TooManyRequestsError", "RateLimitError"}


class CohereModel:
    """Cohere chat via ``cohere.ClientV2``. Reads the API key from ``CO_API_KEY``.

    Calls are spaced by ``RMF_AGENT_MIN_INTERVAL_S`` seconds (default 3.5, i.e. under 18
    calls/min) and HTTP 429s are retried with exponential backoff (up to ``max_retries``).
    """

    is_fake = False

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.0,
        client: Any = None,
        min_interval_s: float | None = None,
        max_retries: int = 4,
    ) -> None:
        import cohere

        key = api_key or os.environ.get("CO_API_KEY")
        if client is None and not key:
            raise RuntimeError("CO_API_KEY is not set. Use --offline to run without a key.")
        self.name = model
        self.temperature = temperature
        self.client = client or cohere.ClientV2(api_key=key, client_name="rmf-agent")
        if min_interval_s is None:
            min_interval_s = float(os.environ.get("RMF_AGENT_MIN_INTERVAL_S", "3.5"))
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries

    def _call(self, kwargs: dict[str, Any]) -> Any:
        backoff = 8.0
        for attempt in range(self.max_retries + 1):
            _LIMITER.wait(self.min_interval_s)
            CALL_COUNT["calls"] += 1
            try:
                return self.client.chat(**kwargs)
            except Exception as exc:
                if not _is_rate_limit(exc) or attempt == self.max_retries:
                    raise
                CALL_COUNT["retries"] += 1
                time.sleep(backoff)
                backoff *= 2
        raise RuntimeError("unreachable")

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
            "messages": sanitize_messages(messages),
            "temperature": self.temperature,
        }
        if tool_choice and tool_choice.upper() == "NONE":
            # Newer Command models reject ``tool_choice`` ("not supported for this model").
            # A text-only call is the same thing: send no tools, unless the history already
            # holds tool calls (the API needs the tool definitions to accept those).
            if any(m.get("role") == "tool" for m in messages):
                kwargs["tools"] = tools
        else:
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if timeout_s:
            kwargs["request_options"] = {"timeout_in_seconds": max(1, int(timeout_s))}
        response = self._call(kwargs)
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
