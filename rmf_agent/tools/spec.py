"""Tool specifications built from typed Python functions.

One function is the single source of truth for a tool: its docstring becomes the description,
its annotated signature becomes a pydantic model (argument validation) and a JSON schema (sent to
Cohere, and derived again by FastMCP for the MCP server).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, create_model


class ToolError(Exception):
    """A tool-level failure the model should see and can act on (not found, rejected, ...)."""

    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details

    def payload(self) -> dict[str, Any]:
        return {"error": self.code, "message": self.message, **self.details}


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., dict[str, Any]]
    args_model: type[BaseModel]
    write: bool = False
    free_text: bool = False  # output contains strings that came from the fleet system

    @classmethod
    def from_function(
        cls, fn: Callable[..., dict[str, Any]], *, write: bool = False, free_text: bool = False
    ) -> ToolSpec:
        hints = typing.get_type_hints(fn, include_extras=True)
        fields: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param.name] = (hints[param.name], default)
        model = create_model(f"{fn.__name__}_args", __config__=ConfigDict(extra="forbid"), **fields)
        return cls(
            name=fn.__name__,
            description=inspect.getdoc(fn) or fn.__name__,
            fn=fn,
            args_model=model,
            write=write,
            free_text=free_text,
        )

    def parameters_schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return schema

    def cohere_tool(self) -> dict[str, Any]:
        """The tool in Cohere v2 ``tools=[...]`` format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema(),
            },
        }
