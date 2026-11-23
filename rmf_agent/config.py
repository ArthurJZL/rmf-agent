"""Runtime settings, read from environment variables with explicit overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Any

DEFAULT_MODEL = "command-a-plus-05-2026"
"""Default Cohere chat model. Override with ``RMF_AGENT_MODEL`` or ``--model``."""

DEFAULT_SCENARIO = "nominal"


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # Where the api-server is. ``None`` means the in-process fake, loaded from ``scenario``.
    api_url: str | None = None
    api_token: str | None = field(default=None, repr=False)
    scenario: str = DEFAULT_SCENARIO
    # Write safety. Writes are off unless explicitly enabled.
    allow_writes: bool = False
    allowed_categories: tuple[str, ...] = ("patrol", "delivery")
    cancel_any: bool = False
    max_active_agent_tasks: int = 3
    confirm_ttl_s: float = 120.0
    requester: str = "rmf-agent"
    audit_path: str | None = "rmf-agent-audit.jsonl"
    # Model.
    model: str = DEFAULT_MODEL

    @classmethod
    def from_env(cls, **overrides: Any) -> Settings:
        env = os.environ
        categories = env.get("RMF_AGENT_ALLOWED_CATEGORIES")
        base = cls(
            api_url=env.get("RMF_API_URL") or None,
            api_token=env.get("RMF_API_TOKEN") or None,
            scenario=env.get("RMF_AGENT_SCENARIO", DEFAULT_SCENARIO),
            allow_writes=_bool(env.get("RMF_AGENT_ALLOW_WRITES"), False),
            allowed_categories=(
                tuple(c.strip() for c in categories.split(",") if c.strip())
                if categories
                else cls.allowed_categories
            ),
            cancel_any=_bool(env.get("RMF_AGENT_CANCEL_ANY"), False),
            audit_path=env.get("RMF_AGENT_AUDIT_PATH", cls.audit_path),
            model=env.get("RMF_AGENT_MODEL", DEFAULT_MODEL),
        )
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})
