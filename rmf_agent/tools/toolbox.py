"""The toolbox: tool registry, argument validation, error shaping, timing and auditing."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ..api import ApiError, HttpApi, InProcessApi, RmfApi
from ..config import Settings
from ..sim.world import FakeRmfServer
from .audit import AuditLog
from .common import DATA_NOTICE
from .gate import ConfirmationGate
from .read import ReadTools
from .spec import ToolError, ToolSpec
from .write import WriteTools


@dataclass
class ToolOutcome:
    name: str
    args: dict[str, Any]
    ok: bool
    payload: dict[str, Any]
    latency_ms: float
    error: str | None = None
    write: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def as_text(self) -> str:
        return json.dumps(self.payload, separators=(",", ":"), ensure_ascii=False)


class Toolbox:
    def __init__(
        self,
        api: RmfApi,
        settings: Settings,
        audit: AuditLog | None = None,
        gate: ConfirmationGate | None = None,
    ) -> None:
        self.api = api
        self.settings = settings
        self.audit = audit or AuditLog(settings.audit_path)
        self.gate = gate or ConfirmationGate(ttl_s=settings.confirm_ttl_s)
        read = ReadTools(api)
        write = WriteTools(api, settings, self.gate, read)
        specs = [
            ToolSpec.from_function(read.fleet_overview),
            ToolSpec.from_function(read.find_stuck_robots, free_text=True),
            ToolSpec.from_function(read.explain_robot, free_text=True),
            ToolSpec.from_function(read.task_status, free_text=True),
            ToolSpec.from_function(read.list_tasks),
            ToolSpec.from_function(read.late_tasks),
            ToolSpec.from_function(read.door_and_lift_status),
            ToolSpec.from_function(read.recent_alerts, free_text=True),
            ToolSpec.from_function(read.task_log, free_text=True),
            ToolSpec.from_function(read.find_places),
            ToolSpec.from_function(write.dispatch_task, write=True),
            ToolSpec.from_function(write.cancel_task, write=True),
        ]
        self.specs = {s.name: s for s in specs}

    def available(self) -> list[ToolSpec]:
        """Tools offered to a model. Write tools are hidden unless writes are enabled."""
        return [s for s in self.specs.values() if self.settings.allow_writes or not s.write]

    def call(self, name: str, args: dict[str, Any] | None) -> ToolOutcome:
        args = dict(args or {})
        start = time.perf_counter()
        spec = self.specs.get(name)
        write = bool(spec and spec.write)
        try:
            if spec is None:
                raise ToolError(
                    "unknown_tool",
                    f"No tool named {name!r}.",
                    available=[s.name for s in self.available()],
                )
            if spec.write and not self.settings.allow_writes:
                raise ToolError(
                    "writes_disabled",
                    "Write tools are disabled. Start with --allow-writes to enable.",
                )
            try:
                validated = spec.args_model.model_validate(args).model_dump()
            except ValidationError as exc:
                raise ToolError(
                    "invalid_arguments",
                    "Arguments failed validation.",
                    problems=[f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()],
                ) from exc
            payload = {"ok": True, **spec.fn(**validated)}
            if spec.free_text:
                payload["data_notice"] = DATA_NOTICE
            outcome = ToolOutcome(name, args, True, payload, 0.0, write=write)
        except ToolError as exc:
            outcome = ToolOutcome(
                name, args, False, {"ok": False, **exc.payload()}, 0.0, error=exc.code, write=write
            )
        except ApiError as exc:
            outcome = ToolOutcome(
                name,
                args,
                False,
                {"ok": False, "error": "upstream_error", "status": exc.status, "path": exc.path},
                0.0,
                error="upstream_error",
                write=write,
            )
        outcome.latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self._audit(outcome)
        return outcome

    def _audit(self, outcome: ToolOutcome) -> None:
        payload = outcome.payload
        if not outcome.ok:
            decision = f"rejected:{outcome.error}"
        elif payload.get("committed"):
            decision = "committed"
        elif payload.get("dry_run"):
            decision = "dry_run_ok"
        else:
            decision = "ok"
        self.audit.write(
            tool=outcome.name,
            args=outcome.args,
            write=outcome.write,
            dry_run=outcome.args.get("dry_run", True) if outcome.write else None,
            decision=decision,
            task_id=payload.get("task_id") if outcome.write else None,
            latency_ms=outcome.latency_ms,
        )


def build_toolbox(
    settings: Settings,
    *,
    server: FakeRmfServer | None = None,
    audit: AuditLog | None = None,
    gate: ConfirmationGate | None = None,
) -> Toolbox:
    """Toolbox against ``settings.api_url`` over HTTP, or the in-process fake scenario."""
    api: RmfApi
    if settings.api_url:
        api = HttpApi(settings.api_url, settings.api_token)
    else:
        api = InProcessApi(server or FakeRmfServer.from_scenario(settings.scenario))
    return Toolbox(api, settings, audit=audit, gate=gate)
