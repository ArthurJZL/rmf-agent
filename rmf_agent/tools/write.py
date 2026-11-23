"""Guarded write tools: dispatch_task and cancel_task.

Both default to ``dry_run=True``. A dry run validates the request and returns a plan plus a
single-use ``confirm_token``; committing requires the same arguments, ``dry_run=False`` and that
token. Policy checks (category allowlist, place validation, ownership for cancels, a cap on
active agent-created tasks) run on both phases.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from ..api import ApiError, RmfApi
from ..config import Settings
from .common import OPEN_STATUSES, TERMINAL_STATUSES, StrList
from .gate import ConfirmationGate
from .read import ReadTools
from .spec import ToolError

UNAVAILABLE = {"offline", "shutdown", "error", "uninitialized"}

COMMIT_HINT = (
    "Show this plan to the operator and ask for confirmation. Only after the operator confirms, "
    "call again with the same arguments, dry_run=false and this confirm_token."
)


class WriteTools:
    def __init__(
        self, api: RmfApi, settings: Settings, gate: ConfirmationGate, read: ReadTools
    ) -> None:
        self.api = api
        self.settings = settings
        self.gate = gate
        self.read = read

    def dispatch_task(
        self,
        category: Annotated[str, Field(description="Task category: patrol or delivery.")],
        places: Annotated[
            StrList | None, Field(description="patrol: places to visit in order.")
        ] = None,
        rounds: Annotated[int, Field(ge=1, le=5, description="patrol: number of rounds.")] = 1,
        pickup: Annotated[str | None, Field(description="delivery: pickup place.")] = None,
        dropoff: Annotated[str | None, Field(description="delivery: dropoff place.")] = None,
        fleet: Annotated[str | None, Field(description="Restrict to one fleet.")] = None,
        dry_run: Annotated[
            bool, Field(description="True validates only and returns a token.")
        ] = True,
        confirm_token: Annotated[
            str | None, Field(description="Token from the dry run; required to commit.")
        ] = None,
    ) -> dict[str, Any]:
        """WRITE. Dispatch a patrol or delivery task. Always run with dry_run=true first and show
        the plan to the operator; commit only after the operator explicitly confirms.
        """
        args = {
            "category": category,
            "places": places,
            "rounds": rounds,
            "pickup": pickup,
            "dropoff": dropoff,
            "fleet": fleet,
        }
        if category not in self.settings.allowed_categories:
            raise ToolError(
                "category_not_allowed",
                f"Category {category!r} is not allowlisted.",
                allowed=list(self.settings.allowed_categories),
            )
        request = self._request(args)
        problems = self._validate_dispatch(args)
        if problems:
            raise ToolError("validation_failed", "The request is not valid.", problems=problems)
        if dry_run:
            return {
                "dry_run": True,
                "plan": _dispatch_plan(args),
                "request": request,
                "confirm_token": self.gate.issue("dispatch_task", args),
                "expires_in_s": self.gate.ttl_s,
                "next_step": COMMIT_HINT,
            }
        self._redeem("dispatch_task", args, confirm_token)
        try:
            response = self.api.post(
                "/tasks/dispatch_task", {"type": "dispatch_task_request", "request": request}
            )
        except ApiError as exc:
            raise ToolError(
                "upstream_rejected", "The api-server rejected the task.", upstream=exc.body
            ) from exc
        state = response["state"]
        return {"committed": True, "task_id": state["booking"]["id"], "status": state["status"]}

    def cancel_task(
        self,
        task_id: Annotated[str, Field(description="Task id to cancel.")],
        reason: Annotated[str, Field(description="Short reason, recorded in the audit log.")],
        dry_run: Annotated[
            bool, Field(description="True validates only and returns a token.")
        ] = True,
        confirm_token: Annotated[
            str | None, Field(description="Token from the dry run; required to commit.")
        ] = None,
    ) -> dict[str, Any]:
        """WRITE. Cancel one task that rmf-agent dispatched. Always run with dry_run=true first and
        show the plan to the operator; commit only after the operator explicitly confirms.
        """
        args = {"task_id": task_id, "reason": reason}
        task = self.read.task_state(task_id)
        status = task.get("status")
        requester = task["booking"].get("requester")
        if status in TERMINAL_STATUSES:
            raise ToolError("already_terminal", f"Task {task_id} is already {status}.")
        if requester != self.settings.requester and not self.settings.cancel_any:
            raise ToolError(
                "not_owned",
                f"rmf-agent only cancels tasks it dispatched (requester={self.settings.requester});"
                f" {task_id} was requested by {requester}.",
            )
        if dry_run:
            robot = (task.get("assigned_to") or {}).get("name")
            return {
                "dry_run": True,
                "plan": f"Cancel {task_id} (status {status}"
                + (f", assigned to {robot})" if robot else ", not assigned)"),
                "confirm_token": self.gate.issue("cancel_task", args),
                "expires_in_s": self.gate.ttl_s,
                "next_step": COMMIT_HINT,
            }
        self._redeem("cancel_task", args, confirm_token)
        response = self.api.post(
            "/tasks/cancel_task",
            {
                "type": "cancel_task_request",
                "task_id": task_id,
                "labels": [f"requester={self.settings.requester}", f"reason={reason}"],
            },
        )
        if not response.get("success"):
            raise ToolError(
                "upstream_rejected", "The api-server refused the cancel.", upstream=response
            )
        return {"committed": True, "task_id": task_id, "status": "canceled"}

    # ---- helpers -----------------------------------------------------------------------

    def _redeem(self, tool: str, args: dict[str, Any], token: str | None) -> None:
        if not token:
            raise ToolError(
                "confirmation_required", "Committing needs the confirm_token from a dry run."
            )
        self.gate.redeem(tool, args, token)

    def _request(self, args: dict[str, Any]) -> dict[str, Any]:
        if args["category"] == "patrol":
            description = {"places": args["places"] or [], "rounds": args["rounds"]}
        else:
            description = {
                "pickup": {"place": args["pickup"], "handler": args["pickup"], "payload": []},
                "dropoff": {"place": args["dropoff"], "handler": args["dropoff"], "payload": []},
            }
        request = {
            "category": args["category"],
            "description": description,
            "requester": self.settings.requester,
            "labels": [f"requester={self.settings.requester}"],
        }
        if args["fleet"]:
            request["fleet_name"] = args["fleet"]
        return request

    def _validate_dispatch(self, args: dict[str, Any]) -> list[str]:
        problems = []
        known = {p["place"] for p in self.read.places()}
        if args["category"] == "patrol":
            wanted = args["places"] or []
            if not wanted:
                problems.append("patrol needs at least one place")
            if args["pickup"] or args["dropoff"]:
                problems.append("patrol does not take pickup/dropoff")
        else:
            wanted = [p for p in (args["pickup"], args["dropoff"]) if p]
            if len(wanted) != 2:
                problems.append("delivery needs both pickup and dropoff")
        unknown = [p for p in wanted if p not in known]
        if unknown:
            problems.append(f"unknown places: {unknown}; use find_places to list valid ones")

        robots = self.read.robots()
        if args["fleet"] and args["fleet"] not in {f for f, _ in robots}:
            problems.append(f"unknown fleet {args['fleet']!r}")
        available = [
            r
            for f, r in robots
            if (not args["fleet"] or f == args["fleet"]) and r.get("status") not in UNAVAILABLE
        ]
        if not available:
            problems.append("no available robot in the selected fleet(s)")

        active = self.api.get(
            "/tasks", {"requester": self.settings.requester, "status": ",".join(OPEN_STATUSES)}
        )
        if len(active) >= self.settings.max_active_agent_tasks:
            problems.append(
                f"rmf-agent already has {len(active)} open tasks "
                f"(limit {self.settings.max_active_agent_tasks})"
            )
        return problems


def _dispatch_plan(args: dict[str, Any]) -> str:
    where = f"fleet {args['fleet']}" if args["fleet"] else "any fleet"
    if args["category"] == "patrol":
        route = " -> ".join(args["places"])
        return f"Dispatch a patrol {route}, {args['rounds']} round(s), on {where}"
    return f"Dispatch a delivery from {args['pickup']} to {args['dropoff']}, on {where}"
