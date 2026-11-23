"""Rule-based robot diagnosis. Rules produce findings with evidence; the model explains them.

"Stuck" is an inference, so every finding carries the data it was derived from. Rules:

==================  ========  ===============================================================
code                severity  condition
==================  ========  ===============================================================
STATUS_ERROR        high      robot status is ``error``
OFFLINE             high      robot status is ``offline`` or ``shutdown``
STALE_STATE         high      no state update for more than ``stale_after_s``
WAITING_ON_DOOR     high      latest task log line names a door, and that door is not open
WAITING_ON_LIFT     high      latest task log line names a lift, and the lift is out of service
TASK_BLOCKED        high      current task status is ``blocked`` or ``error``
NO_PROGRESS         high      working, but no task log entry for more than ``no_progress_min``
HAS_ISSUES          warning   the robot reports issues (high when the robot is in error)
LOW_BATTERY_ON_TASK warning   battery below ``low_battery`` while working on a task
LOW_BATTERY         warning   battery below ``low_battery`` and not charging
TASK_DELAYED        warning   current task status is ``delayed``
==================  ========  ===============================================================

A robot is reported as stuck when it has at least one ``high`` finding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import (
    DOOR_MODE_NAMES,
    LIFT_IN_SERVICE_MODES,
    LIFT_MODE_NAMES,
    minutes,
    public,
    referenced_doors,
    referenced_lifts,
    ros_time_ms,
)


@dataclass(frozen=True)
class Thresholds:
    stale_after_s: float = 60.0
    no_progress_min: float = 10.0
    low_battery: float = 0.20


def finding(code: str, severity: str, **evidence: Any) -> dict[str, Any]:
    return {"code": code, "severity": severity, "evidence": evidence}


def diagnose(
    robot: dict[str, Any],
    now_ms: int,
    task: dict[str, Any] | None,
    log: list[dict[str, Any]],
    doors: dict[str, dict[str, Any]],
    lifts: dict[str, dict[str, Any]],
    limits: Thresholds,
) -> list[dict[str, Any]]:
    """Findings for one robot. ``log`` is the flattened log of its current task, oldest first."""
    out: list[dict[str, Any]] = []
    status = robot.get("status")
    battery = robot.get("battery")
    age_s = (now_ms - robot["unix_millis_time"]) / 1000

    if status == "error":
        out.append(finding("STATUS_ERROR", "high", status=status))
    if status in {"offline", "shutdown"}:
        out.append(finding("OFFLINE", "high", status=status))
    if age_s > limits.stale_after_s:
        out.append(finding("STALE_STATE", "high", last_update_age_s=round(age_s)))
    if robot.get("issues"):
        severity = "high" if status == "error" else "warning"
        out.append(finding("HAS_ISSUES", severity, issues=robot["issues"]))
    if battery is not None and battery < limits.low_battery:
        pct = round(battery * 100)
        if status == "working" and robot.get("task_id"):
            out.append(
                finding("LOW_BATTERY_ON_TASK", "warning", battery_pct=pct, task_id=robot["task_id"])
            )
        elif status != "charging":
            out.append(finding("LOW_BATTERY", "warning", battery_pct=pct))

    if task:
        task_status = task.get("status")
        if task_status in {"blocked", "error"}:
            out.append(
                finding(
                    "TASK_BLOCKED", "high", task_id=task["booking"]["id"], task_status=task_status
                )
            )
        elif task_status == "delayed":
            out.append(finding("TASK_DELAYED", "warning", task_id=task["booking"]["id"]))

    if status == "working" and log:
        last = log[-1]
        waited_min = minutes(now_ms - last["_ms"])
        out.extend(_door_findings(last, waited_min, doors, now_ms))
        out.extend(_lift_findings(last, waited_min, lifts))
        waiting = any(f["code"] in {"WAITING_ON_DOOR", "WAITING_ON_LIFT"} for f in out)
        if not waiting and waited_min > limits.no_progress_min:
            out.append(
                finding(
                    "NO_PROGRESS", "high", minutes_since_last_log=waited_min, last_log=public(last)
                )
            )
    return out


def _door_findings(
    last: dict[str, Any], waited_min: float, doors: dict[str, dict[str, Any]], now_ms: int
) -> list[dict[str, Any]]:
    out = []
    for name in referenced_doors(last["text"]):
        door = doors.get(name)
        if door is None:
            continue
        mode = DOOR_MODE_NAMES.get(door["current_mode"]["value"], "unknown")
        if mode != "open":
            out.append(
                finding(
                    "WAITING_ON_DOOR",
                    "high",
                    door=name,
                    door_mode=mode,
                    door_state_age_min=minutes(now_ms - ros_time_ms(door["door_time"])),
                    waiting_min=waited_min,
                    last_log=public(last),
                )
            )
    return out


def _lift_findings(
    last: dict[str, Any], waited_min: float, lifts: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    out = []
    for name in referenced_lifts(last["text"]):
        lift = lifts.get(name)
        if lift is None:
            continue
        mode = LIFT_MODE_NAMES.get(lift["current_mode"], "unknown")
        if mode not in LIFT_IN_SERVICE_MODES:
            out.append(
                finding(
                    "WAITING_ON_LIFT",
                    "high",
                    lift=name,
                    lift_mode=mode,
                    lift_floor=lift["current_floor"],
                    waiting_min=waited_min,
                    last_log=public(last),
                )
            )
    return out


def is_stuck(findings: list[dict[str, Any]]) -> bool:
    return any(f["severity"] == "high" for f in findings)
