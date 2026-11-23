"""Read-only, task-shaped tools. Each answers one operator question with compact JSON + evidence."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Field

from ..api import ApiError, RmfApi
from .common import (
    DOOR_MODE_NAMES,
    LIFT_DOOR_NAMES,
    LIFT_IN_SERVICE_MODES,
    LIFT_MODE_NAMES,
    LIFT_MOTION_NAMES,
    OPEN_STATUSES,
    TERMINAL_STATUSES,
    StrList,
    destinations,
    flatten_log,
    iso,
    minutes,
    public,
    referenced_doors,
    referenced_lifts,
    ros_time_ms,
)
from .diagnose import Thresholds, diagnose, is_stuck
from .spec import ToolError

LOG_WINDOW_MS = 3_600_000  # diagnosis looks at the last hour of a task's log


class ReadTools:
    def __init__(self, api: RmfApi, limits: Thresholds | None = None) -> None:
        self.api = api
        self.limits = limits or Thresholds()

    # ---- shared lookups ----------------------------------------------------------------

    def now(self) -> int:
        return int(self.api.get("/time"))

    def robots(self) -> list[tuple[str, dict[str, Any]]]:
        return [
            (fleet["name"], robot)
            for fleet in self.api.get("/fleets")
            for robot in (fleet.get("robots") or {}).values()
        ]

    def find_robot(self, name: str) -> tuple[str, dict[str, Any]]:
        robots = self.robots()
        for fleet, robot in robots:
            if robot["name"] == name:
                return fleet, robot
        raise ToolError(
            "not_found", f"No robot named {name!r}.", known_robots=[r["name"] for _, r in robots]
        )

    def task_state(self, task_id: str) -> dict[str, Any]:
        try:
            return self.api.get(f"/tasks/{task_id}/state")
        except ApiError as exc:
            if exc.status == 404:
                raise ToolError("not_found", f"No task with id {task_id!r}.") from exc
            raise

    def task_log_entries(self, task_id: str, window_ms: int = LOG_WINDOW_MS) -> list[dict]:
        try:
            return flatten_log(self.api.get(f"/tasks/{task_id}/log", {"between": f"-{window_ms}"}))
        except ApiError as exc:
            if exc.status == 404:
                return []
            raise

    def requests_for(self, task_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not task_ids:
            return {}
        rows = self.api.get("/tasks/requests", {"task_ids": ",".join(task_ids)})
        return {tid: row for tid, row in zip(task_ids, rows, strict=True) if row}

    def door_states(self) -> dict[str, dict[str, Any]]:
        return {
            d["name"]: self.api.get(f"/doors/{d['name']}/state") for d in self.api.get("/doors")
        }

    def lift_states(self) -> dict[str, dict[str, Any]]:
        return {
            lift["name"]: self.api.get(f"/lifts/{lift['name']}/state")
            for lift in self.api.get("/lifts")
        }

    def task_summary(self, task: dict[str, Any], now: int, request: dict | None) -> dict[str, Any]:
        booking = task["booking"]
        assigned = task.get("assigned_to") or {}
        start = task.get("unix_millis_start_time")
        finish = task.get("unix_millis_finish_time")
        estimate = task.get("original_estimate_millis")
        status = task.get("status")
        summary: dict[str, Any] = {
            "task_id": booking["id"],
            "category": task.get("category"),
            "status": status,
            "requester": booking.get("requester"),
            "robot": assigned.get("name"),
            "fleet": assigned.get("group"),
            "destinations": destinations(request),
            "requested_at": iso(booking.get("unix_millis_request_time")),
            "started_at": iso(start),
            "finished_at": iso(finish),
            "estimate_min": minutes(estimate) if estimate is not None else None,
        }
        if start and not finish:
            summary["running_min"] = minutes(now - start)
        elif status not in TERMINAL_STATUSES and booking.get("unix_millis_request_time"):
            summary["waiting_min"] = minutes(now - booking["unix_millis_request_time"])
        return {k: v for k, v in summary.items() if v not in (None, [])}

    def diagnose_robot(
        self,
        robot: dict[str, Any],
        now: int,
        doors: dict[str, dict[str, Any]],
        lifts: dict[str, dict[str, Any]],
        limits: Thresholds | None = None,
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Return (current task state, its log, findings) for a robot."""
        task, log = None, []
        if robot.get("task_id"):
            try:
                task = self.task_state(robot["task_id"])
            except ToolError:
                task = None
            log = self.task_log_entries(robot["task_id"])
        return task, log, diagnose(robot, now, task, log, doors, lifts, limits or self.limits)

    # ---- tools -------------------------------------------------------------------------

    def fleet_overview(
        self,
        fleet: Annotated[str | None, Field(description="Only this fleet. Omit for all.")] = None,
    ) -> dict[str, Any]:
        """Snapshot of every robot: status, battery, current task, level and issue count.

        Use for "what is going on right now", robot counts, and battery questions.
        """
        now = self.now()
        fleets: dict[str, list[dict[str, Any]]] = {}
        counts: dict[str, int] = {}
        for fleet_name, robot in self.robots():
            if fleet and fleet_name != fleet:
                continue
            row = {
                "robot": robot["name"],
                "status": robot.get("status"),
                "battery_pct": round((robot.get("battery") or 0) * 100),
                "task_id": robot.get("task_id") or None,
                "level": (robot.get("location") or {}).get("map"),
                "issues": len(robot.get("issues") or []),
                "last_update_age_s": round((now - robot["unix_millis_time"]) / 1000),
            }
            fleets.setdefault(fleet_name, []).append(
                {k: v for k, v in row.items() if v is not None}
            )
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        if fleet and not fleets:
            raise ToolError("not_found", f"No fleet named {fleet!r}.")
        all_rows = [r for rows in fleets.values() for r in rows]
        lowest = min(all_rows, key=lambda r: r["battery_pct"])
        return {
            "as_of": iso(now),
            "total_robots": len(all_rows),
            "status_counts": counts,
            "lowest_battery": {k: lowest[k] for k in ("robot", "battery_pct", "status")},
            "fleets": [{"fleet": name, "robots": rows} for name, rows in fleets.items()],
        }

    def find_stuck_robots(
        self,
        stale_after_s: Annotated[
            float, Field(ge=5, le=3600, description="State older than this counts as stale.")
        ] = 60,
        no_progress_min: Annotated[
            float, Field(ge=1, le=120, description="Working with no task log entry this long.")
        ] = 10,
    ) -> dict[str, Any]:
        """Scan all robots with rule-based checks and list those that look stuck, with evidence.

        Rules: error/offline status, stale state, waiting on a door that is not open, waiting on a
        lift that is out of service, blocked task, no task progress. Lower-severity findings (low
        battery, issues) are listed under "watch". Use for "is any robot stuck, and why?".
        """
        limits = Thresholds(stale_after_s, no_progress_min, self.limits.low_battery)
        now = self.now()
        doors, lifts = self.door_states(), self.lift_states()
        stuck, watch = [], []
        robots = self.robots()
        for fleet, robot in robots:
            _task, _log, findings = self.diagnose_robot(robot, now, doors, lifts, limits)
            if not findings:
                continue
            row = {
                "robot": robot["name"],
                "fleet": fleet,
                "status": robot.get("status"),
                "task_id": robot.get("task_id") or None,
                "findings": findings,
            }
            (stuck if is_stuck(findings) else watch).append(row)
        return {
            "as_of": iso(now),
            "checked_robots": len(robots),
            "stuck_count": len(stuck),
            "stuck": stuck,
            "watch": watch,
        }

    def explain_robot(
        self, robot: Annotated[str, Field(description="Robot name, e.g. tug_2.")]
    ) -> dict[str, Any]:
        """Everything needed to explain one robot: state, current task, recent log lines,
        rule findings, the doors and lifts its task log mentions, and alerts on its task.
        """
        now = self.now()
        fleet, state = self.find_robot(robot)
        doors, lifts = self.door_states(), self.lift_states()
        task, log, findings = self.diagnose_robot(state, now, doors, lifts)
        location = state.get("location") or {}
        out: dict[str, Any] = {
            "as_of": iso(now),
            "robot": robot,
            "fleet": fleet,
            "status": state.get("status"),
            "battery_pct": round((state.get("battery") or 0) * 100),
            "level": location.get("map"),
            "position": {"x": location.get("x"), "y": location.get("y")},
            "last_update_age_s": round((now - state["unix_millis_time"]) / 1000),
            "issues": state.get("issues") or [],
            "findings": findings,
        }
        if task:
            request = self.requests_for([task["booking"]["id"]]).get(task["booking"]["id"])
            out["task"] = self.task_summary(task, now, request)
            out["task"]["recent_log"] = [public(e) for e in log[-5:]]
            out["alerts"] = [
                _alert_row(a, now)
                for a in self.api.get(f"/alerts/requests/task/{task['booking']['id']}")
            ]
            text = " ".join(e["text"] for e in log[-5:])
            out["related_doors"] = [
                _door_row(n, doors[n], now)
                for n in dict.fromkeys(referenced_doors(text))
                if n in doors
            ]
            out["related_lifts"] = [
                _lift_row(lifts[n], now)
                for n in dict.fromkeys(referenced_lifts(text))
                if n in lifts
            ]
        return out

    def task_status(
        self, task_id: Annotated[str, Field(description="Task id, e.g. delivery.dispatch-204.")]
    ) -> dict[str, Any]:
        """Status, assignment, timing, destinations and last log lines of one task."""
        now = self.now()
        task = self.task_state(task_id)
        request = self.requests_for([task_id]).get(task_id)
        out = self.task_summary(task, now, request)
        out["as_of"] = iso(now)
        out["dispatch_status"] = (task.get("dispatch") or {}).get("status")
        out["last_log"] = [public(e) for e in self.task_log_entries(task_id)[-3:]]
        out["alert_ids"] = [a["id"] for a in self.api.get(f"/alerts/requests/task/{task_id}")]
        return out

    def list_tasks(
        self,
        status: Annotated[
            StrList | None,
            Field(description='Filter by status, e.g. ["queued"] or ["failed"].'),
        ] = None,
        assigned_to: Annotated[str | None, Field(description="Robot name.")] = None,
        category: Annotated[str | None, Field(description="patrol or delivery.")] = None,
        requester: Annotated[str | None, Field(description="Who requested the task.")] = None,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """List tasks with optional filters. Rows are compact: id, status, robot, destinations,
        and how long the task has been waiting or running.
        """
        params = {
            "status": ",".join(status) if status else None,
            "assigned_to": assigned_to,
            "category": category,
            "requester": requester,
            "limit": str(limit + 1),
        }
        now = self.now()
        rows = self.api.get("/tasks", {k: v for k, v in params.items() if v})
        requests = self.requests_for([t["booking"]["id"] for t in rows[:limit]])
        tasks = [self.task_summary(t, now, requests.get(t["booking"]["id"])) for t in rows[:limit]]
        return {
            "as_of": iso(now),
            "count": len(tasks),
            "truncated": len(rows) > limit,
            "tasks": tasks,
        }

    def late_tasks(
        self,
        queue_sla_min: Annotated[
            float, Field(ge=1, le=600, description="A queued task waiting longer is late.")
        ] = 15,
        overrun_factor: Annotated[
            float,
            Field(ge=1.0, le=5.0, description="A running task over estimate x factor is late."),
        ] = 1.25,
    ) -> dict[str, Any]:
        """Open tasks that are late, worst first.

        Late means: queued/standby longer than queue_sla_min ("queued_too_long"), or running longer
        than its original estimate x overrun_factor ("over_estimate"), or RMF status "delayed".
        minutes_late is time past the SLA, or past the estimate.
        """
        now = self.now()
        rows = self.api.get("/tasks", {"status": ",".join(OPEN_STATUSES), "limit": "1000"})
        late = []
        for task in rows:
            booking, status = task["booking"], task.get("status")
            start, estimate = (
                task.get("unix_millis_start_time"),
                task.get("original_estimate_millis"),
            )
            row = {
                "task_id": booking["id"],
                "status": status,
                "robot": (task.get("assigned_to") or {}).get("name"),
            }
            if status in {"queued", "standby"}:
                waiting = now - booking["unix_millis_request_time"]
                if waiting > queue_sla_min * 60_000:
                    late.append(
                        {
                            **row,
                            "reason": "queued_too_long",
                            "waiting_min": minutes(waiting),
                            "minutes_late": minutes(waiting - queue_sla_min * 60_000),
                        }
                    )
            elif start and estimate and now - start > estimate * overrun_factor:
                late.append(
                    {
                        **row,
                        "reason": "over_estimate",
                        "running_min": minutes(now - start),
                        "estimate_min": minutes(estimate),
                        "minutes_late": minutes(now - start - estimate),
                    }
                )
            elif status == "delayed":
                late.append({**row, "reason": "status_delayed", "minutes_late": None})
        late.sort(key=lambda r: r["minutes_late"] or 0, reverse=True)
        return {
            "as_of": iso(now),
            "rules": {"queue_sla_min": queue_sla_min, "overrun_factor": overrun_factor},
            "open_count": len(rows),
            "queued_count": sum(1 for t in rows if t.get("status") in {"queued", "standby"}),
            "late_count": len(late),
            "late": [{k: v for k, v in r.items() if v is not None} for r in late],
        }

    def door_and_lift_status(self) -> dict[str, Any]:
        """Current mode of every door and lift, which robots are waiting on them, and which
        ones need attention (door offline/unknown, lift out of service, or a robot waiting).
        """
        now = self.now()
        doors, lifts = self.door_states(), self.lift_states()
        levels = {
            d["name"]: lvl["name"]
            for lvl in self.api.get("/building_map")["levels"]
            for d in lvl["doors"]
        }
        waiting: dict[str, list[dict[str, Any]]] = {}
        for _fleet, robot in self.robots():
            if robot.get("status") != "working" or not robot.get("task_id"):
                continue
            log = self.task_log_entries(robot["task_id"])
            if not log:
                continue
            last = log[-1]
            for name in referenced_doors(last["text"]) + referenced_lifts(last["text"]):
                waiting.setdefault(name, []).append(
                    {"robot": robot["name"], "since_min": minutes(now - last["_ms"])}
                )
        door_rows = [
            {**_door_row(n, s, now), "level": levels.get(n), "waiting_robots": waiting.get(n, [])}
            for n, s in doors.items()
        ]
        lift_rows = [
            {**_lift_row(s, now), "waiting_robots": waiting.get(n, [])} for n, s in lifts.items()
        ]
        attention = [
            f"door {d['door']} is {d['mode']}" + _waiting_note(d)
            for d in door_rows
            if d["mode"] in {"offline", "unknown"} or (d["waiting_robots"] and d["mode"] != "open")
        ] + [
            f"lift {lift['lift']} is {'in' if lift['in_service'] else 'out of'} service "
            f"(mode {lift['mode']})" + _waiting_note(lift)
            for lift in lift_rows
            if not lift["in_service"] or lift["waiting_robots"]
        ]
        return {"as_of": iso(now), "doors": door_rows, "lifts": lift_rows, "attention": attention}

    def recent_alerts(
        self, since_min: Annotated[float, Field(ge=1, le=1440)] = 60
    ) -> dict[str, Any]:
        """Unresponded alerts raised in the last since_min minutes, newest first."""
        now = self.now()
        alerts = [
            _alert_row(a, now)
            for a in self.api.get("/alerts/unresponded_requests")
            if now - a["unix_millis_alert_time"] <= since_min * 60_000
        ]
        alerts.sort(key=lambda a: a["age_min"])
        return {"as_of": iso(now), "count": len(alerts), "alerts": alerts}

    def task_log(
        self,
        task_id: Annotated[str, Field(description="Task id.")],
        since_min: Annotated[float, Field(ge=1, le=1440)] = 60,
        limit: Annotated[int, Field(ge=1, le=100)] = 20,
    ) -> dict[str, Any]:
        """Event log of one task for the last since_min minutes (oldest first, at most limit
        entries, the most recent kept). Note the api-server default window is only 60 s; this
        tool always asks for since_min.
        """
        now = self.now()
        self.task_state(task_id)  # 404 -> not_found
        entries = self.task_log_entries(task_id, int(since_min * 60_000))
        return {
            "as_of": iso(now),
            "task_id": task_id,
            "entries": [public(e) for e in entries[-limit:]],
            "truncated": len(entries) > limit,
        }

    def find_places(
        self,
        level: Annotated[str | None, Field(description="Level name, e.g. L1.")] = None,
        query: Annotated[str | None, Field(description="Substring of the place name.")] = None,
    ) -> dict[str, Any]:
        """Named nav-graph places a robot can be sent to, with level and charger/pickup/dropoff
        flags. Use to check that a place exists before dispatching.
        """
        places = [
            p
            for p in self.places()
            if (not level or p["level"] == level)
            and (not query or query.lower() in p["place"].lower())
        ]
        return {"count": len(places), "places": places}

    def places(self) -> list[dict[str, Any]]:
        out = []
        for lvl in self.api.get("/building_map")["levels"]:
            for graph in lvl["nav_graphs"]:
                for vertex in graph["vertices"]:
                    if not vertex["name"]:
                        continue
                    params = {p["name"] for p in vertex["params"]}
                    out.append(
                        {
                            "place": vertex["name"],
                            "level": lvl["name"],
                            "charger": "is_charger" in params,
                            "pickup": "pickup_dispenser" in params,
                            "dropoff": "dropoff_ingestor" in params,
                        }
                    )
        return out


def _alert_row(alert: dict[str, Any], now: int) -> dict[str, Any]:
    row = {
        "alert_id": alert["id"],
        "tier": alert["tier"],
        "title": alert["title"],
        "subtitle": alert.get("subtitle") or None,
        "message": alert["message"],
        "task_id": alert.get("task_id"),
        "time": iso(alert["unix_millis_alert_time"]),
        "age_min": minutes(now - alert["unix_millis_alert_time"]),
    }
    return {k: v for k, v in row.items() if v is not None}


def _door_row(name: str, state: dict[str, Any], now: int) -> dict[str, Any]:
    return {
        "door": name,
        "mode": DOOR_MODE_NAMES.get(state["current_mode"]["value"], "unknown"),
        "state_age_min": minutes(now - ros_time_ms(state["door_time"])),
    }


def _lift_row(state: dict[str, Any], now: int) -> dict[str, Any]:
    mode = LIFT_MODE_NAMES.get(state["current_mode"], "unknown")
    return {
        "lift": state["lift_name"],
        "mode": mode,
        "in_service": mode in LIFT_IN_SERVICE_MODES,
        "current_floor": state["current_floor"],
        "destination_floor": state["destination_floor"],
        "door": LIFT_DOOR_NAMES.get(state["door_state"], "unknown"),
        "motion": LIFT_MOTION_NAMES.get(state["motion_state"], "unknown"),
        "state_age_min": minutes(now - ros_time_ms(state["lift_time"])),
    }


def _waiting_note(row: dict[str, Any]) -> str:
    if not row["waiting_robots"]:
        return ""
    who = ", ".join(f"{w['robot']} ({w['since_min']} min)" for w in row["waiting_robots"])
    return f"; waiting: {who}"
