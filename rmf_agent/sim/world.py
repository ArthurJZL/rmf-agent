"""In-process fake of the rmf-web api-server endpoints that rmf-agent uses.

Paths, query parameters and response shapes mirror rmf-web
``packages/api-server/api_server/routes``.
State comes from a scenario and is changed only by ``POST /tasks/dispatch_task`` and
``POST /tasks/cancel_task``; the clock is fixed at the scenario's ``now`` unless a test calls
``advance``. There is no motion planning, traffic or ROS here: the point is a deterministic,
realistic API surface for tool, agent and evaluation tests.
"""

from __future__ import annotations

import copy
import re
import threading
from collections.abc import Callable
from typing import Any

from .scenario import WorldState, load_scenario

TERMINAL = {"completed", "failed", "canceled", "killed", "skipped"}
SIM_CATEGORIES = {"patrol", "delivery"}

Response = tuple[int, Any]
Handler = Callable[..., Response]


class FakeRmfServer:
    def __init__(self, state: WorldState) -> None:
        self.state = state
        self._lock = threading.Lock()
        self._next_task = 900
        self._routes: list[tuple[str, re.Pattern[str], Handler]] = [
            ("GET", re.compile(r"/time"), self._time),
            ("GET", re.compile(r"/building_map"), self._building_map),
            ("GET", re.compile(r"/fleets"), self._fleets),
            ("GET", re.compile(r"/fleets/(?P<name>[^/]+)/state"), self._fleet_state),
            ("GET", re.compile(r"/doors"), self._doors),
            ("GET", re.compile(r"/doors/(?P<name>[^/]+)/state"), self._door_state),
            ("GET", re.compile(r"/lifts"), self._lifts),
            ("GET", re.compile(r"/lifts/(?P<name>[^/]+)/state"), self._lift_state),
            ("GET", re.compile(r"/alerts/unresponded_requests"), self._alerts),
            ("GET", re.compile(r"/alerts/requests/task/(?P<task_id>[^/]+)"), self._task_alerts),
            ("GET", re.compile(r"/tasks"), self._tasks),
            ("GET", re.compile(r"/tasks/requests"), self._task_requests),
            ("GET", re.compile(r"/tasks/(?P<task_id>[^/]+)/state"), self._task_state),
            ("GET", re.compile(r"/tasks/(?P<task_id>[^/]+)/request"), self._task_request),
            ("GET", re.compile(r"/tasks/(?P<task_id>[^/]+)/log"), self._task_log),
            ("POST", re.compile(r"/tasks/dispatch_task"), self._dispatch),
            ("POST", re.compile(r"/tasks/cancel_task"), self._cancel),
        ]

    @classmethod
    def from_scenario(cls, name_or_path: str) -> FakeRmfServer:
        return cls(load_scenario(name_or_path))

    def handle(
        self,
        method: str,
        path: str,
        query: dict[str, str] | None = None,
        body: Any = None,
    ) -> Response:
        """Serve one request. Returns ``(http_status, json_body)``."""
        path = "/" + path.strip("/")
        for route_method, pattern, handler in self._routes:
            match = pattern.fullmatch(path)
            if match and route_method == method.upper():
                with self._lock:
                    kwargs = match.groupdict()
                    if method.upper() == "POST":
                        return handler(body or {}, **kwargs)
                    return handler(query or {}, **kwargs)
        return 404, {"detail": "Not Found"}

    def advance(self, ms: int) -> None:
        """Move the simulated clock forward (tests only; nothing else changes)."""
        with self._lock:
            self.state.now_ms += ms

    # ---- reads -------------------------------------------------------------------------

    def _time(self, _q: dict) -> Response:
        return 200, self.state.now_ms

    def _building_map(self, _q: dict) -> Response:
        return 200, copy.deepcopy(self.state.building_map)

    def _fleets(self, _q: dict) -> Response:
        return 200, copy.deepcopy(list(self.state.fleets.values()))

    def _fleet_state(self, _q: dict, name: str) -> Response:
        fleet = self.state.fleets.get(name)
        return (200, copy.deepcopy(fleet)) if fleet else (404, {"detail": "Not Found"})

    def _doors(self, _q: dict) -> Response:
        doors = [d for level in self.state.building_map["levels"] for d in level["doors"]]
        return 200, copy.deepcopy(doors)

    def _door_state(self, _q: dict, name: str) -> Response:
        door = self.state.doors.get(name)
        return (200, copy.deepcopy(door)) if door else (404, {"detail": "Not Found"})

    def _lifts(self, _q: dict) -> Response:
        return 200, copy.deepcopy(self.state.building_map["lifts"])

    def _lift_state(self, _q: dict, name: str) -> Response:
        lift = self.state.lifts.get(name)
        return (200, copy.deepcopy(lift)) if lift else (404, {"detail": "Not Found"})

    def _alerts(self, _q: dict) -> Response:
        return 200, copy.deepcopy(self.state.alerts)

    def _task_alerts(self, _q: dict, task_id: str) -> Response:
        return 200, [copy.deepcopy(a) for a in self.state.alerts if a["task_id"] == task_id]

    def _tasks(self, q: dict) -> Response:
        def wanted(key: str) -> set[str] | None:
            return set(q[key].split(",")) if q.get(key) else None

        ids, categories, requesters = wanted("task_id"), wanted("category"), wanted("requester")
        assigned, statuses = wanted("assigned_to"), wanted("status")
        rows = []
        for task in self.state.tasks.values():
            robot = (task.get("assigned_to") or {}).get("name")
            if ids and task["booking"]["id"] not in ids:
                continue
            if categories and task.get("category") not in categories:
                continue
            if requesters and task["booking"].get("requester") not in requesters:
                continue
            if assigned and robot not in assigned:
                continue
            if statuses and task.get("status") not in statuses:
                continue
            rows.append(task)
        offset = int(q.get("offset") or 0)
        limit = int(q.get("limit") or 100)
        return 200, copy.deepcopy(rows[offset : offset + limit])

    def _task_requests(self, q: dict) -> Response:
        ids = q.get("task_ids", "").split(",") if q.get("task_ids") else []
        return 200, [copy.deepcopy(self.state.task_requests.get(i)) for i in ids]

    def _task_state(self, _q: dict, task_id: str) -> Response:
        task = self.state.tasks.get(task_id)
        return (200, copy.deepcopy(task)) if task else (404, {"detail": "Not Found"})

    def _task_request(self, _q: dict, task_id: str) -> Response:
        request = self.state.task_requests.get(task_id)
        return (200, copy.deepcopy(request)) if request else (404, {"detail": "Not Found"})

    def _task_log(self, q: dict, task_id: str) -> Response:
        log = self.state.task_logs.get(task_id)
        if log is None:
            return 404, {"detail": "Not Found"}
        between = q.get("between") or "-60000"  # api-server default: last 60 s
        if between.startswith("-"):
            start, end = self.state.now_ms - int(between[1:]), self.state.now_ms
        else:
            start, end = (int(x) for x in between.split(","))

        def window(entries: list[dict]) -> list[dict]:
            return [e for e in entries if start <= e["unix_millis_time"] <= end]

        result = {"task_id": task_id, "log": window(log["log"]), "phases": {}}
        for phase_id, phase in log["phases"].items():
            result["phases"][phase_id] = {
                "log": window(phase["log"]),
                "events": {k: window(v) for k, v in phase["events"].items()},
            }
        return 200, copy.deepcopy(result)

    # ---- writes ------------------------------------------------------------------------

    def _places(self) -> set[str]:
        return {
            v["name"]
            for level in self.state.building_map["levels"]
            for graph in level["nav_graphs"]
            for v in graph["vertices"]
            if v["name"]
        }

    def _dispatch(self, body: dict) -> Response:
        request = body.get("request") or {}
        if body.get("type") != "dispatch_task_request" or "category" not in request:
            return 422, {"detail": "expected a dispatch_task_request with request.category"}
        category = request["category"]
        description = request.get("description") or {}
        errors = []
        if category not in SIM_CATEGORIES:
            errors.append(f"category {category!r} is not supported by any fleet")
        else:
            if category == "patrol":
                wanted = list(description.get("places", []))
            else:
                wanted = [description.get(k, {}).get("place") for k in ("pickup", "dropoff")]
            missing = [p for p in wanted if p not in self._places()]
            if missing or not wanted:
                errors.append(f"unknown places: {missing or wanted}")
        fleet = request.get("fleet_name")
        if fleet and fleet not in self.state.fleets:
            errors.append(f"unknown fleet {fleet!r}")
        if errors:
            return 400, {
                "success": False,
                "errors": [{"code": 1, "category": "invalid_request", "detail": e} for e in errors],
            }

        task_id = f"{category}.dispatch-{self._next_task}"
        self._next_task += 1
        now = self.state.now_ms
        stored = {
            **request,
            "unix_millis_request_time": now,
            "unix_millis_earliest_start_time": request.get("unix_millis_earliest_start_time")
            or now,
        }
        self.state.task_requests[task_id] = stored
        state = {
            "booking": {
                "id": task_id,
                "unix_millis_earliest_start_time": stored["unix_millis_earliest_start_time"],
                "unix_millis_request_time": now,
                "labels": request.get("labels") or [],
                "requester": request.get("requester"),
            },
            "category": category,
            "detail": description,
            "status": "queued",
            "dispatch": {"status": "queued"},
        }
        self.state.tasks[task_id] = state
        self.state.task_logs[task_id] = {
            "task_id": task_id,
            "log": [{"seq": 0, "tier": "info", "unix_millis_time": now, "text": "Task queued"}],
            "phases": {},
        }
        return 200, {"success": True, "state": copy.deepcopy(state)}

    def _cancel(self, body: dict) -> Response:
        if body.get("type") != "cancel_task_request" or "task_id" not in body:
            return 422, {"detail": "expected a cancel_task_request with task_id"}
        task = self.state.tasks.get(body["task_id"])
        if task is None:
            return 200, _failure(f"task {body['task_id']!r} not found")
        if task["status"] in TERMINAL:
            return 200, _failure(f"task is already {task['status']}")
        now = self.state.now_ms
        task["status"] = "canceled"
        task["unix_millis_finish_time"] = now
        task["cancellation"] = {"unix_millis_request_time": now, "labels": body.get("labels", [])}
        robot = (task.get("assigned_to") or {}).get("name")
        for fleet in self.state.fleets.values():
            state = fleet["robots"].get(robot or "")
            if state and state["task_id"] == task["booking"]["id"]:
                state.update(status="idle", task_id="")
        return 200, {"success": True}


def _failure(detail: str) -> dict[str, Any]:
    return {"success": False, "errors": [{"code": 1, "category": "cancel", "detail": detail}]}
