"""Load a scenario file into rmf-web api-server JSON shapes.

A scenario is a YAML file that overrides the default world (``data/world.yaml``): robot states,
door and lift states, tasks with event logs, and alerts. Times are written relative to the
scenario clock (``"-12m"``, ``"+4m"``, ``"-30s"``) so files stay readable; the loader turns them
into the unix-millisecond fields the real api-server uses.

The output dicts follow the pydantic models in rmf-web ``packages/api-server/api_server/models``
(``FleetState``, ``RobotState``, ``TaskState``, ``TaskEventLog``, ``TaskRequest``, ``DoorState``,
``LiftState``, ``BuildingMap``, ``AlertRequest``).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import datetime
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

DATA = resources.files("rmf_agent.sim") / "data"

# rmf_door_msgs/DoorMode and rmf_lift_msgs/LiftState constants.
DOOR_MODES = {"closed": 0, "moving": 1, "open": 2, "offline": 3, "unknown": 4}
LIFT_DOOR = {"closed": 0, "moving": 1, "open": 2}
LIFT_MOTION = {"stopped": 0, "up": 1, "down": 2, "unknown": 3}
LIFT_MODES = {"unknown": 0, "human": 1, "agv": 2, "fire": 3, "offline": 4, "emergency": 5}
# rmf_building_map_msgs/Param types.
PARAM_STRING, PARAM_INT, PARAM_DOUBLE, PARAM_BOOL = 1, 2, 3, 4

_DURATION = re.compile(r"^([+-]?)(\d+(?:\.\d+)?)(ms|s|m|h)$")
_UNIT_MS = {"ms": 1, "s": 1_000, "m": 60_000, "h": 3_600_000}


def duration_ms(text: str | int) -> int:
    """Parse ``"10m"``, ``"-30s"`` or a plain integer of milliseconds."""
    if isinstance(text, int):
        return text
    match = _DURATION.match(text.strip())
    if not match:
        raise ValueError(f"bad duration: {text!r}")
    sign, value, unit = match.groups()
    ms = int(float(value) * _UNIT_MS[unit])
    return -ms if sign == "-" else ms


def at(now_ms: int, rel: str | int | None) -> int | None:
    """Turn a relative time (``"-12m"``) into unix millis; ``None`` stays ``None``."""
    if rel is None:
        return None
    if rel == "now":
        return now_ms
    return now_ms + duration_ms(rel)


@dataclass
class WorldState:
    """Mutable server state, already in api-server JSON shapes."""

    name: str
    description: str
    now_ms: int
    building_map: dict[str, Any]
    fleets: dict[str, dict[str, Any]]
    doors: dict[str, dict[str, Any]]
    lifts: dict[str, dict[str, Any]]
    tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_requests: dict[str, dict[str, Any]] = field(default_factory=dict)
    task_logs: dict[str, dict[str, Any]] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)


def list_scenarios() -> list[str]:
    return sorted(p.name.removesuffix(".yaml") for p in (DATA / "scenarios").iterdir())


def load_scenario(name_or_path: str | Path) -> WorldState:
    """Load a bundled scenario by name, or a scenario file by path."""
    path = Path(name_or_path)
    if path.suffix not in {".yaml", ".yml"}:
        path = Path(str(DATA / "scenarios" / f"{name_or_path}.yaml"))
    if not path.exists():
        raise FileNotFoundError(f"unknown scenario {name_or_path!r}; bundled: {list_scenarios()}")
    scenario = yaml.safe_load(path.read_text())
    world = yaml.safe_load((DATA / "world.yaml").read_text())
    return build_world(world, scenario)


def build_world(world: dict[str, Any], scenario: dict[str, Any]) -> WorldState:
    now_ms = int(datetime.fromisoformat(scenario["now"].replace("Z", "+00:00")).timestamp() * 1000)
    building_map = _building_map(world)
    fleets = _fleets(world["fleets"], scenario.get("robots", {}), now_ms)
    state = WorldState(
        name=scenario["name"],
        description=scenario.get("description", "").strip(),
        now_ms=now_ms,
        building_map=building_map,
        fleets=fleets,
        doors=_doors(building_map, scenario.get("doors", {}), now_ms),
        lifts=_lifts(building_map, scenario.get("lifts", {}), now_ms),
    )
    robot_fleet = {r: f for f, fs in fleets.items() for r in fs["robots"]}
    for spec in scenario.get("tasks", []):
        task_id = spec["id"]
        request = _task_request(spec, now_ms)
        state.task_requests[task_id] = request
        state.tasks[task_id] = _task_state(spec, request, robot_fleet, now_ms)
        state.task_logs[task_id] = _task_log(spec, now_ms)
    state.alerts = [_alert(a, now_ms) for a in scenario.get("alerts", [])]
    return state


def _param(name: str, value: Any) -> dict[str, Any]:
    param = {
        "name": name,
        "type": PARAM_STRING,
        "value_int": 0,
        "value_float": 0.0,
        "value_string": "",
        "value_bool": False,
    }
    if isinstance(value, bool):
        param.update(type=PARAM_BOOL, value_bool=value)
    else:
        param.update(value_string=str(value))
    return param


def _building_map(world: dict[str, Any]) -> dict[str, Any]:
    levels = []
    for level in world["levels"]:
        vertices = [
            {
                "x": v["x"],
                "y": v["y"],
                "name": v["name"],
                "params": [_param(k, val) for k, val in v.get("params", {}).items()],
            }
            for v in level["vertices"]
        ]
        doors = [
            {
                "name": d["name"],
                "v1_x": d["v1"][0],
                "v1_y": d["v1"][1],
                "v2_x": d["v2"][0],
                "v2_y": d["v2"][1],
                "door_type": d["door_type"],
                "motion_range": 1.57,
                "motion_direction": 1,
            }
            for d in level.get("doors", [])
        ]
        levels.append(
            {
                "name": level["name"],
                "elevation": level["elevation"],
                "images": [],
                "places": [],
                "doors": doors,
                "nav_graphs": [{"name": "0", "vertices": vertices, "edges": [], "params": []}],
                "wall_graph": {"name": "", "vertices": [], "edges": [], "params": []},
            }
        )
    lifts = [
        {
            "name": lift["name"],
            "levels": lift["levels"],
            "doors": [],
            "wall_graph": {"name": "", "vertices": [], "edges": [], "params": []},
            "ref_x": lift["ref_x"],
            "ref_y": lift["ref_y"],
            "ref_yaw": 0.0,
            "width": lift["width"],
            "depth": lift["depth"],
        }
        for lift in world.get("lifts", [])
    ]
    return {"name": world["name"], "levels": levels, "lifts": lifts}


def _fleets(
    defaults: dict[str, dict[str, Any]], overrides: dict[str, dict[str, Any]], now_ms: int
) -> dict[str, dict[str, Any]]:
    known = {r for robots in defaults.values() for r in robots}
    unknown = set(overrides) - known
    if unknown:
        raise ValueError(f"scenario overrides unknown robots: {sorted(unknown)}")
    fleets = {}
    for fleet_name, robots in defaults.items():
        states = {}
        for robot_name, base in robots.items():
            spec = {**base, **overrides.get(robot_name, {})}
            states[robot_name] = {
                "name": robot_name,
                "status": spec["status"],
                "task_id": spec.get("task_id", ""),
                "unix_millis_time": at(now_ms, spec.get("updated", "-2s")),
                "location": {"map": spec["map"], "x": spec["x"], "y": spec["y"], "yaw": 0.0},
                "battery": spec["battery"],
                "issues": copy.deepcopy(spec.get("issues", [])),
                "commission": {
                    "dispatch_tasks": True,
                    "direct_tasks": True,
                    "idle_behavior": True,
                },
                "mutex_groups": {"locked": [], "requesting": []},
            }
        fleets[fleet_name] = {"name": fleet_name, "robots": states}
    return fleets


def _time_msg(ms: int) -> dict[str, int]:
    return {"sec": ms // 1000, "nanosec": (ms % 1000) * 1_000_000}


def _doors(building_map: dict, overrides: dict, now_ms: int) -> dict[str, dict]:
    doors = {}
    for level in building_map["levels"]:
        for door in level["doors"]:
            spec = overrides.get(door["name"], {})
            doors[door["name"]] = {
                "door_time": _time_msg(at(now_ms, spec.get("updated", "-1m"))),
                "door_name": door["name"],
                "current_mode": {"value": DOOR_MODES[spec.get("mode", "closed")]},
            }
    return doors


def _lifts(building_map: dict, overrides: dict, now_ms: int) -> dict[str, dict]:
    lifts = {}
    for lift in building_map["lifts"]:
        spec = overrides.get(lift["name"], {})
        floor = spec.get("floor", lift["levels"][0])
        mode = LIFT_MODES[spec.get("mode", "agv")]
        lifts[lift["name"]] = {
            "lift_time": _time_msg(at(now_ms, spec.get("updated", "-20s"))),
            "lift_name": lift["name"],
            "available_floors": lift["levels"],
            "current_floor": floor,
            "destination_floor": spec.get("destination", floor),
            "door_state": LIFT_DOOR[spec.get("door", "closed")],
            "motion_state": LIFT_MOTION[spec.get("motion", "stopped")],
            "available_modes": [LIFT_MODES["human"], LIFT_MODES["agv"]],
            "current_mode": mode,
            "session_id": "",
        }
    return lifts


def task_description(spec: dict[str, Any]) -> dict[str, Any]:
    """Build an rmf_task ``patrol`` or ``delivery`` description from short scenario fields."""
    if spec["category"] == "patrol":
        return {"places": list(spec["places"]), "rounds": int(spec.get("rounds", 1))}
    if spec["category"] == "delivery":
        return {
            "pickup": {"place": spec["pickup"], "handler": spec["pickup"], "payload": []},
            "dropoff": {"place": spec["dropoff"], "handler": spec["dropoff"], "payload": []},
        }
    raise ValueError(f"unsupported task category in scenario: {spec['category']}")


def _task_request(spec: dict[str, Any], now_ms: int) -> dict[str, Any]:
    requested = at(now_ms, spec["requested"])
    return {
        "unix_millis_earliest_start_time": at(now_ms, spec.get("earliest_start")) or requested,
        "unix_millis_request_time": requested,
        "category": spec["category"],
        "description": task_description(spec),
        "labels": [f"requester={spec['requester']}"],
        "requester": spec["requester"],
    }


def _task_state(
    spec: dict[str, Any], request: dict[str, Any], robot_fleet: dict[str, str], now_ms: int
) -> dict[str, Any]:
    estimate = duration_ms(spec["estimate"])
    robot = spec.get("robot")
    status = spec["status"]
    state: dict[str, Any] = {
        "booking": {
            "id": spec["id"],
            "unix_millis_earliest_start_time": request["unix_millis_earliest_start_time"],
            "unix_millis_request_time": request["unix_millis_request_time"],
            "labels": request["labels"],
            "requester": spec["requester"],
        },
        "category": spec["category"],
        "detail": request["description"],
        "unix_millis_start_time": at(now_ms, spec.get("started")),
        "unix_millis_finish_time": at(now_ms, spec.get("finished")),
        "original_estimate_millis": estimate,
        "estimate_millis": estimate,
        "status": status,
        "dispatch": {"status": "queued" if status == "queued" else "dispatched"},
    }
    if robot:
        state["assigned_to"] = {"group": robot_fleet[robot], "name": robot}
        state["dispatch"]["assignment"] = {
            "fleet_name": robot_fleet[robot],
            "expected_robot_name": robot,
        }
    return state


def _task_log(spec: dict[str, Any], now_ms: int) -> dict[str, Any]:
    entries = [
        {
            "seq": seq,
            "tier": entry.get("tier", "info"),
            "unix_millis_time": at(now_ms, entry["t"]),
            "text": entry["text"],
        }
        for seq, entry in enumerate(spec.get("log", []))
    ]
    return {"task_id": spec["id"], "log": [], "phases": {"1": {"log": entries, "events": {}}}}


def _alert(spec: dict[str, Any], now_ms: int) -> dict[str, Any]:
    return {
        "id": spec["id"],
        "unix_millis_alert_time": at(now_ms, spec["t"]),
        "title": spec["title"],
        "subtitle": spec.get("subtitle", ""),
        "message": spec["message"],
        "display": True,
        "tier": spec["tier"],
        "responses_available": spec.get("responses", ["acknowledge"]),
        "alert_parameters": [],
        "task_id": spec.get("task_id"),
    }
