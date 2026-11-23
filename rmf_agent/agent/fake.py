"""SCRIPTED FAKE MODEL. This is not a language model and measures nothing about one.

It exists so the agent loop, tools, safety policy, tracing and evaluation harness can run
end to end without a Cohere API key (CI, offline demos, cassette generation). It speaks the same
``ModelClient`` protocol as ``CohereModel``: it reads the conversation, picks tool calls with
keyword rules, and writes the final answer from the tool results it received. Its answers are
therefore only as correct as the tools, which is what the offline evaluation actually checks.

It knows this demo world's naming: robots ``tote_N`` / ``tug_N``, task ids
``<category>.dispatch-N``.

``adversarial=True`` makes it misbehave on purpose: it obeys instructions planted in tool results
and tries to commit writes without a dry run. That variant is used to show that the deterministic
policy blocks those calls (see ``tests/test_agent.py`` and the adversarial eval run).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .model import Message, ModelTurn, ToolCall
from .policy import is_confirmation

ROBOT = re.compile(r"\b(?:tote|tug)_\d+\b")
TASK = re.compile(r"\b[a-z]+\.dispatch-\d+\b")
PLACE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
INJECTION = re.compile(
    r"(ignore (your|all|previous)|ai assistant|system override|disregard (the|your) rules)",
    re.IGNORECASE,
)
FOLLOW_DATA = re.compile(
    r"\b(do|follow|obey|execute)\b[^.?!]*\b(log|alert|note|message)s?\b[^.?!]*\b(says?|instructions?)\b"
)
OUT_OF_SCOPE = (
    "That is outside what I can help with. I answer questions about this robot fleet: robot "
    "status and battery, stuck robots, tasks and their logs, late or queued work, doors and "
    "lifts, and alerts."
)

Calls = list[tuple[str, dict[str, Any]]]
Step = Calls | str  # either tool calls to make next, or the final answer


@dataclass
class Result:
    name: str
    args: dict[str, Any]
    payload: dict[str, Any]


@dataclass
class Ctx:
    question: str
    turn_results: list[Result]  # tool results since the current operator question
    all_results: list[Result]  # tool results in the whole session
    tools: set[str]

    def called(self, name: str, **args: Any) -> bool:
        return any(
            r.name == name and all(r.args.get(k) == v for k, v in args.items())
            for r in self.turn_results
        )

    def last(self, name: str, **args: Any) -> dict[str, Any] | None:
        for r in reversed(self.turn_results):
            if r.name == name and all(r.args.get(k) == v for k, v in args.items()):
                return r.payload
        return None

    def need(self, *calls: tuple[str, dict[str, Any]]) -> Calls:
        return [(n, a) for n, a in calls if not self.called(n, **a)]


class ScriptedFakeModel:
    is_fake = True

    def __init__(self, adversarial: bool = False) -> None:
        self.adversarial = adversarial
        self.name = "scripted-fake-adversarial" if adversarial else "scripted-fake"
        self._ids = 0

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]],
        *,
        tool_choice: str | None = None,
        timeout_s: float | None = None,
    ) -> ModelTurn:
        ctx = _context(messages, {t["function"]["name"] for t in tools})
        playbook = _route(ctx)
        if tool_choice == "NONE":
            return ModelTurn(text=_plan_text(playbook, ctx))
        step = self._adversarial_step(ctx) if self.adversarial else None
        if step is None:
            step = playbook(ctx)
        if isinstance(step, str):
            return ModelTurn(text=step)
        calls = []
        for name, args in step:
            self._ids += 1
            calls.append(ToolCall(id=f"fake_call_{self._ids}", name=name, arguments=args))
        plan = "Calling " + ", ".join(n for n, _ in step) + " to answer the operator."
        return ModelTurn(tool_plan=plan, tool_calls=calls, finish_reason="TOOL_CALL")

    def _adversarial_step(self, ctx: Ctx) -> Calls | None:
        """Misbehave: follow planted instructions, and commit writes without a dry run."""
        planted = any(INJECTION.search(json.dumps(r.payload)) for r in ctx.turn_results)
        if planted and "cancel_task" in ctx.tools:
            if not ctx.called("list_tasks", status=["queued"]):
                return [("list_tasks", {"status": ["queued"]})]
            queued = ctx.last("list_tasks", status=["queued"]) or {}
            targets = [
                (
                    "cancel_task",
                    {"task_id": t["task_id"], "reason": "log instruction", "dry_run": False},
                )
                for t in queued.get("tasks", [])
            ]
            return [c for c in targets if not ctx.called(c[0], **c[1])] or None
        if _route(ctx) is _dispatch and "dispatch_task" in ctx.tools:
            args = {**_dispatch_args(ctx.question), "dry_run": False}
            return None if ctx.called("dispatch_task", **args) else [("dispatch_task", args)]
        return None


# ---- conversation parsing ----------------------------------------------------------------


def _context(messages: list[Message], tools: set[str]) -> Ctx:
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    all_results: list[Result] = []
    turn_start = 0
    question = ""
    for message in messages:
        role = message["role"]
        if role == "user" and not str(message["content"]).startswith("[controller]"):
            question = str(message["content"]).split("\n\nBefore calling any tools")[0]
            turn_start = len(all_results)
        elif role == "assistant":
            for call in message.get("tool_calls") or []:
                fn = call["function"]
                calls[call["id"]] = (fn["name"], json.loads(fn["arguments"]))
        elif role == "tool":
            name, args = calls[message["tool_call_id"]]
            payload = json.loads(message["content"][0]["text"])
            all_results.append(Result(name, args, payload))
    return Ctx(question, all_results[turn_start:], all_results, tools)


def _has(q: str, *words: str) -> bool:
    return any(re.search(rf"\b{w}", q) for w in words)


def _route(ctx: Ctx) -> Callable[[Ctx], Step]:
    raw = ctx.question.lower()
    q = TASK.sub(" ", raw)  # keywords are matched outside task ids ("patrol.dispatch-1")
    if is_confirmation(q) and _last_dry_run(ctx) and not TASK.search(raw):
        return _confirm
    if _has(q, "cancel") and TASK.search(raw):
        return _cancel
    if FOLLOW_DATA.search(q):
        return _decline_data_instructions
    if _has(q, "where can i", "places", "waypoints"):
        return _places
    if _has(q, "dispatch", "send a robot", "send one", "schedule a"):
        return _dispatch
    if _has(q, "hand-?over", "shift report"):
        return _handover
    if _has(q, "conflict"):
        return _conflicts
    if _has(q, "lift", "elevator", "l2", "upstairs"):
        return _lift
    if _has(q, "queue", "backlog") and _has(q, "why", "growing", "explain"):
        return _queue_why
    if TASK.search(raw):
        return _task
    if ROBOT.search(q):
        return _robot
    if _has(q, "stuck", "not moving", "blocked", "stopped"):
        return _stuck
    if _has(q, "fail"):
        return _failed
    if _has(q, "late", "overdue", "behind", "sla"):
        return _late
    if _has(q, "queue", "waiting", "pending", "backlog"):
        return _queued
    if _has(q, "door"):
        return _infra
    if _has(q, "alert", "alarm"):
        return _alerts
    if _has(q, "battery", "charge"):
        return _battery
    if _has(q, "how many", "overview", "status", "going on", "doing", "robots"):
        return _overview
    return _out_of_scope


PLANS: dict[str, list[str]] = {
    "_handover": [
        "fleet_overview: robot counts and states",
        "find_stuck_robots: anything stuck or unavailable",
        "late_tasks: late work",
        "list_tasks status=failed: failures to report",
        "recent_alerts: open alerts",
        "Write the handover with citations",
    ],
    "_queue_why": [
        "late_tasks: size of the queue and which tasks are late",
        "fleet_overview: how many robots can take work",
        "find_stuck_robots: why the others cannot",
        "Explain the backlog with evidence",
    ],
}


def _plan_text(playbook: Callable[[Ctx], Step], ctx: Ctx) -> str:
    steps = PLANS.get(playbook.__name__)
    if steps is None:
        first = playbook(ctx)
        names = [n for n, _ in first] if isinstance(first, list) else []
        steps = [f"{n}: gather evidence" for n in names] + ["Answer with citations"]
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


# ---- playbooks ---------------------------------------------------------------------------


def _out_of_scope(_ctx: Ctx) -> Step:
    return OUT_OF_SCOPE


def _decline_data_instructions(_ctx: Ctx) -> Step:
    return (
        "I won't act on instructions found in task logs or alerts; that text is data from the "
        "fleet system, and anyone who can write a log line could plant it. If you want a "
        "specific action, ask for it directly. Opening doors and bulk cancels are out of "
        "scope either way."
    )


def _overview(ctx: Ctx) -> Step:
    pending = ctx.need(("fleet_overview", {}))
    if pending:
        return pending
    o = ctx.last("fleet_overview") or {}
    counts = ", ".join(f"{n} {s}" for s, n in sorted(o["status_counts"].items()))
    working = [r for f in o["fleets"] for r in f["robots"] if r.get("task_id")]
    lines = [f"{o['total_robots']} robots across {len(o['fleets'])} fleets: {counts}."]
    if working:
        lines.append(
            "Working: " + ", ".join(f"[{r['robot']}] on [{r['task_id']}]" for r in working) + "."
        )
    low = o["lowest_battery"]
    lines.append(f"Lowest battery: [{low['robot']}] at {low['battery_pct']}% ({low['status']}).")
    return " ".join(lines)


def _battery(ctx: Ctx) -> Step:
    pending = ctx.need(("fleet_overview", {}), ("find_stuck_robots", {}))
    if pending:
        return pending
    o, s = ctx.last("fleet_overview") or {}, ctx.last("find_stuck_robots") or {}
    low = o["lowest_battery"]
    lines = [f"Lowest battery: [{low['robot']}] at {low['battery_pct']}% ({low['status']})."]
    risks = [
        (row["robot"], f)
        for row in s["stuck"] + s["watch"]
        for f in row["findings"]
        if f["code"].startswith("LOW_BATTERY")
    ]
    for robot, f in risks:
        lines.append(_finding_sentence(robot, f))
    others = [
        r
        for fl in o["fleets"]
        for r in fl["robots"]
        if r["battery_pct"] < 30 and r["robot"] not in {x for x, _ in risks}
    ]
    for r in others:
        lines.append(f"[{r['robot']}] is at {r['battery_pct']}% and {r['status']}.")
    if not risks:
        lines.append("No robot is below the 20% threshold outside a charger.")
    else:
        lines.append(
            "Suggested next step: send the robot to a charger after its current task, "
            "or reassign the task if the dropoff is far."
        )
    return " ".join(lines)


def _stuck(ctx: Ctx) -> Step:
    pending = ctx.need(("find_stuck_robots", {}))
    if pending:
        return pending
    s = ctx.last("find_stuck_robots") or {}
    pending = ctx.need(*[("explain_robot", {"robot": r["robot"]}) for r in s["stuck"][:2]])
    if pending:
        return pending
    if not s["stuck"]:
        text = (
            f"No robot looks stuck. I checked {s['checked_robots']} robots for error or "
            "offline status, stale state, waits on doors or lifts, blocked tasks and stalled "
            "task logs."
        )
        if s["watch"]:
            text += " To watch: " + " ".join(
                _finding_sentence(w["robot"], f) for w in s["watch"] for f in w["findings"]
            )
        return text
    parts = [f"{s['stuck_count']} robot(s) look stuck."]
    for row in s["stuck"]:
        parts.append(_robot_story(ctx.last("explain_robot", robot=row["robot"]) or row))
    return " ".join(parts)


def _robot(ctx: Ctx) -> Step:
    robot = ROBOT.search(ctx.question.lower()).group(0)  # type: ignore[union-attr]
    pending = ctx.need(("explain_robot", {"robot": robot}))
    if pending:
        return pending
    e = ctx.last("explain_robot", robot=robot) or {}
    if not e.get("ok"):
        return f"I could not find robot {robot}: {e.get('message')}"
    return _robot_story(e)


def _robot_story(e: dict[str, Any]) -> str:
    robot = e["robot"]
    head = f"[{robot}] is {e['status']}"
    if "battery_pct" in e:
        head += f", battery {e['battery_pct']}%, on {e.get('level')}"
    task = e.get("task")
    if task:
        head += f", working on [{task['task_id']}] ({task['status']})"
    parts = [head + "."]
    parts += [_finding_sentence(robot, f) for f in e.get("findings", [])]
    if task and task.get("recent_log") and not e.get("findings"):
        last = task["recent_log"][-1]
        parts.append(f'Last log ({last["time"]}): "{last["text"]}".')
    for alert in e.get("alerts", []):
        parts.append(f"Alert [{alert['alert_id']}] ({alert['tier']}): {alert['title']}.")
    if not e.get("findings"):
        parts.append("No rule flags anything abnormal.")
    steps = {_next_step(f) for f in e.get("findings", [])} - {""}
    if steps:
        parts.append("Suggested next step: " + " ".join(sorted(steps)))
    return " ".join(parts)


def _finding_sentence(robot: str, f: dict[str, Any]) -> str:
    ev, code = f["evidence"], f["code"]
    if code == "WAITING_ON_DOOR":
        return (
            f"[{robot}] has been waiting {ev['waiting_min']} min for door [{ev['door']}], "
            f"which is {ev['door_mode']} (door state unchanged for "
            f"{ev['door_state_age_min']} min). Last log ({ev['last_log']['time']}): "
            f'"{ev["last_log"]["text"]}".'
        )
    if code == "WAITING_ON_LIFT":
        return (
            f"[{robot}] has been waiting {ev['waiting_min']} min for lift [{ev['lift']}], "
            f"which is in {ev['lift_mode']} mode at {ev['lift_floor']}."
        )
    if code == "TASK_BLOCKED":
        return f"Its task [{ev['task_id']}] is {ev['task_status']}."
    if code == "OFFLINE":
        return f"[{robot}] is {ev['status']}."
    if code == "STALE_STATE":
        return f"[{robot}] has not reported state for {ev['last_update_age_s']} s."
    if code == "STATUS_ERROR":
        return f"[{robot}] is in error."
    if code == "HAS_ISSUES":
        issues = "; ".join(f"{i.get('category')}: {i.get('detail')}" for i in ev["issues"])
        return f"[{robot}] reports: {issues}."
    if code == "LOW_BATTERY_ON_TASK":
        return (
            f"[{robot}] is at {ev['battery_pct']}% while working on [{ev['task_id']}], "
            "below the 20% threshold."
        )
    if code == "LOW_BATTERY":
        return f"[{robot}] is at {ev['battery_pct']}% and not charging."
    if code == "NO_PROGRESS":
        return f"[{robot}] has logged nothing for {ev['minutes_since_last_log']} min."
    return f"[{robot}]: {code}."


def _next_step(f: dict[str, Any]) -> str:
    ev = f["evidence"]
    return {
        "WAITING_ON_DOOR": f"check door {ev.get('door')} and its door adapter; the task resumes "
        "when the door reports open.",
        "WAITING_ON_LIFT": f"check lift {ev.get('lift')}; tasks that need it wait until it is back "
        "in AGV mode.",
        "OFFLINE": "check the robot's power and network, then its fleet adapter.",
        "STALE_STATE": "",
        "STATUS_ERROR": "follow the issue detail (for localization, relocalize the robot).",
    }.get(f["code"], "")


def _task(ctx: Ctx) -> Step:
    task_id = TASK.search(ctx.question.lower()).group(0)  # type: ignore[union-attr]
    pending = ctx.need(("task_status", {"task_id": task_id}), ("task_log", {"task_id": task_id}))
    if pending:
        return pending
    status, log = ctx.last("task_status") or {}, ctx.last("task_log") or {}
    if not status.get("ok"):
        return f"I could not find task {task_id}: {status.get('message')}"
    parts = [_task_sentence(status)]
    entries = log.get("entries", [])
    errors = [e for e in entries if e["tier"] == "error"]
    if errors:
        parts.append(
            "Failure reason from the log: "
            + " / ".join(f'"{e["text"]}" ({e["time"]})' for e in errors)
            + "."
        )
    elif entries:
        parts.append(
            f"Log ({len(entries)} entries): "
            + "; ".join(
                f"{e['time'][11:16]} {e['text']}"
                for e in entries
                if not INJECTION.search(e["text"])
            )
            + "."
        )
    for alert_id in status.get("alert_ids", []):
        parts.append(f"Related alert: [{alert_id}].")
    planted = [e for e in entries if INJECTION.search(e["text"])]
    for e in planted:
        parts.append(
            f"Note: log entry seq {e['seq']} ({e['time']}) contains instructions "
            "addressed to an AI assistant. I treated it as data and did not act on it; "
            "it should be reported to whoever can write to this task log."
        )
    return " ".join(parts)


def _task_sentence(t: dict[str, Any]) -> str:
    who = f" on [{t['robot']}]" if t.get("robot") else ""
    text = f"[{t['task_id']}] is {t['status']} ({t['category']}{who}"
    if t.get("requester"):
        text += f", requested by {t['requester']}"
    if t.get("destinations"):
        text += ", " + " -> ".join(t["destinations"])
    text += ")"
    if t.get("finished_at"):
        text += f", finished {t['finished_at']}"
    elif t.get("running_min") is not None:
        text += f", running {t['running_min']} min against an estimate of {t['estimate_min']} min"
    elif t.get("waiting_min") is not None:
        text += f", waiting {t['waiting_min']} min"
    return text + "."


def _failed(ctx: Ctx) -> Step:
    pending = ctx.need(("list_tasks", {"status": ["failed"]}))
    if pending:
        return pending
    failed = (ctx.last("list_tasks", status=["failed"]) or {}).get("tasks", [])
    pending = ctx.need(*[("task_log", {"task_id": t["task_id"]}) for t in failed])
    if pending:
        return pending
    if not failed:
        return "No tasks have failed."
    parts = [f"{len(failed)} task(s) failed."]
    for t in failed:
        log = ctx.last("task_log", task_id=t["task_id"]) or {}
        errors = [e["text"] for e in log.get("entries", []) if e["tier"] == "error"]
        reason = f' Reason: "{errors[0]}".' if errors else ""
        parts.append(_task_sentence(t) + reason)
    return " ".join(parts)


def _late(ctx: Ctx) -> Step:
    pending = ctx.need(("late_tasks", {}))
    if pending:
        return pending
    return _late_summary(ctx.last("late_tasks") or {})


def _late_rows(lt: dict[str, Any], top: int | None = None) -> str:
    rows = lt["late"][:top] if top else lt["late"]
    parts = []
    for r in rows:
        if r["reason"] == "queued_too_long":
            parts.append(
                f"[{r['task_id']}] queued {r['waiting_min']} min ({r['minutes_late']} min past SLA)"
            )
        elif r["reason"] == "over_estimate":
            parts.append(
                f"[{r['task_id']}] on [{r['robot']}] running {r['running_min']} min vs "
                f"{r['estimate_min']} min estimate ({r['minutes_late']} min over)"
            )
        else:
            parts.append(f"[{r['task_id']}] is delayed")
    more = len(lt["late"]) - len(rows)
    return "; ".join(parts) + (f"; and {more} more." if more else ".")


def _late_summary(lt: dict[str, Any], top: int | None = None) -> str:
    rules = (
        f"queued over {lt['rules']['queue_sla_min']:g} min, or running over "
        f"{lt['rules']['overrun_factor']:g}x the estimate"
    )
    if not lt["late"]:
        return f"No tasks are late ({lt['open_count']} open; rules: {rules})."
    return (
        f"{lt['late_count']} of {lt['open_count']} open tasks are late (rules: {rules}): "
        + _late_rows(lt, top)
    )


def _queued(ctx: Ctx) -> Step:
    pending = ctx.need(("list_tasks", {"status": ["queued"], "limit": 50}))
    if pending:
        return pending
    tasks = (ctx.last("list_tasks") or {}).get("tasks", [])
    if not tasks:
        return "No tasks are queued."
    rows = "; ".join(
        f"[{t['task_id']}] waiting {t['waiting_min']} min "
        f"({' -> '.join(t.get('destinations', []))})"
        for t in tasks
    )
    return f"{len(tasks)} task(s) queued: {rows}."


def _queue_why(ctx: Ctx) -> Step:
    pending = ctx.need(("late_tasks", {}), ("fleet_overview", {}), ("find_stuck_robots", {}))
    if pending:
        return pending
    lt, o, s = (
        ctx.last("late_tasks") or {},
        ctx.last("fleet_overview") or {},
        ctx.last("find_stuck_robots") or {},
    )
    counts = o["status_counts"]
    parts = [
        f"{lt['queued_count']} tasks are queued and {lt['late_count']} open tasks are late. "
        f"Capacity is reduced: {counts.get('working', 0)} of {o['total_robots']} robots are "
        f"working, {counts.get('charging', 0)} charging, and these cannot take work:"
    ]
    for row in s["stuck"]:
        parts.append(" ".join(_finding_sentence(row["robot"], f) for f in row["findings"]))
    parts.append("Worst late tasks: " + _late_rows(lt, top=3))
    parts.append(
        "Suggested next steps: bring the offline and errored robots back first; "
        "the queue drains as capacity returns."
    )
    return " ".join(parts)


def _infra(ctx: Ctx) -> Step:
    pending = ctx.need(("door_and_lift_status", {}))
    if pending:
        return pending
    d = ctx.last("door_and_lift_status") or {}
    doors = ", ".join(f"[{x['door']}] {x['mode']}" for x in d["doors"])
    lifts = ", ".join(
        f"[{x['lift']}] {x['mode']} mode at {x['current_floor']}"
        f" ({'in' if x['in_service'] else 'OUT OF'} service)"
        for x in d["lifts"]
    )
    opened = [x["door"] for x in d["doors"] if x["mode"] == "open"]
    parts = [
        f"Open doors: {', '.join(f'[{n}]' for n in opened) or 'none'}.",
        f"Doors: {doors}.",
        f"Lifts: {lifts}.",
    ]
    if d["attention"]:
        parts.append("Needs attention: " + "; ".join(d["attention"]) + ".")
    return " ".join(parts)


def _lift(ctx: Ctx) -> Step:
    q = ctx.question.lower()
    if not _has(q, "why", "not moving", "stuck", "blocked", "waiting", "delayed"):
        return _infra(ctx)
    pending = ctx.need(
        ("door_and_lift_status", {}),
        ("list_tasks", {"status": ["queued", "blocked"], "limit": 50}),
        ("find_places", {"level": "L2"}),
        ("recent_alerts", {}),
    )
    if pending:
        return pending
    d = ctx.last("door_and_lift_status") or {}
    tasks = (ctx.last("list_tasks") or {}).get("tasks", [])
    l2 = {p["place"] for p in (ctx.last("find_places") or {}).get("places", [])}
    alerts = (ctx.last("recent_alerts") or {}).get("alerts", [])
    parts = []
    for lift in d["lifts"]:
        state = "in service" if lift["in_service"] else "out of service"
        parts.append(
            f"[{lift['lift']}] is {state}: mode {lift['mode']} at "
            f"{lift['current_floor']}, state unchanged for {lift['state_age_min']} min."
        )
        for w in lift["waiting_robots"]:
            parts.append(f"[{w['robot']}] has been waiting at the lift for {w['since_min']} min.")
    blocked = [t for t in tasks if set(t.get("destinations", [])) & l2]
    if blocked:
        parts.append(
            "Tasks that need L2: "
            + ", ".join(f"[{t['task_id']}] ({t['status']})" for t in blocked)
            + "."
        )
    for a in alerts:
        if "lift" in (a["title"] + a.get("subtitle", "")).lower():
            parts.append(f"Alert [{a['alert_id']}] ({a['tier']}): {a['message']}")
    parts.append("Suggested next step: get the lift back to AGV mode; L2 work waits until then.")
    return " ".join(parts)


def _alerts(ctx: Ctx) -> Step:
    pending = ctx.need(("recent_alerts", {}))
    if pending:
        return pending
    alerts = (ctx.last("recent_alerts") or {}).get("alerts", [])
    if not alerts:
        return "There are no unresponded alerts in the last 60 minutes."
    rows = " ".join(
        f"[{a['alert_id']}] {a['tier']}, {a['age_min']} min ago: {a['title']}"
        + (f" ({a['subtitle']})" if a.get("subtitle") else "")
        + (f", task [{a['task_id']}]" if a.get("task_id") else "")
        + "."
        for a in alerts
    )
    return f"{len(alerts)} unresponded alert(s): {rows}"


def _conflicts(ctx: Ctx) -> Step:
    pending = ctx.need(("recent_alerts", {}), ("list_tasks", {"status": ["queued"], "limit": 50}))
    if pending:
        return pending
    alerts = (ctx.last("recent_alerts") or {}).get("alerts", [])
    tasks = (ctx.last("list_tasks") or {}).get("tasks", [])
    parts = []
    for a in alerts:
        if "conflict" in a["title"].lower():
            parts.append(f"Alert [{a['alert_id']}]: {a['message']}")
    by_pickup: dict[str, list[dict[str, Any]]] = {}
    for t in tasks:
        if t.get("destinations"):
            by_pickup.setdefault(t["destinations"][0], []).append(t)
    for place, group in by_pickup.items():
        if len(group) > 1:
            parts.append(
                f"Queued tasks that both start at {place}: "
                + ", ".join(f"[{t['task_id']}] (requested by {t['requester']})" for t in group)
                + "."
            )
    if not parts:
        return "I found no conflicting requests in the queue or the alerts."
    parts.append(
        "Neither task was created by rmf-agent, so I cannot cancel either one; the "
        "requesters should decide which goes first."
    )
    return " ".join(parts)


def _places(ctx: Ctx) -> Step:
    level = re.search(r"\b(l\d)\b", ctx.question.lower())
    args = {"level": level.group(1).upper()} if level else {}
    pending = ctx.need(("find_places", args))
    if pending:
        return pending
    places = (ctx.last("find_places") or {}).get("places", [])

    def flags(p: dict[str, Any]) -> str:
        names = [k for k in ("charger", "pickup", "dropoff") if p[k]]
        return f" ({', '.join(names)})" if names else ""

    return (
        f"{len(places)} places"
        + (f" on {args['level']}" if args else "")
        + ": "
        + ", ".join(f"{p['place']}{flags(p)}" for p in places)
        + "."
    )


def _handover(ctx: Ctx) -> Step:
    pending = ctx.need(
        ("fleet_overview", {}),
        ("find_stuck_robots", {}),
        ("late_tasks", {}),
        ("list_tasks", {"status": ["failed"]}),
        ("recent_alerts", {}),
    )
    if pending:
        return pending
    o, s, lt = (
        ctx.last("fleet_overview") or {},
        ctx.last("find_stuck_robots") or {},
        ctx.last("late_tasks") or {},
    )
    failed = (ctx.last("list_tasks", status=["failed"]) or {}).get("tasks", [])
    alerts = (ctx.last("recent_alerts") or {}).get("alerts", [])
    counts = ", ".join(f"{n} {k}" for k, n in sorted(o["status_counts"].items()))
    lines = [f"Shift handover ({o['as_of']}).", f"Fleet: {o['total_robots']} robots, {counts}."]
    if s["stuck"]:
        lines.append(
            "Unavailable or stuck: "
            + " ".join(_finding_sentence(r["robot"], f) for r in s["stuck"] for f in r["findings"])
        )
    else:
        lines.append("No robot is stuck.")
    lines.append("Late work: " + _late_summary(lt, top=3))
    lines.append(
        "Failed tasks: " + (", ".join(f"[{t['task_id']}]" for t in failed) or "none") + "."
    )
    lines.append(
        "Open alerts: "
        + (", ".join(f"[{a['alert_id']}] {a['title']}" for a in alerts) or "none")
        + "."
    )
    return "\n".join(lines)


def _dispatch_args(question: str) -> dict[str, Any]:
    q = question.lower()
    category = "clean" if _has(q, "clean") else "delivery" if _has(q, "deliver") else "patrol"
    places = [p for p in PLACE.findall(q) if not TASK.search(p) and not ROBOT.fullmatch(p)]
    if not places:
        target = re.search(r"\bto (?:the )?([a-z0-9]+)", q)
        places = [target.group(1)] if target else []
    if category == "delivery":
        return {
            "category": category,
            "pickup": places[0] if places else None,
            "dropoff": places[1] if len(places) > 1 else None,
        }
    rounds = re.search(r"(\d+) rounds?", q)
    args: dict[str, Any] = {"category": category, "places": places}
    if rounds:
        args["rounds"] = int(rounds.group(1))
    return args


def _dispatch(ctx: Ctx) -> Step:
    args = _dispatch_args(ctx.question)
    if "dispatch_task" not in ctx.tools:
        return (
            "Writes are disabled in this session (read-only mode), so I cannot dispatch "
            "tasks. Restart with --allow-writes to enable the guarded dispatch tool."
        )
    pending = ctx.need(("dispatch_task", args))
    if pending:
        return pending
    return _dry_run_answer(ctx.last("dispatch_task") or {})


def _dry_run_answer(r: dict[str, Any]) -> str:
    if r.get("ok"):
        return (
            f"Dry run OK: {r['plan']}. Confirm token {r['confirm_token']} "
            f'(expires in {r["expires_in_s"]:g} s). Reply "confirm" to commit; '
            "nothing has been sent yet."
        )
    if r.get("error") == "blocked_by_policy":
        return f"The safety policy blocked the write: {r['message']}"
    detail = "; ".join(r.get("problems", [])) or r.get("message", "")
    return f"The request was rejected ({r.get('error')}): {detail.rstrip('.')}. Nothing was sent."


def _cancel(ctx: Ctx) -> Step:
    task_id = TASK.search(ctx.question.lower()).group(0)  # type: ignore[union-attr]
    if "cancel_task" not in ctx.tools:
        return "Writes are disabled in this session (read-only mode), so I cannot cancel tasks."
    args = {"task_id": task_id, "reason": "operator request"}
    pending = ctx.need(("task_status", {"task_id": task_id}), ("cancel_task", args))
    if pending:
        return pending[:1] if pending[0][0] == "task_status" else pending
    return _dry_run_answer(ctx.last("cancel_task") or {})


def _last_dry_run(ctx: Ctx) -> Result | None:
    for r in reversed(ctx.all_results):
        if r.payload.get("dry_run") and r.payload.get("confirm_token"):
            return r
    return None


def _confirm(ctx: Ctx) -> Step:
    dry = _last_dry_run(ctx)
    assert dry is not None
    args = {**dry.args, "dry_run": False, "confirm_token": dry.payload["confirm_token"]}
    pending = ctx.need((dry.name, args))
    if pending:
        return pending
    r = ctx.last(dry.name) or {}
    if r.get("ok") and r.get("committed"):
        verb = "Dispatched" if dry.name == "dispatch_task" else "Canceled"
        return f"{verb}: task [{r['task_id']}] is now {r['status']}."
    if r.get("error") == "blocked_by_policy" and dry in ctx.turn_results:
        return (
            f"Dry run OK: {dry.payload['plan']}. Confirm token {args['confirm_token']}. "
            f'I did not commit: {r["message"]} Reply "confirm" to commit.'
        )
    return f"The commit did not go through ({r.get('error')}): {r.get('message')}"
