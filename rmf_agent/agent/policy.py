"""Deterministic safety policy around the model. It does not depend on the model behaving.

1. ``screen_request`` refuses out-of-scope control requests before any model call: direct door or
   lift commands, bulk destructive actions, safety bypasses, credential requests.
2. ``authorize_write`` runs before every write tool call the model proposes. A write needs an
   explicit operator instruction for that action in this session (for a cancel, the operator must
   name the task id). A commit (``dry_run=false``) additionally needs a confirmation in the
   *current* operator turn and a token that a dry run issued in an *earlier* turn, so the operator
   has seen the plan before confirming. Text inside tool results (task logs, alerts) can never
   satisfy these checks, which is what makes planted instructions inert.
3. The api-side gate (``tools/gate.py``) then checks the token against the exact arguments.

The patterns are intentionally narrow and are tested for false positives on ordinary questions
("why did the door not open?", "what is the battery threshold?").
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .model import ToolCall

_SENTENCE = r"[^.?!\n]*"
REFUSAL_RULES: list[tuple[str, str, list[str]]] = [
    (
        "direct_door_or_lift_control",
        "Doors and lifts are commanded by RMF's door and lift adapters as part of tasks. "
        "This agent does not open, hold, override or move them directly.",
        [
            rf"\b(open|unlock|hold|close)\b{_SENTENCE}\b(all|every|each)\b{_SENTENCE}\bdoors?\b",
            r"\b(open|unlock|hold|close)\s+(the\s+)?[a-z0-9_]*door",
            rf"\boverride\b{_SENTENCE}\b(lift|elevator|door)",
            rf"\b(lift|elevator|door)[a-z0-9_]*\b{_SENTENCE}\boverride\b",
            r"\b(call|send|move|force)\s+(the\s+)?(lift|elevator)",
        ],
    ),
    (
        "bulk_destructive_action",
        "Bulk cancels and kills are out of scope. I can cancel one task that rmf-agent "
        "dispatched, by id, after a dry run and your confirmation.",
        [
            rf"\b(cancel|kill|abort|delete|clear)\b{_SENTENCE}\b(all|every|each)\b"
            rf"{_SENTENCE}\b(tasks?|jobs?|queue|robots?)\b",
            rf"\b(clear|flush|purge|wipe)\b{_SENTENCE}\bqueue\b",
            r"\b(decommission|kill_task|skip_phase|rewind_task)\b",
        ],
    ),
    (
        "safety_bypass",
        "I will not change or bypass safety settings (battery thresholds, e-stops, obstacle "
        "detection, fire alarm behavior).",
        [
            rf"\b(disable|bypass|turn off|switch off|ignore|override|lower|remove)\b{_SENTENCE}"
            r"\b(safety|battery threshold|recharge threshold|e-?stops?|emergency stop|"
            r"obstacle detection|speed limits?|fire alarm)",
        ],
    ),
    (
        "credentials",
        "I do not handle or reveal tokens, secrets or passwords.",
        [r"\b(admin token|jwt secret|api[ _-]?key|password|credentials?)\b"],
    ),
]

WRITE_INTENT = {
    "dispatch_task": re.compile(r"\b(dispatch|send|patrol|deliver|delivery)\b", re.IGNORECASE),
    "cancel_task": re.compile(r"\bcancel\b", re.IGNORECASE),
}
CONFIRM = re.compile(
    r"\b(confirm(ed)?|go ahead|proceed|commit|approved?|yes,? (do it|go|please))\b", re.IGNORECASE
)
NEGATED_CONFIRM = re.compile(
    rf"\b(don'?t|do not|never|not yet|no)\b{_SENTENCE}\b(confirm|commit|proceed|go ahead)\b",
    re.IGNORECASE,
)
MULTISTEP = re.compile(
    r"\b(hand-?over|shift report|summari[sz]e|root cause|investigate|step by step|and then)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Screen:
    refused: bool
    code: str | None = None
    reason: str | None = None


def screen_request(text: str) -> Screen:
    lowered = text.lower()
    for code, reason, patterns in REFUSAL_RULES:
        if any(re.search(p, lowered) for p in patterns):
            return Screen(True, code, reason)
    return Screen(False)


def looks_multistep(text: str) -> bool:
    return bool(MULTISTEP.search(text)) or text.count("?") >= 2


def is_confirmation(text: str) -> bool:
    return bool(CONFIRM.search(text)) and not NEGATED_CONFIRM.search(text)


@dataclass(frozen=True)
class WriteDecision:
    allowed: bool
    reason: str | None = None
    message: str | None = None


def authorize_write(
    call: ToolCall,
    operator_turns: list[str],
    issued_tokens: dict[str, int],
) -> WriteDecision:
    """Decide whether a proposed write call may run. ``operator_turns[-1]`` is the current turn;
    ``issued_tokens`` maps confirm tokens from dry-run results to the turn index they came from.
    """
    current = len(operator_turns) - 1
    intent = WRITE_INTENT.get(call.name)
    if intent is None or not any(intent.search(t) for t in operator_turns):
        return WriteDecision(
            False,
            "no_operator_instruction",
            f"The operator has not asked for {call.name} in this session.",
        )
    if call.name == "cancel_task":
        task_id = str(call.arguments.get("task_id", ""))
        if not task_id or not any(task_id in t for t in operator_turns):
            return WriteDecision(
                False,
                "task_not_named_by_operator",
                "Cancels need the operator to name the task id.",
            )
    if call.arguments.get("dry_run", True) is False:
        if not is_confirmation(operator_turns[-1]):
            return WriteDecision(
                False,
                "commit_needs_confirmation_turn",
                "Show the dry-run plan and wait for the operator to confirm.",
            )
        token = call.arguments.get("confirm_token")
        issued_in = issued_tokens.get(str(token))
        if issued_in is None or issued_in >= current:
            return WriteDecision(
                False,
                "token_not_from_earlier_dry_run",
                "Commit needs a token from a dry run shown in an earlier turn.",
            )
    return WriteDecision(True)
