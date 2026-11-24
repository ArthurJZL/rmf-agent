"""The agent loop: ReAct over Cohere tool calls, with an optional plan-then-execute phase.

Per operator turn:
  1. ``screen_request`` may refuse before any model call.
  2. Plan mode (explicit, or auto for multi-part requests): one model call with tools disabled
     writes a short plan; the plan is kept in the conversation and the loop executes it.
  3. Loop until the model answers without tool calls, or ``max_steps`` model calls are used:
     model call (with timeout) -> for each tool call: write policy check -> tool execution (with
     timeout) -> tool message. Everything is traced.
  4. Citations in the final answer are checked against the text of the tool results.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from ..tools import Toolbox, ToolOutcome
from .model import Message, ModelClient, ModelTurn, ToolCall
from .policy import authorize_write, looks_multistep, screen_request
from .prompts import EXECUTE_PROMPT, PLAN_PROMPT, SYSTEM_PROMPT, system_prompt
from .trace import Tracer

Mode = Literal["react", "plan", "auto"]
T = TypeVar("T")
CITATION = re.compile(r"\[([A-Za-z0-9_.:\-]+)\]")


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int = 8
    step_timeout_s: float = 60.0
    tool_timeout_s: float = 15.0
    mode: Mode = "auto"


@dataclass
class ToolCallRecord:
    step: int
    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None
    latency_ms: float
    result_chars: int
    blocked: str | None = None
    committed: bool = False


@dataclass
class TurnResult:
    question: str
    answer: str
    mode: str
    stopped_reason: str  # final | refused | max_steps | model_timeout | model_error
    refusal_code: str | None = None
    plan: str | None = None
    steps: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    model_latency_ms: list[float] = field(default_factory=list)
    total_latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    citations: list[str] = field(default_factory=list)
    ungrounded_citations: list[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.stopped_reason == "refused"

    @property
    def writes_committed(self) -> list[ToolCallRecord]:
        return [c for c in self.tool_calls if c.committed]


@dataclass
class Session:
    messages: list[Message]
    operator_turns: list[str] = field(default_factory=list)
    issued_tokens: dict[str, int] = field(default_factory=dict)
    tool_text: list[str] = field(default_factory=list)


class Agent:
    def __init__(
        self,
        model: ModelClient,
        toolbox: Toolbox,
        config: AgentConfig | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        self.model = model
        self.toolbox = toolbox
        self.config = config or AgentConfig()
        self.tracer = tracer or Tracer()
        self.tools = [spec.cohere_tool() for spec in toolbox.available()]

    def new_session(self) -> Session:
        settings = getattr(self.toolbox, "settings", None)
        prompt = (
            system_prompt(settings.allow_writes, settings.allowed_categories)
            if settings is not None
            else SYSTEM_PROMPT
        )
        return Session(messages=[{"role": "system", "content": prompt}])

    def ask(self, session: Session, question: str) -> TurnResult:
        start = time.perf_counter()
        session.operator_turns.append(question)
        session.messages.append({"role": "user", "content": question})
        turn = len(session.operator_turns) - 1
        self.tracer.event(
            "turn_start",
            turn=turn,
            question=question,
            model=self.model.name,
            fake_model=self.model.is_fake,
            tools=[t["function"]["name"] for t in self.tools],
        )

        screen = screen_request(question)
        if screen.refused:
            answer = f"I can't do that. {screen.reason}"
            session.messages.append({"role": "assistant", "content": answer})
            self.tracer.event("refused", code=screen.code)
            result = TurnResult(question, answer, "none", "refused", refusal_code=screen.code)
            return self._finish(result, start)

        mode = self.config.mode
        if mode == "auto":
            mode = "plan" if looks_multistep(question) else "react"
        result = TurnResult(question, "", mode, "max_steps")

        if mode == "plan":
            plan_messages = [
                *session.messages[:-1],
                {"role": "user", "content": f"{question}\n\n{PLAN_PROMPT}"},
            ]
            plan_turn = self._model_call(plan_messages, result, tool_choice="NONE")
            if plan_turn is None:
                return self._finish(result, start)
            result.plan = plan_turn.text.strip()
            self.tracer.event("plan", plan=result.plan)
            session.messages.append({"role": "assistant", "content": result.plan})
            session.messages.append({"role": "user", "content": EXECUTE_PROMPT})

        while result.steps < self.config.max_steps:
            model_turn = self._model_call(session.messages, result)
            if model_turn is None:
                break
            session.messages.append(model_turn.as_message())
            if not model_turn.tool_calls:
                result.answer = model_turn.text.strip()
                result.stopped_reason = "final"
                break
            for call in model_turn.tool_calls:
                payload = self._run_tool(call, session, turn, result)
                session.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": [{"type": "text", "text": payload}],
                    }
                )
        if result.stopped_reason == "max_steps":
            called = ", ".join(c.name for c in result.tool_calls) or "none"
            result.answer = (
                f"I stopped after {result.steps} model steps without reaching an "
                f"answer. Tools called: {called}."
            )
        if result.stopped_reason != "final":
            session.messages.append({"role": "assistant", "content": result.answer})

        result.citations = sorted(set(CITATION.findall(result.answer)))
        corpus = "\n".join(session.tool_text)
        result.ungrounded_citations = [c for c in result.citations if c not in corpus]
        return self._finish(result, start)

    # ---- internals ---------------------------------------------------------------------

    def _model_call(
        self, messages: list[Message], result: TurnResult, tool_choice: str | None = None
    ) -> ModelTurn | None:
        result.steps += 1
        started = time.perf_counter()
        try:
            turn = _with_timeout(
                lambda: self.model.chat(
                    messages,
                    self.tools,
                    tool_choice=tool_choice,
                    timeout_s=self.config.step_timeout_s,
                ),
                self.config.step_timeout_s,
            )
        except FutureTimeout:
            result.stopped_reason = "model_timeout"
            result.answer = f"The model did not respond within {self.config.step_timeout_s} s."
            self.tracer.event("model_timeout", step=result.steps)
            return None
        except Exception as exc:
            result.stopped_reason = "model_error"
            result.answer = f"The model call failed: {type(exc).__name__}: {exc}"
            self.tracer.event("model_error", step=result.steps, error=repr(exc))
            return None
        latency = round((time.perf_counter() - started) * 1000, 2)
        result.model_latency_ms.append(latency)
        for attr in ("input_tokens", "output_tokens"):
            value = getattr(turn, attr)
            if value is not None:
                setattr(result, attr, (getattr(result, attr) or 0) + value)
        self.tracer.event(
            "model_call",
            step=result.steps,
            latency_ms=latency,
            finish_reason=turn.finish_reason,
            input_tokens=turn.input_tokens,
            output_tokens=turn.output_tokens,
            tool_plan=turn.tool_plan,
            text=turn.text or None,
            tool_calls=[{"name": c.name, "arguments": c.arguments} for c in turn.tool_calls],
        )
        return turn

    def _run_tool(self, call: ToolCall, session: Session, turn: int, result: TurnResult) -> str:
        spec = self.toolbox.specs.get(call.name)
        if spec is not None and spec.write:
            decision = authorize_write(call, session.operator_turns, session.issued_tokens)
            if not decision.allowed:
                payload = {
                    "ok": False,
                    "error": "blocked_by_policy",
                    "reason": decision.reason,
                    "message": decision.message,
                }
                result.tool_calls.append(
                    ToolCallRecord(
                        result.steps,
                        call.name,
                        call.arguments,
                        False,
                        "blocked_by_policy",
                        0.0,
                        0,
                        blocked=decision.reason,
                    )
                )
                self.tracer.event(
                    "policy_block",
                    step=result.steps,
                    tool=call.name,
                    arguments=call.arguments,
                    reason=decision.reason,
                )
                return json.dumps(payload)
        try:
            outcome = _with_timeout(
                lambda: self.toolbox.call(call.name, call.arguments), self.config.tool_timeout_s
            )
        except FutureTimeout:
            outcome = ToolOutcome(
                call.name,
                call.arguments,
                False,
                {"ok": False, "error": "tool_timeout"},
                0.0,
                error="tool_timeout",
            )
        text = outcome.as_text()
        session.tool_text.append(text)
        token = outcome.payload.get("confirm_token") if outcome.ok else None
        if token:
            session.issued_tokens[token] = turn
        committed = bool(outcome.ok and outcome.payload.get("committed"))
        result.tool_calls.append(
            ToolCallRecord(
                result.steps,
                call.name,
                call.arguments,
                outcome.ok,
                outcome.error,
                outcome.latency_ms,
                len(text),
                committed=committed,
            )
        )
        self.tracer.event(
            "tool_call",
            step=result.steps,
            tool=call.name,
            arguments=call.arguments,
            ok=outcome.ok,
            error=outcome.error,
            latency_ms=outcome.latency_ms,
            result_chars=len(text),
            committed=committed,
        )
        return text

    def _finish(self, result: TurnResult, start: float) -> TurnResult:
        result.total_latency_ms = round((time.perf_counter() - start) * 1000, 2)
        self.tracer.event(
            "turn_end",
            stopped_reason=result.stopped_reason,
            steps=result.steps,
            total_latency_ms=result.total_latency_ms,
            answer=result.answer,
            citations=result.citations,
            ungrounded=result.ungrounded_citations,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
        )
        return result


def _with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """Run ``fn`` in a worker thread and stop waiting after ``timeout_s``.

    Python cannot kill the thread; a timed-out call keeps running in the background and its
    result is discarded. The Cohere call also gets the timeout as a request option.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=timeout_s)
    finally:
        pool.shutdown(wait=False)
