"""Run eval cases against a model (scripted fake, Cohere live, or cassette replay)."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..agent.cassette import RecordingModel, ReplayModel
from ..agent.fake import ScriptedFakeModel
from ..agent.loop import Agent, AgentConfig
from ..agent.model import CohereModel, ModelClient
from ..agent.trace import Tracer
from ..config import Settings
from ..tools import build_toolbox
from ..tools.audit import AuditLog
from ..tools.gate import ConfirmationGate
from .cases import Case
from .judge import Judge
from .scoring import CaseRun, CaseScore, score_case, summarize

ModelKind = Literal["fake", "adversarial", "live", "replay"]


@dataclass
class EvalOptions:
    model: ModelKind = "fake"
    model_name: str = Settings.model
    replay_dir: Path | None = None
    record_dir: Path | None = None
    trace_path: Path | None = None
    max_steps: int = 8
    judge: Judge | None = None


def make_model(case: Case, options: EvalOptions) -> ModelClient:
    if options.model == "fake":
        return ScriptedFakeModel()
    if options.model == "adversarial":
        return ScriptedFakeModel(adversarial=True)
    if options.model == "replay":
        if options.replay_dir is None:
            raise ValueError("replay needs replay_dir")
        return ReplayModel(options.replay_dir / f"{case.id}.json")
    return CohereModel(options.model_name)


def run_case(case: Case, options: EvalOptions, model: ModelClient | None = None) -> CaseRun:
    model = model or make_model(case, options)
    recorder = None
    if options.record_dir:
        recorder = RecordingModel(
            model, options.record_dir / f"{case.id}.json", meta={"case_id": case.id}
        )
        model = recorder
    settings = Settings(scenario=case.scenario, allow_writes=case.allow_writes, audit_path=None)
    audit = AuditLog(None)
    # A per-case secret makes confirm tokens reproducible, so cassettes replay without drift.
    gate = ConfirmationGate(secret=f"eval:{case.id}".encode())
    toolbox = build_toolbox(settings, audit=audit, gate=gate)
    tracer = Tracer(options.trace_path, run_id=case.id)
    agent = Agent(
        model,
        toolbox,
        AgentConfig(
            max_steps=options.max_steps,
            mode=case.mode,
            # live calls may sleep for rate limiting / 429 backoff inside one step
            step_timeout_s=240.0 if options.model == "live" else 60.0,
        ),
        tracer,
    )
    session = agent.new_session()
    turns, error = [], None
    try:
        for question in case.turns:
            turns.append(agent.ask(session, question))
    except Exception as exc:  # a crashed case is a failed case, not a crashed eval
        error = f"{type(exc).__name__}: {exc}"
    if recorder:
        recorder.save()
    drift = getattr(model, "drift", None)
    if drift and error is None:
        error = f"cassette replay drifted at model calls {drift}; re-record the cassette"
    return CaseRun(case=case, turns=turns, audit=audit.records, error=error)


def run_eval(
    cases: list[Case], options: EvalOptions, progress: Callable[[CaseScore], None] | None = None
) -> dict[str, Any]:
    scores: list[CaseScore] = []
    judged: dict[str, dict[str, Any]] = {}
    is_fake = None
    model_name = None
    for case in cases:
        model = make_model(case, options)
        is_fake, model_name = model.is_fake, model.name
        run = run_case(case, options, model)
        score = score_case(run)
        scores.append(score)
        if options.judge and case.rubric and run.turns:
            judged[case.id] = options.judge.grade(case.turns[-1], run.answer, case.rubric)
        if progress:
            progress(score)
    return {
        "meta": _meta(options, model_name, bool(is_fake), len(cases)),
        "summary": summarize(scores),
        "llm_judge": {"judge": options.judge.name, "scores": judged} if options.judge else None,
        "cases": [s.to_dict() for s in scores],
    }


def _meta(options: EvalOptions, model_name: str | None, is_fake: bool, n: int) -> dict[str, Any]:
    import cohere

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_kind": options.model,
        "model": model_name,
        "fake_model": is_fake,
        "cases": n,
        "git_commit": commit,
        "python": platform.python_version(),
        "cohere_sdk": cohere.__version__,
        "max_steps": options.max_steps,
    }
