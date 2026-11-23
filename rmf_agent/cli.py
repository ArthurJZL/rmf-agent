"""Command line: ``rmf-agent ask | chat | eval | sim | scenarios``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .agent.cassette import RecordingModel, ReplayModel
from .agent.fake import ScriptedFakeModel
from .agent.loop import Agent, AgentConfig, TurnResult
from .agent.model import CohereModel, ModelClient
from .agent.trace import Tracer
from .config import Settings
from .sim import list_scenarios

DEFAULT_CASES = Path("eval/cases.yaml")


def _agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--scenario", help=f"Fake api-server scenario ({', '.join(list_scenarios())}).")
    p.add_argument("--api-url", help="Real rmf-web api-server URL instead of the fake.")
    p.add_argument(
        "--allow-writes",
        action="store_true",
        default=None,
        help="Expose dispatch_task / cancel_task (still dry-run + confirm token).",
    )
    p.add_argument("--model", help="Cohere model name (default: RMF_AGENT_MODEL or built-in).")
    p.add_argument(
        "--offline",
        action="store_true",
        help="Use the scripted fake model (no API key; not a language model).",
    )
    p.add_argument("--replay", type=Path, help="Replay model responses from a cassette file.")
    p.add_argument("--record", type=Path, help="Record model responses to a cassette file.")
    p.add_argument("--mode", choices=["auto", "react", "plan"], default="auto")
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--step-timeout", type=float, default=60.0, help="Seconds per model call.")
    p.add_argument("--trace", type=Path, help="Write a JSONL trace of the run here.")
    p.add_argument("--audit-path", help="JSONL audit log of tool calls.")
    p.add_argument("-v", "--verbose", action="store_true", help="Print plan and tool calls.")


def _model(args: argparse.Namespace, settings: Settings) -> ModelClient:
    model: ModelClient
    if args.replay:
        model = ReplayModel(args.replay)
    elif args.offline:
        model = ScriptedFakeModel()
    else:
        model = CohereModel(settings.model)
    if args.record:
        model = RecordingModel(model, args.record)
    return model


def _build_agent(args: argparse.Namespace) -> tuple[Agent, Settings]:
    from .tools import build_toolbox

    settings = Settings.from_env(
        scenario=args.scenario,
        api_url=args.api_url,
        allow_writes=args.allow_writes,
        model=args.model,
        audit_path=args.audit_path,
    )
    config = AgentConfig(max_steps=args.max_steps, step_timeout_s=args.step_timeout, mode=args.mode)
    agent = Agent(_model(args, settings), build_toolbox(settings), config, Tracer(args.trace))
    return agent, settings


def _print_turn(result: TurnResult, agent: Agent, verbose: bool) -> None:
    if verbose and result.plan:
        print(f"plan:\n{result.plan}\n")
    if verbose:
        for call in result.tool_calls:
            status = (call.blocked and f"BLOCKED ({call.blocked})") or (
                "ok" if call.ok else call.error
            )
            print(
                f"  tool {call.name}({_compact(call.arguments)}) -> {status}, "
                f"{call.result_chars} chars, {call.latency_ms} ms"
            )
        print()
    print(result.answer)
    label = (
        f"{agent.model.name} (SCRIPTED FAKE, offline)" if agent.model.is_fake else agent.model.name
    )
    tokens = (
        f", tokens {result.input_tokens} in / {result.output_tokens} out"
        if result.input_tokens is not None
        else ""
    )
    print(
        f"\n[model {label} | mode {result.mode} | {result.steps} model call(s) | "
        f"{len(result.tool_calls)} tool call(s) | {result.total_latency_ms} ms{tokens}]"
    )


def _compact(arguments: dict) -> str:
    return ", ".join(f"{k}={v!r}" for k, v in arguments.items())


def _source(settings: Settings) -> str:
    return settings.api_url or f"fake api-server, scenario '{settings.scenario}'"


def cmd_ask(args: argparse.Namespace) -> int:
    agent, settings = _build_agent(args)
    if args.verbose:
        writes = "writes enabled (guarded)" if settings.allow_writes else "read-only"
        print(f"[{_source(settings)} | {writes}]")
    result = agent.ask(agent.new_session(), args.question)
    _print_turn(result, agent, args.verbose)
    if isinstance(agent.model, RecordingModel):
        agent.model.save()
    return 0 if result.stopped_reason in {"final", "refused"} else 1


def cmd_chat(args: argparse.Namespace) -> int:
    agent, settings = _build_agent(args)
    session = agent.new_session()
    print(f"rmf-agent chat on {_source(settings)}. Empty line or Ctrl-D to exit.")
    while True:
        try:
            question = input("\noperator> ").strip()
        except EOFError:
            break
        if not question:
            break
        _print_turn(agent.ask(session, question), agent, args.verbose)
    if isinstance(agent.model, RecordingModel):
        agent.model.save()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from .evals.cases import load_cases
    from .evals.judge import CohereJudge
    from .evals.report import metrics_rows, write_results
    from .evals.runner import EvalOptions, run_eval

    kind = "live" if args.live else args.model
    cases = load_cases(args.cases)
    if args.case:
        cases = [c for c in cases if c.id in set(args.case)]
    settings = Settings.from_env(model=args.model_name)
    options = EvalOptions(
        model=kind,
        model_name=settings.model,
        replay_dir=args.replay_dir,
        record_dir=args.record_dir,
        trace_path=args.trace,
        max_steps=args.max_steps,
        judge=CohereJudge(args.judge_model or settings.model) if args.judge else None,
    )
    out = args.out or Path("eval/results") / ("live" if kind == "live" else f"offline-{kind}")

    def progress(score) -> None:
        mark = "pass" if score.success else "FAIL"
        print(f"{mark:4} {score.case_id:32} {', '.join(score.failures)}")

    results = run_eval(cases, options, progress)
    json_path, md_path = write_results(results, out)
    if results["meta"]["fake_model"]:
        print(
            "\nNOTE: scripted fake model. These results test the harness, tools and safety "
            "policy, not a language model."
        )
    for name, value in metrics_rows(results["summary"]):
        print(f"  {name:44} {value}")
    if kind == "live":
        from .agent.model import CALL_COUNT

        print(f"  Cohere API calls: {CALL_COUNT['calls']} (429 retries: {CALL_COUNT['retries']})")
    print(f"\nwrote {md_path} and {json_path}")
    summary = results["summary"]
    if summary["safety_violations"]:
        print("safety violations found", file=sys.stderr)
        return 2
    if args.fail_under is not None and summary["task_success_rate"] < args.fail_under:
        print(f"task success below {args.fail_under}", file=sys.stderr)
        return 1
    return 0


def cmd_sim(args: argparse.Namespace) -> int:
    import uvicorn

    from .sim import FakeRmfServer
    from .sim.http_app import create_app

    server = FakeRmfServer.from_scenario(args.scenario)
    print(f"fake rmf-web api-server, scenario '{args.scenario}': {server.state.description}")
    uvicorn.run(create_app(server), host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_scenarios(_args: argparse.Namespace) -> int:
    from .sim import load_scenario

    for name in list_scenarios():
        print(f"{name:22} {load_scenario(name).description}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmf-agent", description="Cohere agent for Open-RMF fleet operations."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="Ask one question.")
    ask.add_argument("question")
    _agent_args(ask)
    ask.set_defaults(func=cmd_ask)

    chat = sub.add_parser("chat", help="Multi-turn session (needed to confirm a write).")
    _agent_args(chat)
    chat.set_defaults(func=cmd_chat)

    ev = sub.add_parser("eval", help="Run the evaluation cases.")
    ev.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    ev.add_argument("--case", action="append", help="Run only this case id (repeatable).")
    ev.add_argument(
        "--model",
        choices=["fake", "adversarial", "replay"],
        default="fake",
        help="fake (default), adversarial fake, or replay cassettes.",
    )
    ev.add_argument("--live", action="store_true", help="Use the real Cohere API (CO_API_KEY).")
    ev.add_argument("--model-name", help="Cohere model for --live.")
    ev.add_argument("--replay-dir", type=Path, help="Cassette directory for --model replay.")
    ev.add_argument("--record-dir", type=Path, help="Record a cassette per case here.")
    ev.add_argument("--judge", action="store_true", help="Also run the Cohere LLM judge.")
    ev.add_argument("--judge-model", help="Model for --judge (default: the agent model).")
    ev.add_argument("--max-steps", type=int, default=8)
    ev.add_argument("--trace", type=Path, help="Append all case traces to this JSONL file.")
    ev.add_argument("--out", type=Path, help="Output directory (default eval/results/<kind>).")
    ev.add_argument("--fail-under", type=float, help="Exit 1 if task success is below this.")
    ev.set_defaults(func=cmd_eval)

    sim = sub.add_parser("sim", help="Serve the fake api-server over HTTP.")
    sim.add_argument("--scenario", default="nominal", choices=list_scenarios())
    sim.add_argument("--host", default="127.0.0.1")
    sim.add_argument("--port", type=int, default=8000)
    sim.set_defaults(func=cmd_sim)

    scen = sub.add_parser("scenarios", help="List bundled scenarios.")
    scen.set_defaults(func=cmd_scenarios)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = args.func(args)
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"rmf-agent: {exc}", file=sys.stderr)
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()
