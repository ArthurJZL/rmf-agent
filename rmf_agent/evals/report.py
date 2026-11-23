"""Render eval results as Markdown and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FAKE_BANNER = (
    "> **OFFLINE RUN WITH A SCRIPTED FAKE MODEL.** `{model}` is a keyword-rule script, not a "
    "language model. These numbers check the tools, the safety policy and the harness wiring "
    "end to end. They say nothing about how a real model (Cohere or otherwise) performs. "
    "Latencies are in-process wall clock and measure harness overhead only."
)
LIVE_BANNER = "> Live run against the Cohere API with model `{model}`."


def _fmt(value: Any, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    if pct:
        return f"{value * 100:.1f}%"
    return str(value)


def metrics_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    s = summary
    model_ms, e2e_ms = s["model_call_latency_ms"], s["e2e_latency_ms"]
    tokens = (
        f"{s['input_tokens']} in / {s['output_tokens']} out"
        if s["input_tokens"] is not None
        else "not reported by this model"
    )
    return [
        ("Cases", str(s["cases"])),
        ("Task success rate", f"{_fmt(s['task_success_rate'], True)} ({s['passed']}/{s['cases']})"),
        ("Tool-selection precision", _fmt(s["tool_precision"], True)),
        ("Tool-selection recall", _fmt(s["tool_recall"], True)),
        ("Argument accuracy", _fmt(s["argument_accuracy"], True)),
        ("Safety violations (target 0)", str(s["safety_violations"])),
        ("Write calls blocked by policy", str(s["policy_blocks"])),
        ("Write calls rejected by the token gate", str(s["gate_rejections"])),
        ("Cases with ungrounded citations", str(s["cases_with_ungrounded_citations"])),
        ("Model calls per case (mean / max)", f"{s['steps_mean']} / {s['steps_max']}"),
        (
            "Model-call latency p50 / p95 (ms)",
            f"{_fmt(model_ms['p50'])} / {_fmt(model_ms['p95'])}",
        ),
        (
            "End-to-end latency per case p50 / p95 (ms)",
            f"{_fmt(e2e_ms['p50'])} / {_fmt(e2e_ms['p95'])}",
        ),
        ("Tokens", tokens),
    ]


def render_markdown(results: dict[str, Any]) -> str:
    meta, summary = results["meta"], results["summary"]
    banner = (FAKE_BANNER if meta["fake_model"] else LIVE_BANNER).format(model=meta["model"])
    lines = [
        "# rmf-agent evaluation report",
        "",
        banner,
        "",
        f"- Generated: {meta['generated_at']} (git {meta['git_commit'] or 'unknown'})",
        f"- Model: `{meta['model']}` ({meta['model_kind']}), max {meta['max_steps']} model calls "
        "per turn",
        f"- Python {meta['python']}, cohere SDK {meta['cohere_sdk']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in metrics_rows(summary)],
        "",
        "## By scenario",
        "",
        "| Scenario | Passed |",
        "|---|---|",
        *[f"| {name} | {v['passed']}/{v['cases']} |" for name, v in summary["by_scenario"].items()],
        "",
        "## Cases",
        "",
        "| Case | Scenario | Result | Tools called | Model calls | Notes |",
        "|---|---|---|---|---|---|",
    ]
    judge = (results.get("llm_judge") or {}).get("scores", {})
    for case in results["cases"]:
        notes = "; ".join(case["failures"]) or ""
        if case["case_id"] in judge:
            notes = f"judge {judge[case['case_id']]['score']}/2. " + notes
        if case["policy_blocks"]:
            notes = f"{case['policy_blocks']} write call(s) blocked by policy. " + notes
        lines.append(
            f"| {case['case_id']} | {case['scenario']} | {'pass' if case['success'] else 'FAIL'} "
            f"| {', '.join(case['tools_called']) or '-'} | {case['steps']} | {notes.strip()} |"
        )
    lines += ["", "Full per-case answers and tool calls are in `results.json`.", ""]
    return "\n".join(lines)


def write_results(results: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out / "results.json", out / "report.md"
    json_path.write_text(json.dumps(results, indent=1, ensure_ascii=False) + "\n")
    md_path.write_text(render_markdown(results))
    return json_path, md_path
