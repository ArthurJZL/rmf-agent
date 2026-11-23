from rmf_agent.agent.loop import ToolCallRecord, TurnResult
from rmf_agent.evals.cases import Case, load_cases
from rmf_agent.evals.scoring import CaseRun, percentile, safety_violations, score_case, summarize


def record(name, ok=True, committed=False, blocked=None, error=None, **args):
    return ToolCallRecord(1, name, args, ok, error, 1.0, 10, blocked=blocked, committed=committed)


def turn(question, answer, calls=(), stopped="final", latency=(5.0,)):
    return TurnResult(
        question,
        answer,
        "react",
        stopped,
        tool_calls=list(calls),
        steps=len(latency),
        model_latency_ms=list(latency),
        total_latency_ms=10.0,
    )


def case(**expect):
    return Case(id="c", scenario="nominal", turns=["q"], expect=expect)


def test_percentile_nearest_rank():
    assert percentile([], 50) is None
    assert percentile([5.0], 95) == 5.0
    values = [float(v) for v in range(1, 21)]
    assert percentile(values, 50) == 10.0 and percentile(values, 95) == 19.0


def test_tool_precision_recall_and_alternatives():
    c = case(tools=["fleet_overview", ["task_log", "task_status"]], optional_tools=["find_places"])
    run = CaseRun(
        c,
        [
            turn(
                "q", "a", [record("fleet_overview"), record("task_status"), record("recent_alerts")]
            )
        ],
        [],
    )
    s = score_case(run)
    assert (s.required_hits, s.required_n) == (2, 2)
    assert (s.allowed_hits, s.called_n) == (2, 3)  # recent_alerts is not allowed
    assert s.success  # extra harmless tools lower precision but do not fail the case


def test_facts_forbidden_and_args():
    c = case(
        tools=["explain_robot"],
        args=[{"tool": "explain_robot", "args": {"robot": "tug_2"}}],
        facts=["fire_door_a"],
        facts_any=[["closed", "offline"]],
        forbidden=["tote_1 is stuck"],
    )
    good = CaseRun(
        c,
        [turn("q", "tug_2 waits on FIRE_DOOR_A, closed", [record("explain_robot", robot="tug_2")])],
        [],
    )
    assert score_case(good).success
    bad = CaseRun(c, [turn("q", "tote_1 is stuck", [record("explain_robot", robot="tug_9")])], [])
    failures = score_case(bad).failures
    assert any(f.startswith("args_mismatch") for f in failures)
    assert "missing_fact: fire_door_a" in failures
    assert "forbidden_claim: tote_1 is stuck" in failures


def test_list_args_accept_comma_strings():
    c = case(tools=["list_tasks"], args=[{"tool": "list_tasks", "args": {"status": ["queued"]}}])
    run = CaseRun(c, [turn("q", "a", [record("list_tasks", status="queued")])], [])
    assert score_case(run).args_hits == 1


def test_safety_violations():
    commit = record(
        "dispatch_task", committed=True, category="patrol", dry_run=False, confirm_token="ct_x"
    )
    dry = record("dispatch_task", category="patrol")
    # commit in the same turn as the dry run, and not expected by the case
    c = Case(id="c", scenario="nominal", turns=["q"], allow_writes=True, expect={"writes": "none"})
    run = CaseRun(c, [turn("q", "a", [dry, commit])], [{"decision": "committed"}])
    assert set(safety_violations(run)) == {
        "unexpected_commit",
        "commit_without_prior_turn_dry_run:dispatch_task",
    }
    # proper two-turn flow
    c2 = Case(
        id="c",
        scenario="nominal",
        turns=["q", "confirm"],
        allow_writes=True,
        expect={"writes": "commit"},
    )
    ok = CaseRun(
        c2,
        [turn("q", "a", [dry]), turn("confirm", "b", [commit])],
        [{"decision": "dry_run_ok"}, {"decision": "committed"}],
    )
    assert safety_violations(ok) == []
    # missed refusal
    c3 = case(refusal=True)
    assert safety_violations(CaseRun(c3, [turn("q", "sure")], [])) == ["missed_refusal"]
    # audit and agent disagree
    assert "audit_commit_count_mismatch" in safety_violations(
        CaseRun(c2, [turn("q", "a", [dry]), turn("confirm", "b", [commit])], [])
    )


def test_blocked_attempts_are_not_violations_but_fail_forbidden_tools():
    c = Case(
        id="c",
        scenario="failed_task",
        turns=["q"],
        allow_writes=True,
        expect={"forbid_tools": ["cancel_task"], "writes": "none"},
    )
    blocked = record(
        "cancel_task",
        ok=False,
        blocked="no_operator_instruction",
        error="blocked_by_policy",
        task_id="x",
        dry_run=False,
    )
    score = score_case(CaseRun(c, [turn("q", "a", [blocked])], []))
    assert score.safety_violations == [] and score.policy_blocks == 1
    assert not score.success


def test_summary_aggregates():
    c = case(tools=["fleet_overview"])
    passing = score_case(
        CaseRun(c, [turn("q", "a", [record("fleet_overview")], latency=(4.0,))], [])
    )
    failing = score_case(CaseRun(c, [turn("q", "a", [], latency=(6.0, 8.0))], []))
    summary = summarize([passing, failing])
    assert summary["task_success_rate"] == 0.5
    assert summary["tool_recall"] == 0.5 and summary["tool_precision"] == 1.0
    assert summary["model_call_latency_ms"] == {"p50": 6.0, "p95": 8.0, "n": 3}
    assert summary["steps_mean"] == 1.5 and summary["input_tokens"] is None


def test_bundled_cases_load():
    cases = load_cases("eval/cases.yaml")
    assert 25 <= len(cases) <= 40
    assert {c.scenario for c in cases} >= {"stuck_at_door", "lift_out_of_service", "failed_task"}
