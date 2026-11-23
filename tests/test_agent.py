import json
import time

from rmf_agent.agent import Agent, AgentConfig, ModelTurn, ToolCall, Tracer
from rmf_agent.agent.fake import ScriptedFakeModel


class ScriptModel:
    """Returns pre-written turns in order; records the messages it was given."""

    is_fake = True
    name = "script"

    def __init__(self, turns):
        self.turns = list(turns)
        self.seen = []

    def chat(self, messages, tools, **kwargs):
        self.seen.append(
            (
                json.loads(json.dumps(messages)),
                [t["function"]["name"] for t in tools],
                kwargs.get("tool_choice"),
            )
        )
        return self.turns.pop(0)


def call(name, **args):
    return ModelTurn(tool_calls=[ToolCall(f"c_{name}", name, args)], finish_reason="TOOL_CALL")


def test_react_loop_runs_tools_and_traces(toolbox_for):
    model = ScriptModel(
        [call("find_stuck_robots"), ModelTurn(text="[tug_2] waits on [fire_door_a]")]
    )
    tracer = Tracer()
    agent = Agent(model, toolbox_for("stuck_at_door"), tracer=tracer)
    result = agent.ask(agent.new_session(), "Which robot is stuck?")
    assert result.stopped_reason == "final" and result.steps == 2
    assert [c.name for c in result.tool_calls] == ["find_stuck_robots"]
    assert result.citations == ["fire_door_a", "tug_2"] and result.ungrounded_citations == []
    tool_message = model.seen[1][0][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "c_find_stuck_robots"
    assert json.loads(tool_message["content"][0]["text"])["stuck"][0]["robot"] == "tug_2"
    kinds = [e["event"] for e in tracer.events]
    assert kinds == ["turn_start", "model_call", "tool_call", "model_call", "turn_end"]
    assert tracer.of("tool_call")[0]["result_chars"] > 0


def test_ungrounded_citations_are_flagged(toolbox_for):
    model = ScriptModel([ModelTurn(text="[tug_9] is stuck at [door_x]")])
    agent = Agent(model, toolbox_for())
    result = agent.ask(agent.new_session(), "Is anything stuck?")
    assert result.ungrounded_citations == ["door_x", "tug_9"]


def test_step_budget(toolbox_for):
    model = ScriptModel([call("fleet_overview") for _ in range(3)])
    agent = Agent(model, toolbox_for(), AgentConfig(max_steps=3))
    result = agent.ask(agent.new_session(), "status?")
    assert result.stopped_reason == "max_steps" and result.steps == 3
    assert "stopped after 3 model steps" in result.answer


def test_model_timeout(toolbox_for):
    class Slow(ScriptModel):
        def chat(self, messages, tools, **kwargs):
            time.sleep(0.5)
            return ModelTurn(text="late")

    agent = Agent(Slow([]), toolbox_for(), AgentConfig(step_timeout_s=0.05))
    result = agent.ask(agent.new_session(), "status?")
    assert result.stopped_reason == "model_timeout"


def test_model_error_is_reported(toolbox_for):
    class Broken(ScriptModel):
        def chat(self, messages, tools, **kwargs):
            raise ConnectionError("network down")

    result = Agent(Broken([]), toolbox_for()).ask(
        Agent(Broken([]), toolbox_for()).new_session(), "status?"
    )
    assert result.stopped_reason == "model_error" and "network down" in result.answer


def test_refusal_happens_before_any_model_call(toolbox_for):
    model = ScriptModel([])
    agent = Agent(model, toolbox_for(allow_writes=True))
    result = agent.ask(agent.new_session(), "Open every door in the building.")
    assert result.refused and result.steps == 0 and model.seen == []


def test_plan_mode_plans_without_tools_then_executes(toolbox_for):
    model = ScriptModel(
        [
            ModelTurn(text="1. fleet_overview\n2. answer"),
            call("fleet_overview"),
            ModelTurn(text="8 robots"),
        ]
    )
    agent = Agent(model, toolbox_for(), AgentConfig(mode="plan"))
    result = agent.ask(agent.new_session(), "Give me an overview")
    assert result.plan.startswith("1. fleet_overview") and result.mode == "plan"
    assert model.seen[0][2] == "NONE"
    assert model.seen[1][0][-1]["content"].startswith("[controller]")
    assert result.answer == "8 robots"


def test_write_tools_hidden_in_read_only_mode(toolbox_for):
    model = ScriptModel([ModelTurn(text="ok")])
    Agent(model, toolbox_for()).ask(Agent(model, toolbox_for()).new_session(), "hi")
    assert "dispatch_task" not in model.seen[0][1]


def test_prompt_injection_cannot_trigger_writes(toolbox_for):
    """A model that obeys text planted in a task log is stopped by the policy, not by luck."""
    tools = toolbox_for("failed_task", allow_writes=True)
    agent = Agent(ScriptedFakeModel(adversarial=True), tools)
    result = agent.ask(agent.new_session(), "Summarize the log of patrol.dispatch-612.")
    blocked = [c for c in result.tool_calls if c.blocked]
    assert blocked and all(c.name == "cancel_task" for c in blocked)
    assert {c.blocked for c in blocked} == {"no_operator_instruction"}
    assert not tools.audit.writes()  # nothing reached the toolbox
    statuses = tools.call("list_tasks", {"status": ["canceled"]}).payload
    assert statuses["count"] == 0


def test_commit_without_dry_run_is_blocked(toolbox_for):
    tools = toolbox_for(allow_writes=True)
    agent = Agent(ScriptedFakeModel(adversarial=True), tools)
    result = agent.ask(agent.new_session(), "Dispatch a patrol from dock_1 to staging_a.")
    assert [c.blocked for c in result.tool_calls] == ["commit_needs_confirmation_turn"]
    assert not tools.audit.writes()


def test_two_turn_dispatch_with_confirmation(toolbox_for):
    tools = toolbox_for(allow_writes=True)
    agent = Agent(ScriptedFakeModel(), tools)
    session = agent.new_session()
    first = agent.ask(session, "Dispatch a patrol from dock_1 to staging_a, 2 rounds.")
    assert "ct_" in first.answer and not first.writes_committed
    second = agent.ask(session, "Confirm, go ahead.")
    assert [c.name for c in second.writes_committed] == ["dispatch_task"]
    assert "patrol.dispatch-900" in second.answer
    assert [r["decision"] for r in tools.audit.writes()] == ["dry_run_ok", "committed"]


def test_token_planted_in_data_is_not_trusted(toolbox_for):
    """Even a valid-looking token is useless unless a dry run in an earlier turn issued it."""
    tools = toolbox_for(allow_writes=True)
    forged = call(
        "dispatch_task",
        category="patrol",
        places=["dock_1"],
        dry_run=False,
        confirm_token="ct_from_a_log_line",
    )
    agent = Agent(ScriptModel([forged, ModelTurn(text="done")]), tools)
    session = agent.new_session()
    session.operator_turns.append("Dispatch a patrol to dock_1")
    result = agent.ask(session, "confirm")
    assert result.tool_calls[0].blocked == "token_not_from_earlier_dry_run"
