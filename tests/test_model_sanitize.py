from rmf_agent.agent.model import ModelTurn, ToolCall, _is_rate_limit, sanitize_messages


def test_sanitize_drops_tool_plan_only_from_assistant():
    turn = ModelTurn(tool_plan="look", tool_calls=[ToolCall("1", "list_robots", {})])
    msgs = [{"role": "user", "content": "q"}, turn.as_message()]
    clean = sanitize_messages(msgs)
    assert "tool_plan" not in clean[1]
    assert clean[1]["tool_calls"] == msgs[1]["tool_calls"]
    assert "tool_plan" in msgs[1]  # original untouched


def test_rate_limit_detection():
    class E(Exception):
        status_code = 429

    assert _is_rate_limit(E())
    assert not _is_rate_limit(ValueError())


def test_tool_choice_none_sends_no_tools_or_tool_choice():
    from types import SimpleNamespace

    from rmf_agent.agent.model import CohereModel

    seen = {}

    class Client:
        def chat(self, **kwargs):
            seen.update(kwargs)
            msg = SimpleNamespace(
                content=[SimpleNamespace(type="text", text="plan")], tool_calls=None, tool_plan=None
            )
            return SimpleNamespace(message=msg, usage=None, finish_reason="COMPLETE")

    model = CohereModel("m", client=Client(), min_interval_s=0)
    tools = [{"type": "function", "function": {"name": "x", "parameters": {}}}]
    turn = model.chat([{"role": "user", "content": "q"}], tools, tool_choice="NONE")
    assert turn.text == "plan"
    assert "tool_choice" not in seen and "tools" not in seen
