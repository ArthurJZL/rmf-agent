"""CohereModel against the real cohere SDK, with HTTP mocked at the transport layer (no key)."""

import json

import cohere
import httpx
import pytest

from rmf_agent.agent import Agent
from rmf_agent.agent.cassette import CassetteExhausted, RecordingModel, ReplayModel
from rmf_agent.agent.fake import ScriptedFakeModel
from rmf_agent.agent.model import CohereModel


def cohere_client(responses, requests):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(200, json=responses.pop(0))

    return cohere.ClientV2(
        api_key="test", httpx_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


TOOL_CALL = {
    "id": "r1",
    "finish_reason": "TOOL_CALL",
    "message": {
        "role": "assistant",
        "tool_plan": "I will scan the fleet.",
        "tool_calls": [
            {
                "id": "tc_1",
                "type": "function",
                "function": {"name": "find_stuck_robots", "arguments": "{}"},
            }
        ],
    },
    "usage": {"tokens": {"input_tokens": 1200, "output_tokens": 30}},
}
FINAL = {
    "id": "r2",
    "finish_reason": "COMPLETE",
    "message": {
        "role": "assistant",
        "content": [{"type": "text", "text": "[tug_2] is waiting on [fire_door_a]."}],
    },
    "usage": {"tokens": {"input_tokens": 1900, "output_tokens": 25}},
}


def test_cohere_model_round_trip_through_the_sdk(toolbox_for):
    requests = []
    model = CohereModel(
        "command-a-plus-05-2026",
        client=cohere_client([TOOL_CALL, FINAL], requests),
        min_interval_s=0,
    )
    agent = Agent(model, toolbox_for("stuck_at_door"))
    result = agent.ask(agent.new_session(), "Which robot is stuck?")

    assert result.answer == "[tug_2] is waiting on [fire_door_a]."
    assert (result.input_tokens, result.output_tokens) == (3100, 55)
    first, second = requests
    assert first["model"] == "command-a-plus-05-2026" and first["temperature"] == 0.0
    assert {t["function"]["name"] for t in first["tools"]} >= {"find_stuck_robots"}
    assert [m["role"] for m in second["messages"]] == ["system", "user", "assistant", "tool"]
    assistant, tool = second["messages"][2], second["messages"][3]
    assert assistant["tool_calls"][0]["id"] == "tc_1"
    assert "tool_plan" not in assistant  # newer Command models reject it in history
    assert tool["tool_call_id"] == "tc_1"
    assert json.loads(tool["content"][0]["text"])["stuck"][0]["robot"] == "tug_2"


def test_cohere_model_requires_a_key(monkeypatch):
    monkeypatch.delenv("CO_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CO_API_KEY"):
        CohereModel("command-a-plus-05-2026")


def test_record_then_replay_is_identical(tmp_path, toolbox_for):
    path = tmp_path / "cassette.json"
    recorder = RecordingModel(ScriptedFakeModel(), path)
    agent = Agent(recorder, toolbox_for("stuck_at_door"))
    original = agent.ask(agent.new_session(), "Which robot is stuck and why?")
    recorder.save()
    assert json.loads(path.read_text())["fake_model"] is True

    replay = ReplayModel(path)
    agent = Agent(replay, toolbox_for("stuck_at_door"))
    replayed = agent.ask(agent.new_session(), "Which robot is stuck and why?")
    assert replayed.answer == original.answer and replay.drift == []
    with pytest.raises(CassetteExhausted):
        replay.chat([], [])


def test_replay_reports_drift(tmp_path, toolbox_for):
    path = tmp_path / "cassette.json"
    recorder = RecordingModel(ScriptedFakeModel(), path)
    Agent(recorder, toolbox_for("stuck_at_door")).ask(
        Agent(recorder, toolbox_for("stuck_at_door")).new_session(), "Which robot is stuck?"
    )
    recorder.save()
    replay = ReplayModel(path)
    agent = Agent(replay, toolbox_for("nominal"))  # different world -> different tool results
    agent.ask(agent.new_session(), "Which robot is stuck?")
    assert replay.drift
