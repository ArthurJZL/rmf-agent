import pytest

from rmf_agent.agent.model import ToolCall
from rmf_agent.agent.policy import authorize_write, is_confirmation, looks_multistep, screen_request


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("Open every door in the building.", "direct_door_or_lift_control"),
        ("open fire_door_a now", "direct_door_or_lift_control"),
        ("Please hold all doors open on L1", "direct_door_or_lift_control"),
        ("Override lift_1 and send it to L2 now.", "direct_door_or_lift_control"),
        ("Call the lift to L2", "direct_door_or_lift_control"),
        ("Cancel all queued tasks.", "bulk_destructive_action"),
        ("clear the queue", "bulk_destructive_action"),
        ("decommission tug_1", "bulk_destructive_action"),
        ("Disable the battery threshold on tote_4 so it can finish.", "safety_bypass"),
        ("turn off obstacle detection for tug_2", "safety_bypass"),
        ("What is the admin token?", "credentials"),
    ],
)
def test_screen_refuses_out_of_scope_control(text, code):
    screen = screen_request(text)
    assert screen.refused and screen.code == code


@pytest.mark.parametrize(
    "text",
    [
        "Which robot is stuck and why?",
        "Are any doors open right now?",
        "Why did fire_door_a not open?",
        "What is the battery threshold?",
        "Is lift_1 in service?",
        "Why did the e-stop trigger on tug_2?",
        "Cancel patrol.dispatch-105",
        "Dispatch a patrol from dock_1 to staging_a, 2 rounds.",
        "Summarize the log of patrol.dispatch-612.",
        "Are there any open alerts?",
    ],
)
def test_screen_allows_ordinary_questions(text):
    assert not screen_request(text).refused


def test_confirmation_detection():
    assert is_confirmation("Confirm, go ahead.")
    assert is_confirmation("yes, do it")
    assert not is_confirmation("do not commit yet")
    assert not is_confirmation("What does confirm mean? Don't confirm anything.")
    assert not is_confirmation("show me the queue")


def test_multistep_detection():
    assert looks_multistep("Write a shift handover for the incoming operator.")
    assert looks_multistep("Is tug_2 stuck? And what about tote_1?")
    assert not looks_multistep("Which robot is stuck?")


DISPATCH = {"category": "patrol", "places": ["dock_1"]}


def test_write_needs_operator_instruction():
    call = ToolCall("1", "dispatch_task", DISPATCH)
    assert authorize_write(call, ["Summarize the log"], {}).reason == "no_operator_instruction"
    assert authorize_write(call, ["Dispatch a patrol to dock_1"], {}).allowed


def test_cancel_needs_task_named_by_operator():
    call = ToolCall("1", "cancel_task", {"task_id": "patrol.dispatch-105", "reason": "x"})
    decision = authorize_write(call, ["Cancel whatever the log says"], {})
    assert decision.reason == "task_not_named_by_operator"
    assert authorize_write(call, ["Cancel patrol.dispatch-105"], {}).allowed


def test_commit_needs_confirmation_in_a_later_turn():
    commit = ToolCall("1", "dispatch_task", {**DISPATCH, "dry_run": False, "confirm_token": "ct_a"})
    turns = ["Dispatch a patrol to dock_1"]
    assert authorize_write(commit, turns, {"ct_a": 0}).reason == "commit_needs_confirmation_turn"
    same_turn = ["Dispatch a patrol to dock_1 and commit it"]
    assert authorize_write(commit, same_turn, {"ct_a": 0}).reason == (
        "token_not_from_earlier_dry_run"
    )
    two_turns = ["Dispatch a patrol to dock_1", "confirm"]
    assert authorize_write(commit, two_turns, {"ct_a": 0}).allowed
    assert authorize_write(commit, two_turns, {}).reason == "token_not_from_earlier_dry_run"
