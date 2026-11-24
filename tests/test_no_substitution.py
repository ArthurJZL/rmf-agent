"""The agent must never swap an unsupported request for a supported one, and must say clearly
when writes are disabled. Regression tests for the live-eval failures `write-category-not-allowed`
and `write-read-only-mode`.
"""

from rmf_agent.agent.model import ToolCall
from rmf_agent.agent.policy import authorize_write
from rmf_agent.agent.prompts import SYSTEM_PROMPT, system_prompt


def _dispatch(category: str) -> ToolCall:
    return ToolCall("c1", "dispatch_task", {"category": category, "places": ["aisle_3"]})


def test_substituted_category_is_blocked():
    decision = authorize_write(_dispatch("patrol"), ["Dispatch a cleaning task in aisle_3."], {})
    assert not decision.allowed
    assert decision.reason == "category_not_requested"
    assert "Never substitute" in (decision.message or "")


def test_requested_category_is_allowed():
    for text, category in [
        ("Dispatch a patrol from dock_1 to staging_a.", "patrol"),
        ("Send a delivery from dock_1 to pack_1.", "delivery"),
        ("Deliver a tote from dock_1 to pack_1.", "delivery"),
    ]:
        assert authorize_write(_dispatch(category), [text], {}).allowed, text


def test_unknown_category_is_left_to_the_tool_allowlist():
    # The tool rejects it with category_not_allowed and lists the supported categories.
    assert authorize_write(_dispatch("clean"), ["Dispatch a cleaning task."], {}).allowed


def test_category_rejection_lists_supported_categories(toolbox_for):
    outcome = toolbox_for(allow_writes=True).call(
        "dispatch_task", {"category": "clean", "places": ["aisle_3"]}
    )
    assert not outcome.ok
    assert outcome.payload["error"] == "category_not_allowed"
    assert "patrol" in outcome.payload["message"] and "delivery" in outcome.payload["message"]
    assert "Do not substitute" in outcome.payload["message"]


def test_system_prompt_states_session_mode():
    read_only = system_prompt(False, ("patrol", "delivery"))
    assert read_only.startswith(SYSTEM_PROMPT)
    assert "READ-ONLY" in read_only and "writes are disabled" in read_only
    writes = system_prompt(True, ("patrol", "delivery"))
    assert "only these task categories: patrol, delivery" in writes
    assert "Never change what the operator asked for" in SYSTEM_PROMPT
