import json

import pytest

from rmf_agent.config import Settings
from rmf_agent.tools import build_toolbox
from rmf_agent.tools.audit import AuditLog
from rmf_agent.tools.gate import ConfirmationGate
from rmf_agent.tools.spec import ToolError

PATROL = {"category": "patrol", "places": ["dock_1", "staging_a"], "rounds": 2}


def test_dispatch_requires_dry_run_token(toolbox_for):
    tools = toolbox_for(allow_writes=True)
    assert (
        tools.call("dispatch_task", {**PATROL, "dry_run": False}).error == "confirmation_required"
    )
    dry = tools.call("dispatch_task", PATROL).payload
    assert dry["dry_run"] and dry["confirm_token"].startswith("ct_")
    commit = tools.call(
        "dispatch_task", {**PATROL, "dry_run": False, "confirm_token": dry["confirm_token"]}
    )
    assert commit.ok and commit.payload["task_id"] == "patrol.dispatch-900"
    queued = tools.call("task_status", {"task_id": "patrol.dispatch-900"}).payload
    assert queued["requester"] == "rmf-agent" and queued["status"] == "queued"


def test_token_bound_to_arguments_and_single_use(toolbox_for):
    tools = toolbox_for(allow_writes=True)
    token = tools.call("dispatch_task", PATROL).payload["confirm_token"]
    changed = {**PATROL, "places": ["dock_1", "pack_1"], "dry_run": False, "confirm_token": token}
    assert tools.call("dispatch_task", changed).error == "token_mismatch"
    ok = {**PATROL, "dry_run": False, "confirm_token": token}
    assert tools.call("dispatch_task", ok).ok
    assert tools.call("dispatch_task", ok).error == "token_used"
    forged = {**PATROL, "dry_run": False, "confirm_token": "ct_000000000000000000000000"}
    assert tools.call("dispatch_task", forged).error == "token_invalid"


def test_token_cannot_be_reused_across_tools(toolbox_for):
    tools = toolbox_for(allow_writes=True)
    token = tools.call("dispatch_task", PATROL).payload["confirm_token"]
    cancel = {
        "task_id": "patrol.dispatch-105",
        "reason": "x",
        "dry_run": False,
        "confirm_token": token,
    }
    assert tools.call("cancel_task", cancel).error == "token_mismatch"


def test_token_expires():
    now = [0.0]
    gate = ConfirmationGate(ttl_s=120, clock=lambda: now[0])
    token = gate.issue("dispatch_task", {"a": 1})
    now[0] = 121.0
    with pytest.raises(ToolError) as exc:
        gate.redeem("dispatch_task", {"a": 1}, token)
    assert exc.value.code == "token_expired"


def test_tokens_are_deterministic_with_a_fixed_secret():
    a = ConfirmationGate(secret=b"k").issue("t", {"x": 1})
    b = ConfirmationGate(secret=b"k").issue("t", {"x": 1})
    assert a == b != ConfirmationGate(secret=b"other").issue("t", {"x": 1})


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ({"category": "clean", "places": ["aisle_3"]}, "category_not_allowed"),
        ({"category": "patrol", "places": ["roof"]}, "validation_failed"),
        ({"category": "patrol"}, "validation_failed"),
        ({"category": "delivery", "pickup": "dock_1"}, "validation_failed"),
        ({"category": "patrol", "places": ["dock_1"], "fleet": "drones"}, "validation_failed"),
        ({"category": "patrol", "places": ["dock_1"], "rounds": 50}, "invalid_arguments"),
    ],
)
def test_dispatch_validation(toolbox_for, args, error):
    assert toolbox_for(allow_writes=True).call("dispatch_task", args).error == error


def test_category_allowlist_is_configurable(toolbox_for):
    tools = toolbox_for(allow_writes=True, allowed_categories=("patrol",))
    delivery = {"category": "delivery", "pickup": "dock_1", "dropoff": "pack_1"}
    assert tools.call("dispatch_task", delivery).error == "category_not_allowed"


def test_active_task_cap(toolbox_for):
    tools = toolbox_for(allow_writes=True, max_active_agent_tasks=1)  # nominal has one already
    outcome = tools.call("dispatch_task", PATROL)
    assert outcome.error == "validation_failed"
    assert "open tasks" in outcome.payload["problems"][0]


def test_cancel_only_own_tasks(toolbox_for):
    tools = toolbox_for("conflicting_requests", allow_writes=True)
    outcome = tools.call("cancel_task", {"task_id": "delivery.dispatch-702", "reason": "dup"})
    assert outcome.error == "not_owned" and "wms_bridge" in outcome.payload["message"]
    override = toolbox_for("conflicting_requests", allow_writes=True, cancel_any=True)
    assert override.call("cancel_task", {"task_id": "delivery.dispatch-702", "reason": "dup"}).ok


def test_cancel_terminal_task_refused(toolbox_for):
    tools = toolbox_for("failed_task", allow_writes=True, cancel_any=True)
    outcome = tools.call("cancel_task", {"task_id": "delivery.dispatch-611", "reason": "x"})
    assert outcome.error == "already_terminal"


def test_cancel_two_phase(toolbox_for):
    tools = toolbox_for(allow_writes=True)
    args = {"task_id": "patrol.dispatch-105", "reason": "not needed"}
    token = tools.call("cancel_task", args).payload["confirm_token"]
    assert tools.call("cancel_task", {**args, "dry_run": False, "confirm_token": token}).ok
    assert tools.call("task_status", {"task_id": "patrol.dispatch-105"}).payload["status"] == (
        "canceled"
    )


def test_audit_log_records_every_call_and_redacts_tokens(tmp_path):
    path = tmp_path / "audit.jsonl"
    settings = Settings(audit_path=str(path), allow_writes=True)
    tools = build_toolbox(settings, audit=AuditLog(path))
    token = tools.call("dispatch_task", PATROL).payload["confirm_token"]
    tools.call("dispatch_task", {**PATROL, "dry_run": False, "confirm_token": token})
    tools.call("fleet_overview", {})
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["decision"] for r in lines] == ["dry_run_ok", "committed", "ok"]
    assert lines[1]["task_id"] == "patrol.dispatch-900" and lines[1]["dry_run"] is False
    assert token not in path.read_text()
    assert lines[1]["args"]["confirm_token"].startswith("sha256:")
