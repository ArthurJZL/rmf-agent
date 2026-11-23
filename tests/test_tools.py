import json

import pytest


def codes(row):
    return [f["code"] for f in row["findings"]]


def test_read_only_toolbox_hides_write_tools(toolbox_for):
    names = [s.name for s in toolbox_for().available()]
    assert len(names) == 10
    assert "dispatch_task" not in names and "cancel_task" not in names
    assert len(toolbox_for(allow_writes=True).available()) == 12


def test_every_tool_has_a_cohere_schema(toolbox_for):
    for spec in toolbox_for(allow_writes=True).available():
        tool = spec.cohere_tool()
        assert tool["type"] == "function"
        assert tool["function"]["description"]
        assert tool["function"]["parameters"]["type"] == "object"
        json.dumps(tool)


def test_nominal_has_no_stuck_robots(toolbox_for):
    result = toolbox_for("nominal").call("find_stuck_robots", {}).payload
    assert result["checked_robots"] == 8
    assert result["stuck"] == [] and result["watch"] == []


def test_stuck_at_door(toolbox_for):
    tools = toolbox_for("stuck_at_door")
    stuck = tools.call("find_stuck_robots", {}).payload["stuck"]
    assert [(s["robot"], codes(s)) for s in stuck] == [("tug_2", ["WAITING_ON_DOOR"])]
    evidence = stuck[0]["findings"][0]["evidence"]
    assert evidence["door"] == "fire_door_a" and evidence["door_mode"] == "closed"
    assert evidence["waiting_min"] == 8.0
    explained = tools.call("explain_robot", {"robot": "tug_2"}).payload
    assert explained["task"]["task_id"] == "delivery.dispatch-204"
    assert [a["alert_id"] for a in explained["alerts"]] == ["alert-2041"]
    assert explained["related_doors"][0]["door"] == "fire_door_a"
    assert "data_notice" in explained


def test_lift_out_of_service(toolbox_for):
    tools = toolbox_for("lift_out_of_service")
    status = tools.call("door_and_lift_status", {}).payload
    lift = status["lifts"][0]
    assert lift["lift"] == "lift_1" and lift["mode"] == "offline" and not lift["in_service"]
    assert lift["waiting_robots"] == [{"robot": "tote_5", "since_min": 12.0}]
    stuck = tools.call("find_stuck_robots", {}).payload["stuck"]
    assert [(s["robot"], codes(s)) for s in stuck] == [
        ("tote_5", ["TASK_BLOCKED", "WAITING_ON_LIFT"])
    ]


def test_low_battery_is_a_watch_item_not_stuck(toolbox_for):
    result = toolbox_for("low_battery").call("find_stuck_robots", {}).payload
    assert result["stuck"] == []
    assert [(w["robot"], codes(w)) for w in result["watch"]] == [
        ("tote_4", ["LOW_BATTERY_ON_TASK"])
    ]
    overview = toolbox_for("low_battery").call("fleet_overview", {}).payload
    assert overview["lowest_battery"] == {"robot": "tote_4", "battery_pct": 11, "status": "working"}


def test_late_tasks_rules(toolbox_for):
    result = toolbox_for("queue_backlog").call("late_tasks", {}).payload
    assert result["queued_count"] == 10 and result["late_count"] == 7
    late = {r["task_id"]: r for r in result["late"]}
    assert late["delivery.dispatch-530"] == {
        "task_id": "delivery.dispatch-530",
        "status": "queued",
        "reason": "queued_too_long",
        "waiting_min": 55.0,
        "minutes_late": 40.0,
    }
    assert late["patrol.dispatch-520"]["reason"] == "over_estimate"
    assert late["patrol.dispatch-520"]["minutes_late"] == 20.0
    stricter = toolbox_for("queue_backlog").call("late_tasks", {"queue_sla_min": 45}).payload
    assert {r["task_id"] for r in stricter["late"]} == {
        "delivery.dispatch-530",
        "delivery.dispatch-531",
        "patrol.dispatch-520",
    }


def test_task_log_and_status_for_failed_task(toolbox_for):
    tools = toolbox_for("failed_task")
    status = tools.call("task_status", {"task_id": "delivery.dispatch-611"}).payload
    assert status["status"] == "failed" and status["alert_ids"] == ["alert-6110"]
    log = tools.call("task_log", {"task_id": "delivery.dispatch-611", "limit": 2}).payload
    assert log["truncated"] is True
    assert "waypoint is not on the nav graph" in log["entries"][0]["text"]


def test_list_tasks_accepts_comma_string_and_lists(toolbox_for):
    tools = toolbox_for("queue_backlog")
    a = tools.call("list_tasks", {"status": "queued", "limit": 100}).payload
    b = tools.call("list_tasks", {"status": ["queued"], "limit": 100}).payload
    assert a == b and a["count"] == 10
    assert tools.call("list_tasks", {"status": ["queued"], "limit": 3}).payload["truncated"]


@pytest.mark.parametrize(
    ("tool", "args", "error"),
    [
        ("explain_robot", {"robot": "tug_9"}, "not_found"),
        ("task_status", {"task_id": "patrol.dispatch-0"}, "not_found"),
        ("explain_robot", {}, "invalid_arguments"),
        ("late_tasks", {"queue_sla_min": 0}, "invalid_arguments"),
        ("fleet_overview", {"unexpected": 1}, "invalid_arguments"),
        ("open_all_doors", {}, "unknown_tool"),
        ("dispatch_task", {"category": "patrol", "places": ["dock_1"]}, "writes_disabled"),
    ],
)
def test_errors_are_returned_not_raised(toolbox_for, tool, args, error):
    outcome = toolbox_for().call(tool, args)
    assert not outcome.ok and outcome.error == error and outcome.payload["ok"] is False


def test_find_places_flags(toolbox_for):
    places = toolbox_for().call("find_places", {"level": "L2"}).payload["places"]
    assert {p["place"] for p in places} == {"lift_lobby_L2", "pick_l2_1", "pick_l2_2", "charger_l2"}
    assert next(p for p in places if p["place"] == "charger_l2")["charger"] is True
