import pytest
from fastapi.testclient import TestClient

from rmf_agent.api import ApiError, HttpApi, InProcessApi
from rmf_agent.sim import FakeRmfServer, list_scenarios, load_scenario
from rmf_agent.sim.http_app import create_app
from rmf_agent.sim.scenario import duration_ms
from tests.rmf_web_models.alert_request import AlertRequest
from tests.rmf_web_models.rmf_api.dispatch_task_response import TaskDispatchResponse
from tests.rmf_web_models.rmf_api.fleet_state import FleetState
from tests.rmf_web_models.rmf_api.simple_response import SimpleResponse
from tests.rmf_web_models.rmf_api.task_log import TaskEventLog
from tests.rmf_web_models.rmf_api.task_request import TaskRequest
from tests.rmf_web_models.rmf_api.task_state import TaskState
from tests.rmf_web_models.ros_pydantic.rmf_building_map_msgs.msg import BuildingMap, Door, Lift
from tests.rmf_web_models.ros_pydantic.rmf_door_msgs.msg import DoorState
from tests.rmf_web_models.ros_pydantic.rmf_lift_msgs.msg import LiftState

SCENARIOS = list_scenarios()


def api(scenario: str = "nominal") -> InProcessApi:
    return InProcessApi(FakeRmfServer.from_scenario(scenario))


def test_bundled_scenarios():
    assert set(SCENARIOS) == {
        "nominal",
        "stuck_at_door",
        "lift_out_of_service",
        "low_battery",
        "queue_backlog",
        "failed_task",
        "conflicting_requests",
    }


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_responses_validate_against_rmf_web_models(scenario):
    """Every endpoint's JSON parses with the real api-server pydantic models (vendored)."""
    client = api(scenario)
    assert isinstance(client.get("/time"), int)
    BuildingMap.model_validate(client.get("/building_map"))
    fleets = client.get("/fleets")
    assert sum(len(FleetState.model_validate(f).robots) for f in fleets) == 8
    for door in client.get("/doors"):
        Door.model_validate(door)
        DoorState.model_validate(client.get(f"/doors/{door['name']}/state"))
    for lift in client.get("/lifts"):
        Lift.model_validate(lift)
        LiftState.model_validate(client.get(f"/lifts/{lift['name']}/state"))
    for alert in client.get("/alerts/unresponded_requests"):
        AlertRequest.model_validate(alert)
    for task in client.get("/tasks"):
        TaskState.model_validate(task)
        task_id = task["booking"]["id"]
        TaskRequest.model_validate(client.get(f"/tasks/{task_id}/request"))
        TaskEventLog.model_validate(client.get(f"/tasks/{task_id}/log", {"between": "-3600000"}))


def test_writes_validate_against_rmf_web_models():
    client = api()
    dispatched = client.post(
        "/tasks/dispatch_task",
        {
            "type": "dispatch_task_request",
            "request": {"category": "patrol", "description": {"places": ["dock_1"], "rounds": 1}},
        },
    )
    TaskDispatchResponse.model_validate(dispatched)
    task_id = dispatched["state"]["booking"]["id"]
    canceled = client.post(
        "/tasks/cancel_task", {"type": "cancel_task_request", "task_id": task_id}
    )
    SimpleResponse.model_validate(canceled)
    assert client.get(f"/tasks/{task_id}/state")["status"] == "canceled"


def test_relative_times():
    assert duration_ms("-12m") == -720_000
    assert duration_ms("90s") == 90_000
    assert duration_ms("1h") == 3_600_000
    with pytest.raises(ValueError):
        duration_ms("soon")
    state = load_scenario("stuck_at_door")
    tug_2_log = state.task_logs["delivery.dispatch-204"]["phases"]["1"]["log"]
    assert tug_2_log[-1]["unix_millis_time"] == state.now_ms - 8 * 60_000


def test_task_query_filters_and_pagination():
    client = api("queue_backlog")
    queued = client.get("/tasks", {"status": "queued"})
    assert len(queued) == 10
    assert len(client.get("/tasks", {"status": "queued", "limit": "3", "offset": "8"})) == 2
    assert [t["booking"]["id"] for t in client.get("/tasks", {"assigned_to": "tote_2"})] == [
        "patrol.dispatch-520"
    ]
    assert client.get("/tasks", {"requester": "nobody"}) == []


def test_task_log_default_window_is_60_seconds():
    client = api("stuck_at_door")
    assert client.get("/tasks/delivery.dispatch-204/log")["phases"]["1"]["log"] == []
    full = client.get("/tasks/delivery.dispatch-204/log", {"between": "-3600000"})
    assert len(full["phases"]["1"]["log"]) == 5


def test_unknown_paths_and_ids_404():
    client = api()
    for path in ["/tasks/nope/state", "/fleets/nope/state", "/doors/nope/state", "/admin/users"]:
        with pytest.raises(ApiError) as exc:
            client.get(path)
        assert exc.value.status == 404


def test_dispatch_rejects_unknown_places_with_400():
    with pytest.raises(ApiError) as exc:
        api().post(
            "/tasks/dispatch_task",
            {
                "type": "dispatch_task_request",
                "request": {"category": "patrol", "description": {"places": ["roof"]}},
            },
        )
    assert exc.value.status == 400
    assert exc.value.body["success"] is False


def test_cancel_frees_the_robot():
    client = api("nominal")
    client.post(
        "/tasks/cancel_task", {"type": "cancel_task_request", "task_id": "delivery.dispatch-101"}
    )
    tote_1 = client.get("/fleets/tote_fleet/state")["robots"]["tote_1"]
    assert tote_1["status"] == "idle" and tote_1["task_id"] == ""
    again = client.post(
        "/tasks/cancel_task", {"type": "cancel_task_request", "task_id": "delivery.dispatch-101"}
    )
    assert again["success"] is False


def test_http_mode_matches_in_process():
    server = FakeRmfServer.from_scenario("lift_out_of_service")
    http = HttpApi("http://testserver", token="t", client=TestClient(create_app(server)))
    local = InProcessApi(server)
    for path in ["/fleets", "/lifts/lift_1/state", "/tasks", "/alerts/unresponded_requests"]:
        assert http.get(path) == local.get(path)
    with pytest.raises(ApiError) as exc:
        http.get("/tasks/missing/state")
    assert exc.value.status == 404
