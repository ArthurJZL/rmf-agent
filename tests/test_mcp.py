import asyncio
import json

from fastmcp import Client

from rmf_agent.config import Settings
from rmf_agent.mcp_server import create_server
from rmf_agent.tools import build_toolbox
from rmf_agent.tools.audit import AuditLog


def run(coro):
    return asyncio.run(coro)


async def _list(settings: Settings):
    async with Client(create_server(settings)) as client:
        return await client.list_tools()


def test_read_only_server_lists_only_read_tools():
    tools = run(_list(Settings(audit_path=None)))
    assert len(tools) == 10
    assert all(t.annotations.read_only_hint for t in tools)


def test_write_tools_are_marked_destructive():
    tools = {t.name: t for t in run(_list(Settings(audit_path=None, allow_writes=True)))}
    assert tools["dispatch_task"].annotations.destructive_hint is True
    assert "confirm_token" in tools["dispatch_task"].input_schema["properties"]


def test_mcp_calls_go_through_the_toolbox_audit():
    settings = Settings(scenario="stuck_at_door", audit_path=None, allow_writes=True)
    toolbox = build_toolbox(settings, audit=AuditLog(None))

    async def calls():
        async with Client(create_server(settings, toolbox)) as client:
            stuck = await client.call_tool("find_stuck_robots", {})
            denied = await client.call_tool(
                "cancel_task", {"task_id": "delivery.dispatch-204", "reason": "test"}
            )
            return json.loads(stuck.content[0].text), json.loads(denied.content[0].text)

    stuck, denied = run(calls())
    assert stuck["stuck"][0]["robot"] == "tug_2"
    assert denied["error"] == "not_owned"
    assert [r["decision"] for r in toolbox.audit.records] == ["ok", "rejected:not_owned"]
