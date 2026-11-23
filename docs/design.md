# rmf-agent design

This adapts an earlier design draft ("rmf-mcp": an MCP server over the rmf-web api-server) to what
was built: the same tool and safety design, plus a Cohere agent loop and an offline evaluation
harness. Where the draft and the code differ, the code wins; differences are listed at the end.

## 1. Problem and scope

An Open-RMF operator's shift questions ("which robot is stuck and why", "what is late", "write
the handover") need a join across fleet state, task state, task event logs, door and lift state
and alerts. The dashboard shows these in separate panels. An agent with a small set of
well-shaped tools can do the join in one turn and cite what it used.

Goals:

1. **Read first.** Diagnostic and status tools that return evidence (ids, timestamps, log lines).
2. **Task-shaped tools, not endpoint mirrors.** One tool per operator question, compact JSON.
3. **Guarded writes.** `dispatch_task` / `cancel_task` off by default; dry run, confirmation
   token, allowlist, audit log.
4. **Measurable.** A reproducible eval that reports accuracy, safety and latency, with a hard
   gate of zero safety violations.

Non-goals: replacing the dashboard; low-level control (door/lift requests, e-stops,
decommissioning, admin) even though the api-server exposes it; talking to ROS directly; any claim
of production readiness.

Data constraint: only open-source Open-RMF concepts and simulated data. The world in
`rmf_agent/sim/data/world.yaml` is invented. No employer code, customer names, site layouts or
operational data.

## 2. Components

| Component | Module | Notes |
|---|---|---|
| Fake api-server | `rmf_agent/sim` | Scenario YAML -> api-server JSON shapes; routes for the endpoints in section 3; in-process or FastAPI over HTTP |
| API clients | `rmf_agent/api.py` | `HttpApi` (httpx, Bearer token) and `InProcessApi`; same two-verb interface |
| Tools | `rmf_agent/tools` | `ReadTools`, `WriteTools`, `diagnose.py` rules, `ConfirmationGate`, `AuditLog`, `Toolbox` |
| Agent | `rmf_agent/agent` | `CohereModel`, loop, policy, prompts, trace, cassettes, scripted fake model |
| MCP server | `rmf_agent/mcp_server.py` | FastMCP; every call goes through `Toolbox.call` |
| Eval | `rmf_agent/evals`, `eval/` | Case schema, scoring, runner, Markdown/JSON report, optional judge |

## 3. api-server surface implemented by the fake (and used by the tools)

Checked against rmf-web `main` at `c1b13dc` (`packages/api-server/api_server/routes`, `models`):

| Domain | Endpoints |
|---|---|
| Time | `GET /time` (sim time when `RMF_SERVER_USE_SIM_TIME=true`) |
| Fleets | `GET /fleets` -> `list[FleetState]`, `GET /fleets/{name}/state` |
| Tasks | `GET /tasks` (`task_id, category, requester, assigned_to, status`, `limit, offset`), `GET /tasks/requests?task_ids=`, `GET /tasks/{id}/state`, `GET /tasks/{id}/request`, `GET /tasks/{id}/log?between=` (default window: last 60 s) |
| Writes | `POST /tasks/dispatch_task` (`{"type": "dispatch_task_request", "request": TaskRequest}`), `POST /tasks/cancel_task` (`{"type": "cancel_task_request", "task_id", "labels"}`) |
| Building | `GET /building_map`, `GET /doors`, `GET /doors/{name}/state`, `GET /lifts`, `GET /lifts/{name}/state` |
| Alerts | `GET /alerts/unresponded_requests`, `GET /alerts/requests/task/{task_id}` |

Enum values come from rmf-web's generated models: door modes `closed/moving/open/offline/unknown`
(0..4); lift modes `unknown/human/agv/fire/offline/emergency` (0..5); task statuses `queued,
standby, underway, delayed, blocked, error, failed, canceled, killed, skipped, completed`.

`tests/test_sim.py` validates every fake response against the vendored rmf-web pydantic models,
which is the guarantee that "point it at a real api-server" is a configuration change for these
shapes. It does not guarantee that real log text or real fleet behavior match the scenarios.

## 4. Tools

See the README's tool table. Design decisions:

- **Curated, not generated from OpenAPI.** "Why is tug_2 stuck" needs 4-6 endpoint calls and a
  join. Generated tools would push that planning onto the model every time, expose about 75
  operations including admin and door/lift overrides, and return base64 floor plans and raw
  epoch-millisecond timestamps.
- **Rules find, the model explains.** `diagnose.py` turns state into findings with codes and
  evidence. The model's job is to explain and prioritize, not to infer "stuck" from raw numbers.
- **Errors are data.** Tool failures come back as `{"ok": false, "error": <code>, ...}` so the
  model can recover (for example, call `find_places` after `unknown places`).
- **Untrusted text is marked.** Tools whose outputs contain fleet-system strings (logs, alerts,
  issue details) add a `data_notice` field; the system prompt says the same. This is guidance for
  the model only; the write policy does not rely on it.

## 5. Agent loop

Per operator turn (`agent/loop.py`):

1. `screen_request` may refuse immediately (no model call).
2. Plan mode, when requested or when the request looks multi-part (handover, "summarize",
   several questions): one model call with `tool_choice="NONE"` writes a numbered plan, which
   stays in the conversation; a controller message asks the model to execute and revise it.
3. ReAct loop up to `max_steps` model calls. Each model call and tool call runs with a timeout.
   Tool calls to write tools pass through `authorize_write` first. Tool results go back as Cohere
   v2 `tool` messages with `tool_call_id`.
4. Citations (`[id]`) in the final answer are checked against the text of all tool results in the
   session; unknown ids are reported as ungrounded.

`CohereModel` calls `cohere.ClientV2.chat(model, messages, tools, temperature=0, tool_choice,
request_options={"timeout_in_seconds": ...})` and reads `message.tool_plan`, `message.tool_calls`,
`message.content` and `usage.tokens`. The request/response shapes were checked against SDK 7.3.2
with a mocked HTTP transport (`tests/test_model.py`). Default model: `command-a-plus-05-2026`.

## 6. Write safety

Layers, outermost first:

1. Write tools are not offered unless writes are enabled.
2. Request screen (door/lift control, bulk destructive actions, safety bypass, credentials).
3. Write policy: operator intent for that action in the session (cancel: the task id named by the
   operator); commits only in a turn where the operator confirmed, with a token issued by a dry
   run in an earlier turn.
4. Confirmation gate: HMAC over tool + canonical arguments, single use, 120 s TTL.
5. Tool rules: category allowlist, known places, available robot, cap of 3 open agent tasks,
   cancel only own non-terminal tasks.
6. api-server permissions for the rmf-agent user (deployment responsibility).

Every call is written to the JSONL audit log with tokens hashed.

The draft relied on MCP client confirmation (`destructiveHint`) plus the token. The Cohere agent
has no client UI, so the "operator saw the plan" requirement is enforced by the turn rule in
layer 3. On the MCP path, layer 3 does not apply (there is no operator transcript on the server
side); the MCP client's own confirmation of destructive tools plus layers 1, 4 and 5 apply.

## 7. Evaluation

- **Fixtures, not a full sim.** The draft planned to seed a headless `rmf_demos` Gazebo stack and
  inject faults. This build uses the scenario-driven fake instead: it is deterministic, runs in CI
  in seconds and needs no ROS. The cost is realism (no motion, fixed clock, invented log text).
- **Cases** in `eval/cases.yaml`; deterministic checks first; optional LLM judge never changes
  pass/fail.
- **Models.** `scripted-fake` (keyword rules; validates tools, policy and harness),
  `scripted-fake-adversarial` (obeys injected text, skips dry runs; shows the policy holds),
  `--live` (Cohere), `--model replay` (cassettes). Confirmation tokens use a per-case secret during
  eval so cassettes replay without drift; replay reports drift if the conversation diverges.
- **Gate.** CI fails on any safety violation and on less than 100% success with the scripted fake.

## 8. Differences from the draft

| Draft | Built | Why |
|---|---|---|
| MCP server only, Claude as client | Cohere agent loop + thin MCP server over the same toolbox | Target was a Cohere-based agent with its own observability |
| socket.io cache for fleet/door/lift state | REST on every call | Simpler and deterministic; caching is a performance concern for a real deployment |
| `rmf_demos` office/hotel worlds in Docker, harness fault injection | Invented two-level world, seven scenario files | Runs anywhere, no ROS; real-stack validation still to do |
| `get_task_timeline`, `wait_for_task_change`, `get_server_info`, resources, prompts | `task_status` + `task_log`; no waits, resources or prompts | Kept the surface small; handover is a plan-mode request instead of an MCP prompt |
| Rate limit on writes | Cap on open agent-created tasks | Enough for the scenarios; rate limiting needs wall-clock behavior the fake does not model |

## 9. Open items

- Run the live eval with a Cohere key and publish the numbers next to the offline ones.
- Run against a real rmf-web api-server with `rmf_demos`, compare real task log wording with the
  door/lift rules, and add an OpenAPI drift test against `/openapi.json`.
- Add socket.io subscriptions if REST polling is too slow at fleet scale.
