"""Serve the fake api-server over HTTP so the real HTTP client code path can be exercised.

``rmf-agent sim --scenario stuck_at_door --port 8000`` starts it; point ``--api-url`` at it.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .world import FakeRmfServer


def create_app(server: FakeRmfServer) -> FastAPI:
    app = FastAPI(
        title="rmf-agent fake api-server",
        description=f"Scenario: {server.state.name}. Simulated data only.",
    )

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def dispatch(path: str, request: Request) -> JSONResponse:
        body: Any = None
        if request.method == "POST":
            body = await request.json()
        status, payload = server.handle(request.method, path, dict(request.query_params), body)
        return JSONResponse(payload, status_code=status)

    return app
