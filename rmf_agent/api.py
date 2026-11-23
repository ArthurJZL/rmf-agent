"""Clients for the rmf-web api-server: over HTTP (real server or the fake), or in-process."""

from __future__ import annotations

from typing import Any, Protocol

import httpx

from .sim.world import FakeRmfServer


class ApiError(Exception):
    def __init__(self, status: int, path: str, body: Any) -> None:
        super().__init__(f"HTTP {status} for {path}")
        self.status = status
        self.path = path
        self.body = body


class RmfApi(Protocol):
    """The two verbs the tools need. ``params`` values are strings, as on the wire."""

    def get(self, path: str, params: dict[str, str] | None = None) -> Any: ...

    def post(self, path: str, body: dict[str, Any]) -> Any: ...


class InProcessApi:
    """Calls the fake api-server directly, without a socket. Used by tests and the eval."""

    def __init__(self, server: FakeRmfServer) -> None:
        self.server = server

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        return self._check(path, *self.server.handle("GET", path, params))

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._check(path, *self.server.handle("POST", path, body=body))

    @staticmethod
    def _check(path: str, status: int, body: Any) -> Any:
        if status >= 400:
            raise ApiError(status, path, body)
        return body


class HttpApi:
    """httpx client for a real rmf-web api-server (or ``rmf-agent sim``)."""

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        timeout_s: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = client or httpx.Client(base_url=base_url, timeout=timeout_s)
        self.client.headers.update(headers)

    def get(self, path: str, params: dict[str, str] | None = None) -> Any:
        return self._check(self.client.get(path, params=params))

    def post(self, path: str, body: dict[str, Any]) -> Any:
        return self._check(self.client.post(path, json=body))

    @staticmethod
    def _check(response: httpx.Response) -> Any:
        try:
            body = response.json()
        except ValueError:
            body = response.text
        if response.status_code >= 400:
            raise ApiError(response.status_code, response.request.url.path, body)
        return body
