"""Two-phase commit for write tools: a dry run issues a token, the commit must redeem it.

A token is bound to the tool name and the exact normalized arguments (HMAC over canonical JSON),
is single-use, and expires after ``ttl_s``. A commit with any argument changed is rejected.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .spec import ToolError


def _digest(tool: str, args: dict[str, Any]) -> str:
    canonical = json.dumps({"tool": tool, "args": args}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class _Issued:
    digest: str
    issued_at: float
    used: bool = False


class ConfirmationGate:
    def __init__(
        self,
        ttl_s: float = 120.0,
        secret: bytes | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_s = ttl_s
        self._secret = secret or secrets.token_bytes(32)
        self._clock = clock
        self._issued: dict[str, _Issued] = {}
        self._counter = 0
        self._lock = threading.Lock()

    def issue(self, tool: str, args: dict[str, Any]) -> str:
        with self._lock:
            self._counter += 1
            digest = _digest(tool, args)
            mac = hmac.new(self._secret, f"{self._counter}:{digest}".encode(), hashlib.sha256)
            token = "ct_" + mac.hexdigest()[:24]
            self._issued[token] = _Issued(digest=digest, issued_at=self._clock())
            return token

    def redeem(self, tool: str, args: dict[str, Any], token: str) -> None:
        with self._lock:
            issued = self._issued.get(token)
            if issued is None:
                raise ToolError("token_invalid", "Unknown confirmation token. Run a dry run first.")
            if issued.used:
                raise ToolError("token_used", "This confirmation token was already used.")
            if self._clock() - issued.issued_at > self.ttl_s:
                raise ToolError("token_expired", "Confirmation token expired. Run a new dry run.")
            if not hmac.compare_digest(issued.digest, _digest(tool, args)):
                raise ToolError(
                    "token_mismatch",
                    "Arguments differ from the dry run this token was issued for.",
                )
            issued.used = True
