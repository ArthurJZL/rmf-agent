"""Append-only JSONL audit log of every tool call."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any


def redact(args: dict[str, Any]) -> dict[str, Any]:
    """Replace confirmation tokens with a short hash so the log cannot be replayed."""
    out = dict(args)
    token = out.get("confirm_token")
    if token:
        out["confirm_token"] = "sha256:" + hashlib.sha256(token.encode()).hexdigest()[:12]
    return out


class AuditLog:
    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def write(self, **record: Any) -> None:
        entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **record}
        if "args" in entry:
            entry["args"] = redact(entry["args"])
        with self._lock:
            self.records.append(entry)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry, sort_keys=True) + "\n")

    def writes(self) -> list[dict[str, Any]]:
        return [r for r in self.records if r.get("write")]
