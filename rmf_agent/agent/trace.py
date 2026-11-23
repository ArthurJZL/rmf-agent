"""Structured JSONL trace of an agent run: every model call, tool call, policy decision."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


class Tracer:
    def __init__(self, path: str | Path | None = None, run_id: str | None = None) -> None:
        self.path = Path(path) if path else None
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.events: list[dict[str, Any]] = []
        self._seq = 0
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def event(self, kind: str, **fields: Any) -> dict[str, Any]:
        self._seq += 1
        record = {
            "run_id": self.run_id,
            "seq": self._seq,
            "ts": round(time.time(), 3),
            "event": kind,
            **fields,
        }
        self.events.append(record)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["event"] == kind]
