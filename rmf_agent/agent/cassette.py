"""Record model responses to a cassette file and replay them without a model or API key.

A cassette is JSON: a header saying which model produced it, and the list of model turns in
order. Each turn stores a digest of the request messages it answered; on replay a differing
digest is reported as drift (the conversation went somewhere the recording did not).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .model import Message, ModelClient, ModelTurn


def request_digest(messages: list[Message]) -> str:
    return hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()[:16]


class CassetteExhausted(RuntimeError):
    pass


class RecordingModel:
    def __init__(self, inner: ModelClient, path: str | Path, meta: dict[str, Any] | None = None):
        self.inner = inner
        self.name = inner.name
        self.is_fake = inner.is_fake
        self.path = Path(path)
        self.meta = {"model": inner.name, "fake_model": inner.is_fake, **(meta or {})}
        self.turns: list[dict[str, Any]] = []

    def chat(
        self, messages: list[Message], tools: list[dict[str, Any]], **kwargs: Any
    ) -> ModelTurn:
        turn = self.inner.chat(messages, tools, **kwargs)
        self.turns.append({"request_digest": request_digest(messages), "turn": turn.to_dict()})
        return turn

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {**self.meta, "turns": self.turns}
        self.path.write_text(json.dumps(payload, indent=1) + "\n")


class ReplayModel:
    def __init__(self, path: str | Path) -> None:
        data = json.loads(Path(path).read_text())
        self.name = f"replay:{data['model']}"
        self.is_fake = bool(data.get("fake_model", False))
        self.meta = {k: v for k, v in data.items() if k != "turns"}
        self._turns = list(data["turns"])
        self.drift: list[int] = []
        self._index = 0

    def chat(self, messages: list[Message], tools: list[dict[str, Any]], **_: Any) -> ModelTurn:
        if self._index >= len(self._turns):
            raise CassetteExhausted(f"cassette has only {len(self._turns)} turns")
        recorded = self._turns[self._index]
        if recorded["request_digest"] != request_digest(messages):
            self.drift.append(self._index)
        self._index += 1
        return ModelTurn.from_dict(recorded["turn"])
