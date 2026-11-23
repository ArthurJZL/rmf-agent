"""Evaluation case schema (see eval/cases.yaml for the annotated file)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class ArgExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: str
    args: dict[str, Any]


class Expect(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tools: list[str | list[str]] = Field(default_factory=list)
    optional_tools: list[str] = Field(default_factory=list)
    forbid_tools: list[str] = Field(default_factory=list)
    args: list[ArgExpectation] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    facts_any: list[list[str]] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    refusal: bool = False
    writes: Literal["none", "dry_run", "commit"] = "none"
    plan: bool = False

    def required_groups(self) -> list[set[str]]:
        return [set(t) if isinstance(t, list) else {t} for t in self.tools]

    def allowed_tools(self) -> set[str]:
        return set().union(*self.required_groups(), self.optional_tools)


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    scenario: str
    turns: list[str]
    allow_writes: bool = False
    mode: Literal["react", "plan", "auto"] = "auto"
    expect: Expect = Field(default_factory=Expect)
    rubric: str | None = None  # only used by the optional LLM judge


def load_cases(path: str | Path) -> list[Case]:
    raw = yaml.safe_load(Path(path).read_text())
    cases = [Case.model_validate(item) for item in raw]
    ids = [c.id for c in cases]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate case ids: {sorted(duplicates)}")
    return cases
