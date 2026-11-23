"""Optional LLM-judge hook (off by default). Scores never change deterministic pass/fail; they
are reported in a separate column for cases that define a ``rubric``.
"""

from __future__ import annotations

import json
from typing import Any, Protocol

JUDGE_PROMPT = """\
You grade an operations assistant's answer against a rubric. Reply with JSON only:
{{"score": 0 | 1 | 2, "reason": "<one sentence>"}}
2 = fully meets the rubric, 1 = partly, 0 = does not.

Rubric: {rubric}

Operator question: {question}

Assistant answer:
{answer}
"""


class Judge(Protocol):
    name: str

    def grade(self, question: str, answer: str, rubric: str) -> dict[str, Any]: ...


class CohereJudge:
    def __init__(self, model: str, client: Any = None) -> None:
        import os

        import cohere

        self.name = f"cohere:{model}"
        self.model = model
        self.client = client or cohere.ClientV2(api_key=os.environ["CO_API_KEY"])

    def grade(self, question: str, answer: str, rubric: str) -> dict[str, Any]:
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(rubric=rubric, question=question, answer=answer),
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        text = "".join(c.text for c in response.message.content if c.type == "text")
        verdict = json.loads(text)
        score = int(verdict["score"])
        if score not in (0, 1, 2):
            raise ValueError(f"judge score out of range: {score}")
        return {"score": score, "reason": str(verdict.get("reason", ""))}
