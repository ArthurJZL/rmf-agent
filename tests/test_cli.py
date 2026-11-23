import json

import pytest

from rmf_agent.cli import main


def run(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    return exc.value.code, capsys.readouterr()


def test_ask_offline(capsys, tmp_path):
    trace = tmp_path / "trace.jsonl"
    code, out = run(
        [
            "ask",
            "Which robot is stuck and why?",
            "--scenario",
            "stuck_at_door",
            "--offline",
            "--trace",
            str(trace),
            "--audit-path",
            str(tmp_path / "a.jsonl"),
        ],
        capsys,
    )
    assert code == 0
    assert "fire_door_a" in out.out and "SCRIPTED FAKE" in out.out
    events = [json.loads(line)["event"] for line in trace.read_text().splitlines()]
    assert events[0] == "turn_start" and events[-1] == "turn_end"


def test_ask_without_key_fails_cleanly(capsys, monkeypatch):
    monkeypatch.delenv("CO_API_KEY", raising=False)
    code, out = run(["ask", "status?"], capsys)
    assert code == 2 and "CO_API_KEY" in out.err


def test_eval_subset_writes_report(capsys, tmp_path):
    code, _ = run(
        [
            "eval",
            "--case",
            "door-stuck-why",
            "--case",
            "unsafe-open-doors",
            "--out",
            str(tmp_path),
            "--fail-under",
            "1.0",
        ],
        capsys,
    )
    assert code == 0
    report = (tmp_path / "report.md").read_text()
    assert "SCRIPTED FAKE MODEL" in report
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["meta"]["fake_model"] is True and results["summary"]["cases"] == 2


def test_eval_replays_committed_cassettes(capsys, tmp_path):
    code, _ = run(
        [
            "eval",
            "--model",
            "replay",
            "--replay-dir",
            "eval/cassettes/scripted-fake",
            "--out",
            str(tmp_path),
            "--fail-under",
            "1.0",
        ],
        capsys,
    )
    assert code == 0
