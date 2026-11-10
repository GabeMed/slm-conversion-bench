"""`bench replay --resume-from <replay run_id>`: a replay that stopped on an infrastructure failure is
resumed as a new replay that copies the invocations it completed and sends only the others. The
model is the fake OpenAI-compatible endpoint of test_workers; agent environment only."""
import json

import pytest
import yaml

pytest.importorskip("langchain_openai")

from bench import cli, paths  # noqa: E402
from bench.agent import replay, runner  # noqa: E402
from bench.agent.resume import USAGE_FIELDS, _ended  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.data import DataError  # noqa: E402
from bench.judge.base import calls_of, invocations, require_done  # noqa: E402
from test_workers import QUESTIONS, FakeChess, make_worker_repo, marker  # noqa: E402


def identity(call):
    return call["question_id"], call["call_site"], call["invocation_key"]


def lines_of(run_dir):
    return [line for line in (run_dir / "calls.jsonl").read_text().splitlines() if line.strip()]


def read(run_dir):
    return json.loads((run_dir / "manifest.json").read_text()), read_calls(run_dir / "calls.jsonl")


def usage(calls):
    return {name: sum((c["usage"] or {}).get(name) or 0 for c in calls) for name in USAGE_FIELDS}


def added(*totals):
    return {name: sum(t[name] for t in totals) for name in USAGE_FIELDS}


@pytest.fixture
def failed(tmp_path, monkeypatch):
    """A done B0 source on six questions, and a replay of it whose harness fails on question 4 (a
    401 on every prompt that carries it): 1-3 replayed, 4's agent calls answered and its tool calls
    refused or never sent, 5-6 never started. The endpoint then answers everything again."""
    server = FakeChess()
    config_path = make_worker_repo(tmp_path, monkeypatch, server)
    source = runner.run_agent(str(config_path), "B0", "train").name
    server.refuse = marker("4")
    failed_dir = replay.replay(str(config_path), source, engine="production_llm")
    server.refuse, server.prompts[:] = None, []
    return server, config_path, source, failed_dir


def test_a_failed_replay_is_resumed_into_the_replay_one_run_gives(failed):
    server, config_path, source, failed_dir = failed
    old, old_calls = read(failed_dir)
    assert old["status"] == "failed" and list(old["harness_errors"]) == ["4"]
    four = [c for c in old_calls if c["question_id"] == "4"]
    assert any(c["parsed_ok"] for c in four) and any(c["error"] for c in four)  # partly answered

    run_dir = replay.replay(str(config_path), source, engine="production_llm", resume_from=failed_dir.name)
    manifest, calls = read(run_dir)
    assert manifest["status"] == "done" and manifest["n_questions_replayed"] == len(QUESTIONS), manifest
    assert manifest["model_failures"] == {} and validate_calls(calls) == [] and manifest["c1_errors"] == 0

    # copied per invocation: question 4's answered invocations included, its refused one not
    copied = {identity(c) for c in calls if c["run_id"] == failed_dir.name}
    sent = {identity(c) for c in calls if c["run_id"] == manifest["run_id"]}
    assert not copied & sent and {q for q, _, _ in copied} == {"1", "2", "3", "4"}
    assert {q for q, _, _ in sent} == {"4", "5", "6"}
    assert any(c["error"] and c["question_id"] == "4" and identity(c) in sent for c in old_calls)
    assert len(server.prompts) == len(sent)  # only the invocations left reached the model
    kept = [line for line in lines_of(failed_dir) if identity(json.loads(line)) in copied]
    assert lines_of(run_dir)[:len(kept)] == kept  # unchanged, byte for byte

    # the same invocations, attempt by attempt, as a replay in one go
    once, once_calls = read(replay.replay(str(config_path), source, engine="production_llm"))
    assert sorted(map(identity, calls)) == sorted(map(identity, once_calls))
    assert set(invocations(calls)) == set(invocations(once_calls))
    for key in ("n_invocations", "question_ids", "source_calls_sha256", "call_sites", "n_calls"):
        assert manifest[key] == once[key]

    # the cost: what was copied plus what was sent is the replay's; what was discarded is the rest of the failed one
    resumed = manifest["resumed_from"]
    assert resumed["run_id"] == failed_dir.name and resumed["status"] == "failed"
    assert resumed["copied"]["n_calls"] + resumed["discarded"]["n_calls"] == old["n_calls"] == len(old_calls)
    assert added(resumed["copied"]["usage"], resumed["discarded"]["usage"]) == usage(old_calls)
    assert resumed["copied"]["n_invocations"] == len(copied) and resumed["discarded"]["question_ids"] == ["4"]
    new_calls = [c for c in calls if c["run_id"] == manifest["run_id"]]
    assert added(resumed["copied"]["usage"], usage(new_calls)) == usage(calls) == usage(once_calls)
    assert manifest["n_calls"] == resumed["copied"]["n_calls"] + len(new_calls)
    first, this = manifest["sessions"]
    assert (first["started_at"], first["finished_at"]) == (old["started_at"], old["finished_at"])
    assert (this["started_at"], this["finished_at"], this["workers"]) == (manifest["started_at"], manifest["finished_at"], 1)

    # read as any done replay is
    assert require_done(manifest["run_id"], type="replay")["source_run_id"] == source
    assert len(calls_of(manifest["run_id"])) == len(calls)


def test_a_resume_in_workers_gives_the_same_replay(failed):
    server, config_path, source, failed_dir = failed
    one, one_calls = read(replay.replay(str(config_path), source, engine="production_llm", resume_from=failed_dir.name))
    server.prompts[:] = []
    run_dir = replay.replay(str(config_path), source, engine="production_llm", workers=3, resume_from=failed_dir.name)
    three, three_calls = read(run_dir)
    assert three["status"] == "done" and three["sessions"][-1]["workers"] == 3
    assert sorted(map(identity, three_calls)) == sorted(map(identity, one_calls))
    assert three["resumed_from"] == one["resumed_from"]
    order = [c["question_id"] for c in three_calls]
    assert order == sorted(order, key=int)  # merged in the order of the questions, copied lines first in each
    assert len(server.prompts) == len([c for c in three_calls if c["run_id"] == three["run_id"]])
    assert not list(run_dir.glob("calls.w*")) and not list(run_dir.glob("outcome.w*")) and validate_calls(three_calls) == []


def test_a_resumed_replay_that_fails_again_can_be_resumed(failed):
    server, config_path, source, failed_dir = failed
    server.refuse = marker("5")
    again = replay.replay(str(config_path), source, engine="production_llm", resume_from=failed_dir.name)
    second, _ = read(again)
    assert second["status"] == "failed" and list(second["harness_errors"]) == ["5"]
    server.refuse = None
    manifest, calls = read(replay.replay(str(config_path), source, engine="production_llm", resume_from=again.name))
    once, once_calls = read(replay.replay(str(config_path), source, engine="production_llm"))
    assert manifest["status"] == "done" and sorted(map(identity, calls)) == sorted(map(identity, once_calls))
    assert len(manifest["sessions"]) == 3 and validate_calls(calls) == []


def runs_now():
    return sorted(p.name for p in paths.RUNS.iterdir())


def test_a_resume_is_refused_unless_it_is_the_same_replay_of_an_infrastructure_failure(failed, tmp_path):
    server, config_path, source, failed_dir = failed
    name = failed_dir.name
    before = runs_now()

    def refused(match, *args, config=config_path, **kwargs):
        with pytest.raises(DataError, match=match):
            replay.replay(str(config), *(args or (source,)), **{"engine": "production_llm", "resume_from": name, **kwargs})
        assert runs_now() == before  # a refusal leaves no run

    refused("no run", resume_from="replay-nothing")
    refused("is a 'agent' run", resume_from=source)
    refused("question_ids", ids=["1", "2", "3", "4"])
    refused("call_sites", call_sites=["revise"])
    refused("engine", engine=None, arm="B0")
    other = yaml.safe_load(config_path.read_text())
    other["eval"]["timeout_s"] += 1  # outside the replay's policy: only the configuration's hash sees it
    other_path = tmp_path / "other.yaml"
    other_path.write_text(yaml.safe_dump(other))
    refused("config_sha256", config=other_path)

    source_calls = paths.RUNS / source / "calls.jsonl"
    text = source_calls.read_text()
    source_calls.write_text(text + "\n")  # the same C1, other bytes
    refused("source_calls_sha256")
    source_calls.write_text(text)

    manifest_path = failed_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest_path.write_text(json.dumps({**manifest, "few_shot_k": 3}))
    refused("few_shot_k")
    manifest_path.write_text(json.dumps({**manifest, "harness_errors": {}}))
    refused("without a harness error")
    manifest_path.write_text(json.dumps(manifest))

    other_source = runner.run_agent(str(config_path), "B0", "train").name
    before = runs_now()
    refused("source_run_id", other_source)

    done = replay.replay(str(config_path), source, engine="production_llm", resume_from=name)
    before = runs_now()
    refused("'done': only a failed or interrupted replay is resumed", resume_from=done.name)


def test_the_cli_resumes_a_replay(failed, capsys):
    server, config_path, source, failed_dir = failed
    assert cli.main(["replay", source, "--config", str(config_path), "--engine", "production_llm",
                     "--resume-from", failed_dir.name]) == 0
    run_dir = paths.RUNS / capsys.readouterr().out.strip().splitlines()[-1].rsplit("/", 1)[-1]
    assert json.loads((run_dir / "manifest.json").read_text())["resumed_from"]["run_id"] == failed_dir.name


def attempt(n, site="select_tables", text="{}", parsed=True, error=None):
    return {"attempt": n, "call_site": site, "response_text": text, "parsed_ok": parsed, "error": error}


@pytest.mark.parametrize("attempts,complete", [
    ([attempt(1)], True),                                                              # answered
    ([attempt(1, text=None, parsed=False, error="RateLimitError: Error code: 429 - slow down")], False),
    ([attempt(1, text=None, parsed=False, error="APIConnectionError: Connection error.")], False),
    ([attempt(1, text=None, parsed=False, error="AuthenticationError: Error code: 401 - no")], False),
    ([attempt(1, text=None, parsed=False, error="APIStatusError: Error code: 413 - too large")], True),  # the model's
    ([attempt(1, text=None, parsed=False,
              error="BadRequestError: Error code: 400 - {'message': 'prompt is too long: 300000 tokens'}")], True),
    ([attempt(1, text=None, parsed=False, error="BadRequestError: Error code: 400 - credit balance too low")], False),
    ([attempt(1, text=None, parsed=False, error="the engine answered list content, not text")], False),
    ([attempt(1, text="x", parsed=False, error="OutputParserException: no")], False),   # a parse retry left
    ([attempt(1, text="x", parsed=False, error="OutputParserException: no"),
      attempt(2, text=None, parsed=False, error="RateLimitError: Error code: 429"),
      attempt(3, text="", parsed=False, error="empty output")], True),                  # parse budget (2) spent
    ([attempt(1, text="x", parsed=False, error="ValueError: the parser raised")], True),  # not retried
    ([attempt(1, site="agent_schema_selector", text="?", parsed=False, error="ValueError: no tool")], True),
])
def test_which_invocations_are_complete(attempts, complete):
    assert _ended(attempts, parse_max_attempts=2) is complete


def test_the_cli_passes_resume_from(monkeypatch):
    seen = {}
    monkeypatch.setattr(replay, "replay", lambda config, source, **kwargs: seen.update(kwargs) or "r")
    assert cli.main(["replay", "agent-B0-calib-x", "--engine", "cheap_alt"]) == 0 and seen["resume_from"] is None
    assert cli.main(["replay", "agent-B0-calib-x", "--engine", "cheap_alt", "--resume-from", "replay-x"]) == 0
    assert seen["resume_from"] == "replay-x"
