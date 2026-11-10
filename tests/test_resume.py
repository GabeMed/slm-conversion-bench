"""`bench run --resume-from <run_id>`: a run that stopped on an infrastructure failure is resumed as a
new run that copies the questions it completed and answers only the others. The model is the fake
OpenAI-compatible endpoint of test_workers; agent environment only."""
import json
from datetime import datetime

import pytest
import yaml

pytest.importorskip("langchain_openai")

from bench import cli, paths, preflight  # noqa: E402
from bench.agent import runner  # noqa: E402
from bench.contracts.calls import validate_calls  # noqa: E402
from bench.data import DataError  # noqa: E402
from bench.judge.base import calls_of, require_done  # noqa: E402
from synthetic import GOLD  # noqa: E402
from test_workers import QUESTIONS, FakeChess, make_worker_repo, marker, read_run  # noqa: E402


def lines_of(run_dir):
    return [line for line in (run_dir / "calls.jsonl").read_text().splitlines() if line.strip()]


def asked(server, questions):
    return {q for q in questions if any(marker(q) in prompt for prompt in server.prompts)}


@pytest.fixture
def failed(tmp_path, monkeypatch):
    """A run whose harness fails on question 4 (a 401): 1-3 answered, 4 failed, 5-6 never started.
    The endpoint then answers everything, as after the infrastructure is fixed."""
    server = FakeChess(refuse=marker("4"))
    config_path = make_worker_repo(tmp_path, monkeypatch, server)
    source_dir = runner.run_agent(str(config_path), "B0", "train")
    server.refuse, server.prompts[:] = None, []
    return server, config_path, source_dir


def test_a_failed_run_is_resumed_into_a_done_run_answering_only_the_questions_left(failed):
    server, config_path, source_dir = failed
    source, source_calls, source_predictions = read_run(source_dir)
    assert source["status"] == "failed" and list(source["harness_errors"]) == ["4"]
    assert "4" in source_predictions  # a prediction, but its harness failed: not an answer

    run_dir = runner.run_agent(str(config_path), "B0", "train", resume_from=source["run_id"])
    manifest, calls, predictions = read_run(run_dir)

    assert run_dir != source_dir and manifest["status"] == "done", manifest
    assert manifest["question_ids"] == QUESTIONS and list(predictions) == QUESTIONS
    assert set(predictions.values()) == {f" {GOLD} "}
    assert manifest["resumed_question_ids"] == ["1", "2", "3"]
    assert {q: predictions[q] for q in ("1", "2", "3")} == {q: source_predictions[q] for q in ("1", "2", "3")}
    # only the questions left reached the model
    assert asked(server, QUESTIONS) == {"4", "5", "6"}
    # the completed questions' lines, every attempt, byte for byte (their run_id is the source's)
    copied = [line for line in lines_of(source_dir) if json.loads(line)["question_id"] in ("1", "2", "3")]
    assert lines_of(run_dir)[:len(copied)] == copied
    assert {c["run_id"] for c in calls if c["question_id"] in ("1", "2", "3")} == {source["run_id"]}
    assert {c["run_id"] for c in calls if c["question_id"] in ("4", "5", "6")} == {manifest["run_id"]}
    assert {c["question_id"] for c in calls} == set(QUESTIONS)
    assert validate_calls(calls) == [] and manifest["c1_errors"] == 0 and manifest["n_calls"] == len(calls)
    # read as any done run is
    assert require_done(manifest["run_id"], type="agent", arm="B0", split="train")["status"] == "done"
    assert len(calls_of(manifest["run_id"])) == len(calls)


def test_the_manifest_records_the_source_the_sessions_and_what_was_discarded(failed):
    server, config_path, source_dir = failed
    source, source_calls, _ = read_run(source_dir)
    manifest, _, _ = read_run(runner.run_agent(str(config_path), "B0", "train", resume_from=source["run_id"]))

    resumed = manifest["resumed_from"]
    for key in ("run_id", "commit", "dirty", "config_sha256", "status", "started_at", "finished_at", "workers"):
        assert resumed[key] == source[key]
    assert resumed["config_sha256"] == manifest["config_sha256"]
    four = [c for c in source_calls if c["question_id"] == "4"]
    assert resumed["discarded"]["n_calls"] == len(four) > 0 and resumed["discarded"]["question_ids"] == ["4"]
    assert resumed["discarded"]["usage"]["input"] == sum(c["usage"]["input"] or 0 for c in four) > 0  # a failed call has none

    first, this = manifest["sessions"]
    assert (first["started_at"], first["finished_at"], first["commit"], first["workers"]) == \
        (source["started_at"], source["finished_at"], source["commit"], 1)
    assert (this["started_at"], this["finished_at"], this["commit"], this["workers"]) == \
        (manifest["started_at"], manifest["finished_at"], manifest["commit"], 1)
    worked = sum((datetime.fromisoformat(s["finished_at"]) - datetime.fromisoformat(s["started_at"])).total_seconds()
                 for s in manifest["sessions"])
    assert preflight._hours(manifest) == pytest.approx(worked / 3600)


def test_a_question_in_flight_when_the_run_stopped_is_answered_again_from_the_start(failed):
    """In flight when a worker failed, a question's calls were refused and the runner left its
    prediction out: on disk, its lines without a prediction. It is not copied, and is run again."""
    server, config_path, source_dir = failed
    predictions_path = source_dir / "predictions.json"
    predictions = json.loads(predictions_path.read_text())
    del predictions["3"]
    predictions_path.write_text(json.dumps(predictions))
    source = json.loads((source_dir / "manifest.json").read_text())

    run_dir = runner.run_agent(str(config_path), "B0", "train", workers=2, resume_from=source["run_id"])
    manifest, calls, predictions = read_run(run_dir)

    assert manifest["status"] == "done" and manifest["resumed_question_ids"] == ["1", "2"]
    assert asked(server, QUESTIONS) == {"3", "4", "5", "6"}
    assert list(predictions) == QUESTIONS and predictions["3"] == f" {GOLD} "
    assert {c["run_id"] for c in calls if c["question_id"] == "3"} == {manifest["run_id"]}  # only the new attempt
    assert manifest["resumed_from"]["discarded"]["question_ids"] == ["3", "4"]
    assert manifest["sessions"][-1]["workers"] == 2 and validate_calls(calls) == []
    assert not list(run_dir.glob("calls.w*")) and not list(run_dir.glob("outcome.w*"))
    # CHESS's own outputs of the copied questions, for inspection
    kept = sorted(p.name.split("_")[0] for p in (run_dir / "chess" / "resumed" / source["run_id"]).iterdir())
    assert kept == ["1", "2"]


def runs_now():
    return sorted(p.name for p in paths.RUNS.iterdir())


def test_a_resume_is_refused_unless_it_is_the_same_run_on_a_resumable_source(failed, tmp_path):
    server, config_path, source_dir = failed
    source_id = source_dir.name
    before = runs_now()

    from bench.barrier import TestSplitLocked
    with pytest.raises(TestSplitLocked):  # on test, the barrier first (the rule there: test_resume_on_test)
        runner.run_agent(str(config_path), "B0", "test", resume_from=source_id)
    with pytest.raises(DataError, match="arm is 'B0', this run's is 'B2'"):
        runner.run_agent(str(config_path), "B2", "train", engine="production_llm", resume_from=source_id)
    with pytest.raises(DataError, match="split"):
        runner.run_agent(str(config_path), "B0", "calib", resume_from=source_id)
    other = yaml.safe_load(config_path.read_text())
    other["agent"]["max_workers"] = 3
    other_path = tmp_path / "other.yaml"
    other_path.write_text(yaml.safe_dump(other))
    with pytest.raises(DataError, match="config_sha256"):
        runner.run_agent(str(other_path), "B0", "train", resume_from=source_id)
    with pytest.raises(DataError, match="no --ids or --limit"):
        runner.run_agent(str(config_path), "B0", "train", ids=["5"], resume_from=source_id)
    with pytest.raises(DataError, match="no run"):
        runner.run_agent(str(config_path), "B0", "train", resume_from="agent-B0-train-nothing")
    assert runs_now() == before  # a refusal leaves no run

    done = runner.run_agent(str(config_path), "B0", "train", resume_from=source_id)
    with pytest.raises(DataError, match="'done': only a failed, interrupted or killed run is resumed"):
        runner.run_agent(str(config_path), "B0", "train", resume_from=done.name)


def test_the_cli_resumes(failed, capsys):
    server, config_path, source_dir = failed
    assert cli.main(["run", "--config", str(config_path), "--arm", "B0", "--split", "train",
                     "--resume-from", source_dir.name]) == 0
    run_dir = paths.RUNS / capsys.readouterr().out.strip().splitlines()[-1].rsplit("/", 1)[-1]
    assert json.loads((run_dir / "manifest.json").read_text())["resumed_from"]["run_id"] == source_dir.name


# ---------------------------------------------------------------- a session killed outright (SIGKILL)

class Killed(BaseException):
    """Stands for SIGKILL: nothing after it runs in the process."""


def killed_run(config_path, monkeypatch, workers=1):
    """An agent run killed after its questions ended or failed and before it could close anything: its
    manifest stays `running`, predictions.json never written, its lock released by the dying process."""
    import os
    real_write_json = runner._write_json

    def write_json(path, value):
        if path.name == "predictions.json":  # the first thing the run's `finally` does: the kill came first
            fd = runner._SESSION_LOCKS.pop(path.parent)
            os.close(fd)  # the operating system releases a dead process's lock
            raise Killed()
        real_write_json(path, value)
    monkeypatch.setattr(runner, "_write_json", write_json)
    with pytest.raises(Killed):
        runner.run_agent(str(config_path), "B0", "train", workers=workers)
    monkeypatch.setattr(runner, "_write_json", real_write_json)
    (run_dir,) = [p for p in paths.RUNS.iterdir() if p.name.startswith("agent-B0-train")
                  and json.loads((p / "manifest.json").read_text())["status"] == "running"]
    return run_dir


@pytest.mark.parametrize("workers", [1, 2])
def test_a_killed_run_is_resumed_from_what_its_session_journaled(tmp_path, monkeypatch, workers):
    server = FakeChess(refuse=marker("4"))
    config_path = make_worker_repo(tmp_path, monkeypatch, server)
    source_dir = killed_run(config_path, monkeypatch, workers)
    source = json.loads((source_dir / "manifest.json").read_text())
    assert source["status"] == "running" and not (source_dir / "predictions.json").exists()
    assert not (source_dir / runner.SESSION_LOCK).exists() or runner.session_alive(source_dir) is False
    calls_files = sorted(p.name for p in source_dir.glob("calls*.jsonl"))
    with open(source_dir / calls_files[-1], "a") as fh:
        fh.write('{"run_id": "cut off by the ki')  # the line being written when the process died
    from bench.data import sha256_file
    before = {p.name: sha256_file(p) for p in source_dir.glob("*.jsonl")}
    server.refuse, server.prompts[:] = None, []

    run_dir = runner.run_agent(str(config_path), "B0", "train", workers=workers, resume_from=source["run_id"])
    manifest, calls, predictions = read_run(run_dir)
    assert manifest["status"] == "done" and list(predictions) == QUESTIONS
    answered = sorted(manifest["resumed_question_ids"], key=int)
    assert answered and "4" not in answered and set(answered) | asked(server, QUESTIONS) == set(QUESTIONS)
    assert not set(answered) & asked(server, QUESTIONS)  # what the dead session answered is not asked again
    assert manifest["unregistered_source_hashes"] == before  # computed before anything was read
    assert manifest["sessions"][0]["killed"] is True and manifest["sessions"][0]["finished_at"]
    assert validate_calls(calls) == [] and not list(run_dir.glob("answers*.jsonl"))


def test_a_running_source_whose_session_holds_its_lock_or_left_none_is_not_resumed(failed):
    server, config_path, source_dir = failed
    import fcntl
    import os
    path = source_dir / "manifest.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "status": "running"}))
    with pytest.raises(DataError, match="left no session lock"):
        runner.run_agent(str(config_path), "B0", "train", resume_from=source_dir.name)
    fd = os.open(source_dir / runner.SESSION_LOCK, os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX)  # a session alive (here, this process) holds it
    try:
        with pytest.raises(DataError, match="still running"):
            runner.run_agent(str(config_path), "B0", "train", resume_from=source_dir.name)
    finally:
        os.close(fd)


def test_a_sigkilled_process_releases_its_session_lock(tmp_path):
    """The lock is the operating system's: a process killed with SIGKILL holds it no more."""
    import signal
    import subprocess
    import sys
    (tmp_path / runner.SESSION_LOCK).touch()
    holder = subprocess.Popen([sys.executable, "-c", (
        "import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR); fcntl.flock(fd, fcntl.LOCK_EX); "
        "print('held', flush=True); time.sleep(60)"), str(tmp_path / runner.SESSION_LOCK)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        assert runner.session_alive(tmp_path) is True
        holder.send_signal(signal.SIGKILL)
        holder.wait(timeout=10)
        assert runner.session_alive(tmp_path) is False
    finally:
        holder.kill()
