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
    assert resumed["routes"] is None and "routes" not in resumed["discarded"]  # B0 does not embed: no router time

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

    with pytest.raises(DataError, match="refused on test"):
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
    with pytest.raises(DataError, match="'done': only a failed or interrupted run is resumed"):
        runner.run_agent(str(config_path), "B0", "train", resume_from=done.name)


def test_the_cli_resumes(failed, capsys):
    server, config_path, source_dir = failed
    assert cli.main(["run", "--config", str(config_path), "--arm", "B0", "--split", "train",
                     "--resume-from", source_dir.name]) == 0
    run_dir = paths.RUNS / capsys.readouterr().out.strip().splitlines()[-1].rsplit("/", 1)[-1]
    assert json.loads((run_dir / "manifest.json").read_text())["resumed_from"]["run_id"] == source_dir.name


def test_the_router_times_of_the_complete_questions_are_copied_and_their_absence_recorded(tmp_path, monkeypatch):
    from bench.agent import resume

    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    source = tmp_path / "runs" / "agent-B4-src"
    source.mkdir(parents=True)
    rows = [{"question_id": q, "call_site": "agent_ir", "invocation_key": "ir", "embed_ms": 100.0} for q in ("1", "2", "4")]
    (source / "routes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    new = tmp_path / "runs" / "agent-B4-new"
    new.mkdir()
    assert resume.copy_routes({"run_id": "agent-B4-src", "arm": "B4"}, new, {"1", "2"}) == {"copied": 2, "missing": False}
    assert [json.loads(line)["question_id"] for line in (new / "routes.jsonl").read_text().splitlines()] == ["1", "2"]
    (source / "routes.jsonl").unlink()  # a routed run from before routes.jsonl existed
    assert resume.copy_routes({"run_id": "agent-B4-src", "arm": "B5"}, new, {"1"}) == {"copied": 0, "missing": True}
    assert resume.copy_routes({"run_id": "agent-B4-src", "arm": "B3"}, new, {"1"}) is None
