"""`bench run --workers N`: one run answered by N processes. The workers are real processes (spawned),
so nothing patched in this process reaches them: the model is a fake OpenAI-compatible endpoint that
answers CHESS's prompts as a model following the protocol would, and the run goes through the agent's
real client. Agent environment only; no llama.cpp server, no BIRD download."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("langchain_openai")

from bench import cli, paths  # noqa: E402
from bench.agent import hooks, runner  # noqa: E402
from bench.contracts.calls import read_calls, validate_calls  # noqa: E402
from bench.contracts.config import load_config  # noqa: E402
from bench.data import DataError  # noqa: E402
from synthetic import GOLD, make_repo, sha256  # noqa: E402
from test_agent_wiring import ScriptedChess  # noqa: E402

QUESTIONS = [str(q) for q in range(1, 7)]


class FakeChess:
    """An OpenAI-compatible endpoint answering each CHESS prompt as `ScriptedChess` does. A prompt
    that carries `refuse` is answered 401 (the harness's failure: the run fails); `hold(text)` runs
    before any other answer, and may wait."""

    def __init__(self, refuse=None, hold=None):
        fake, script = self, ScriptedChess()
        self.prompts = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                text = body["messages"][-1]["content"]
                fake.prompts.append(text)
                if refuse and refuse in text:
                    code, answer = 401, {"error": {"message": "this key is not allowed", "type": "authentication_error"}}
                else:
                    if hold:
                        hold(text)
                    code, answer = 200, {
                        "id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant", "content": script.answer(text)}}],
                        "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 5, "total_tokens": len(text) // 4 + 5}}
                raw = json.dumps(answer).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def marker(question_id):
    return f"(question {question_id})"


def make_worker_repo(root, monkeypatch, server):
    """`make_repo` with six train questions, each with its own text, preprocessed, and the production
    LLM served by `server`. Returns the configuration's path."""
    _, config_path, _ = make_repo(root, monkeypatch)
    dev = [{"question_id": int(q), "db_id": "tiny", "question": f"How many gas stations in CZE are Premium? {marker(q)}",
            "evidence": None, "difficulty": "simple", "SQL": GOLD} for q in QUESTIONS]
    questions_path = paths.RAW / "bird_dev_questions.json"
    questions_path.write_text(json.dumps(dev))
    config = yaml.safe_load(config_path.read_text())
    config["data"]["bird_dev_questions"]["sha256"] = sha256(questions_path)
    config["agent"]["max_workers"] = 4
    config["roles"]["production_llm"]["endpoint"] = {"kind": "fake", "base_url": server.base_url, "api_key_env": None}
    config_path.write_text(yaml.safe_dump(config))
    paths.SPLITS.write_text(json.dumps({"train": QUESTIONS, "calib": [], "test": ["9"], "excluded": []}))
    monkeypatch.setattr(hooks, "_models", {})
    runner.preprocess(str(config_path), ["tiny"])
    return config_path


def read_run(run_dir):
    return (json.loads((run_dir / "manifest.json").read_text()), read_calls(run_dir / "calls.jsonl"),
            json.loads((run_dir / "predictions.json").read_text()))


def identity(call):
    return call["question_id"], call["call_site"], call["invocation_key"], call["attempt"]


def test_three_workers_give_the_c1_lines_and_predictions_of_one(tmp_path, monkeypatch, capsys):
    server = FakeChess()
    config_path = make_worker_repo(tmp_path, monkeypatch, server)
    one, one_calls, one_predictions = read_run(runner.run_agent(str(config_path), "B0", "train"))
    capsys.readouterr()
    assert cli.main(["run", "--config", str(config_path), "--arm", "B0", "--split", "train", "--workers", "3"]) == 0
    run_dir = paths.RUNS / capsys.readouterr().out.strip().splitlines()[-1].rsplit("/", 1)[-1]
    three, three_calls, three_predictions = read_run(run_dir)

    assert (one["status"], three["status"]) == ("done", "done"), three
    assert (one["workers"], three["workers"]) == (1, 3)
    assert one["config_sha256"] == three["config_sha256"]  # the number of workers is not the configuration's
    assert one["question_ids"] == three["question_ids"] == QUESTIONS
    assert three_predictions == one_predictions and list(three_predictions) == QUESTIONS
    assert set(three_predictions.values()) == {f" {GOLD} "}
    # the same invocations, attempt by attempt (call_id is random per line: the identity is what pairs runs)
    assert sorted(map(identity, three_calls)) == sorted(map(identity, one_calls))
    assert len({c["call_id"] for c in three_calls}) == len(three_calls) == three["n_calls"]
    assert {c["run_id"] for c in three_calls} == {three["run_id"]}  # one run, whoever answered
    assert validate_calls(three_calls) == [] and three["c1_errors"] == 0
    # merged: one calls.jsonl, nothing of the workers' own files left beside it
    assert not list(run_dir.glob("calls.w*")) and not list(run_dir.glob("outcome.w*"))
    assert sorted(p.name for p in run_dir.glob("questions*.json")) == [f"questions.w{k}.json" for k in range(3)]
    asked = [[str(t["question_id"]) for t in json.loads((run_dir / f"questions.w{k}.json").read_text())] for k in range(3)]
    assert asked == [["1", "4"], ["2", "5"], ["3", "6"]]  # the deterministic, disjoint parts


def test_a_question_that_fails_in_one_worker_stops_the_others(tmp_path, monkeypatch):
    """Question 1 (worker 0) fails. The other workers' first calls that carry their question (the
    keyword extraction; the agent's own prompt before it does not) are held until worker 0 has
    reported, so they are mid-question when the run fails: they make no further call, and the
    questions after those never start."""
    def hold(text):
        if not any(marker(q) in text for q in QUESTIONS[1:]):
            return
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline and not list(paths.RUNS.glob("agent-B0-train-*/outcome.w0.json")):
            time.sleep(0.05)

    server = FakeChess(refuse=marker("1"), hold=hold)
    config_path = make_worker_repo(tmp_path, monkeypatch, server)
    manifest, calls, predictions = read_run(runner.run_agent(str(config_path), "B0", "train", workers=3))

    assert manifest["status"] == "failed" and manifest["workers"] == 3
    assert list(manifest["harness_errors"]) == ["1"] and "401" in json.dumps(manifest["harness_errors"])
    by_question = {q: [c for c in calls if c["question_id"] == q] for q in QUESTIONS}
    assert [c["call_site"] for c in by_question["1"]] == ["agent_ir", "extract_keywords"]
    assert "401" in by_question["1"][1]["error"]
    for held in ("2", "3"):  # in flight when the run failed: what was already sent, and not one call more
        assert {c["call_site"] for c in by_question[held]} <= {"agent_ir", "extract_keywords"}
        assert len(by_question[held]) <= 2
        assert held not in predictions and held not in manifest["tool_errors"]  # cut short: not an answer
    for never_started in ("4", "5", "6"):
        assert by_question[never_started] == [] and never_started not in predictions
    assert not any(marker(q) in prompt for prompt in server.prompts for q in ("4", "5", "6"))
    assert validate_calls(calls) == []


def test_the_single_call_arm_runs_in_workers_too(tmp_path, monkeypatch):
    server = FakeChess()
    config_path = make_worker_repo(tmp_path, monkeypatch, server)
    manifest, calls, predictions = read_run(
        runner.run_agent(str(config_path), "B2", "train", engine="production_llm", workers=2))
    assert manifest["status"] == "done" and manifest["workers"] == 2
    assert predictions == {q: f" {GOLD} " for q in QUESTIONS}
    assert sorted(identity(c) for c in calls) == sorted((q, "generate_candidate", "b2:0", 1) for q in QUESTIONS)


def test_the_parts_are_disjoint_cover_the_run_and_depend_only_on_its_order():
    ids = [str(q) for q in range(1, 11)]
    parts = runner.worker_parts(ids, 3)
    assert parts == [["1", "4", "7", "10"], ["2", "5", "8"], ["3", "6", "9"]]
    assert sorted(sum(parts, []), key=int) == ids
    assert runner.worker_parts(ids[:2], 5) == [["1"], ["2"]]  # no worker without a question


def test_fewer_than_one_worker_is_refused(tmp_path, monkeypatch):
    _, config_path, _ = make_repo(tmp_path, monkeypatch)
    with pytest.raises(DataError, match="--workers must be at least 1"):
        runner.run_agent(str(config_path), "B0", "train", workers=0)
    assert not paths.RUNS.exists() or not list(paths.RUNS.iterdir())


SMOKE = paths.ROOT / "configs" / "smoke-local.yaml"


def test_a_worker_that_stops_stops_the_others_and_says_why_without_a_credential(tmp_path, monkeypatch):
    """What stops a worker other than a failed question (a Ctrl-C, a failure before its first
    question) sets the shared flag and is left in its outcome for the run's manifest. The outcome
    file outlives a parent that dies before merging it: it is redacted like the manifest."""
    stop = threading.Event()
    monkeypatch.setenv("OPENAI_API_KEY", "embeddings-key-0001")

    key_shaped = "SELECT hf_AbCdEf123456 FROM t"  # a prediction is the model's SQL, never rewritten

    def execute(outcome, part, stop, run_dir, dataset, config):
        outcome["predictions"]["1"] = key_shaped  # what it finished is kept
        outcome["failures"]["2"] = "AuthenticationError: bad key embeddings-key-0001 (hf_AbCdEf123456)"
        raise KeyboardInterrupt

    config = load_config(SMOKE)
    from bench.provenance import redact
    assert redact(key_shaped, config) != key_shaped  # the redaction would rewrite it
    runner._worker(execute, {"run_dir": tmp_path, "dataset": [], "config": config}, 2, stop)
    written = (tmp_path / "outcome.w2.json").read_text()
    reported = json.loads(written)
    assert stop.is_set() and reported["stopped_by"] == "KeyboardInterrupt: "
    assert reported["predictions"] == {"1": key_shaped}  # as a run of one process writes it
    assert "embeddings-key-0001" not in written and reported["failures"]["2"] == \
        "AuthenticationError: bad key <redacted> (<redacted>)"


def test_the_merge_keeps_every_workers_lines_and_tells_a_dead_worker_from_a_stopped_one(tmp_path):
    (tmp_path / "calls.w0.jsonl").write_text(json.dumps({"call_id": "a"}) + "\n")
    (tmp_path / "outcome.w0.json").write_text(json.dumps({
        **runner.new_outcome(), "predictions": {"1": "SELECT 1"}, "unregistered_call_sites": ["revise"]}))
    # killed: its lines (the last one cut off mid-write) and no outcome
    (tmp_path / "calls.w1.jsonl").write_text(json.dumps({"call_id": "b"}) + "\n" + '{"call_id": "cut')
    (tmp_path / "calls.w2.jsonl").write_text(json.dumps({"call_id": "c"}) + "\n")
    (tmp_path / "outcome.w2.json").write_text(json.dumps({
        **runner.new_outcome(), "tool_errors": {"3": {"revise": "x"}}, "stopped_by": "ImportError: no module"}))
    (tmp_path / "outcome.w3.json").write_text('{"predictions": {"4": "SEL')  # killed while writing its outcome
    (tmp_path / "outcome.w4.json").write_text(json.dumps({**runner.new_outcome(), "predictions": {"5": "SELECT 5"}}))
    outcome = runner.new_outcome()
    exits = (0, -9, 0, -9, 1)  # worker 4 reported, then crashed on its way out
    stopped = runner._merge_workers(tmp_path, [SimpleNamespace(exitcode=code) for code in exits], outcome)
    assert stopped == ["worker 2 stopped: ImportError: no module"]
    assert outcome["harness_errors"] == {
        "worker 1": ["the worker did not end cleanly (exit code -9, no outcome reported)"],
        "worker 3": ["the worker did not end cleanly (exit code -9, no outcome reported)"],
        "worker 4": ["the worker did not end cleanly (exit code 1)"]}
    assert runner.run_status(["1"], outcome, 0, []) == "failed"  # a dead worker fails the run, whatever was answered
    rows = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert rows[2] == '{"call_id": "cut' and [json.loads(row)["call_id"] for row in rows[:2] + rows[3:]] == ["a", "b", "c"]
    assert outcome["predictions"] == {"1": "SELECT 1", "5": "SELECT 5"} and outcome["tool_errors"] == {"3": {"revise": "x"}}
    assert outcome["unregistered_call_sites"] == ["revise"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calls.jsonl"]


def scripted_worker(outcome, part, stop, run_dir, dataset, config):
    """A worker's work, scripted by its part (run in a real spawned process): `dies` exits hard,
    `raises` fails before any question, anything else waits to be stopped and says whether it was."""
    if dataset == ["dies"]:
        os._exit(9)
    if dataset == ["raises"]:
        raise RuntimeError("no agent to run")
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not stop.is_set():
        time.sleep(0.02)
    outcome["predictions"][dataset[0]] = "stopped" if stop.is_set() else "never stopped"


def test_a_worker_that_dies_stops_the_others_and_fails_the_run(tmp_path):
    """A killed worker cannot set the flag itself: the parent sets it on the non-zero exit, whichever
    worker it is (here the last one: a parent waiting for the workers in order would never see it)."""
    outcome = runner.new_outcome()
    runner._in_workers(scripted_worker, {"run_dir": tmp_path, "config": load_config(SMOKE)}, [["waits"], ["dies"]], outcome)
    assert outcome["predictions"] == {"waits": "stopped"}
    assert outcome["harness_errors"] == {"worker 1": ["the worker did not end cleanly (exit code 9, no outcome reported)"]}
    assert runner.run_status(["waits", "dies"], outcome, 0, []) == "failed"


def test_a_worker_that_cannot_start_stops_the_ones_already_started(tmp_path):
    """The second worker's part cannot be sent to a process: the first, already answering, is stopped
    and merged before the failure goes on."""
    outcome = runner.new_outcome()
    with pytest.raises(Exception, match="pickle"):
        runner._in_workers(scripted_worker, {"run_dir": tmp_path, "config": load_config(SMOKE)},
                           [["waits"], [lambda: None]], outcome)
    assert outcome["predictions"] == {"waits": "stopped"}
    assert list(outcome["harness_errors"]) == ["worker 1"]  # never started: no outcome


def test_a_worker_stopped_before_its_questions_stops_the_others_and_raises_in_the_parent(tmp_path):
    """As in a run of one process, where the same failure raises out of the run (`stopped_by`)."""
    outcome = runner.new_outcome()
    with pytest.raises(hooks.HarnessError, match="worker 0 stopped: RuntimeError: no agent to run"):
        runner._in_workers(scripted_worker, {"run_dir": tmp_path, "config": load_config(SMOKE)},
                           [["raises"], ["waits"]], outcome)
    assert outcome["predictions"] == {"waits": "stopped"} and outcome["harness_errors"] == {}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["calls.jsonl"]  # merged before it raised


def test_a_question_running_when_the_run_is_stopped_from_outside_is_not_an_answer():
    """Its model calls were refused, which CHESS swallows: what it returned must not count, or a run
    stopped on every worker's last question would end `done`. The question that fails by itself keeps
    its record, as in a run of one process."""
    stop, outcome = threading.Event(), runner.new_outcome()
    fake_hooks = SimpleNamespace(set_question=lambda q: None, take_harness_errors=lambda: [])

    def answered_then_stopped():
        stop.set()  # from another worker, while this question runs
        return None, {"extract_keywords": "RunAborted: the run has failed in another worker"}
    runner._answer_each(fake_hooks, outcome, stop, [("1", lambda: ("SELECT 1", {})), ("2", answered_then_stopped),
                                                    ("3", lambda: ("SELECT 3", {}))])
    assert outcome["predictions"] == {"1": "SELECT 1"} and outcome["tool_errors"] == {}
    assert runner.run_status(["1", "2", "3"], outcome, 0, []) == "interrupted"

    def raises():
        raise ValueError("the question itself failed")
    stop, outcome = threading.Event(), runner.new_outcome()
    runner._answer_each(fake_hooks, outcome, stop, [("1", raises), ("2", lambda: ("SELECT 2", {}))])
    assert stop.is_set() and outcome["predictions"] == {"1": None} and list(outcome["failures"]) == ["1"]
