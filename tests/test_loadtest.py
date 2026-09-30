"""`bench loadtest` without a server: the replayed request is the agent's own, AIPerf is invoked as pinned,
one run per concurrency level with its manifest, and the refusals. The live run against llama.cpp is
tests/test_loadtest_llamacpp.py."""
import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from bench import cli, loadtest, paths
from bench.contracts.config import engine_spec
from bench.loadtest import LoadtestError, aiperf_command, build_payloads, level_slice, server_root, wait_ready
from synthetic import make_repo
from test_train_fixtures import TINY, TINY_NAME, make_s5_repo, save

CALLS = json.loads((Path(__file__).parent / "fixtures_calls.json").read_text())


def test_each_call_becomes_the_body_the_agents_client_sends_to_the_engines_model():
    params = {"max_tokens": 64, "timeout_s": 600, "top_p": 0.9}
    payloads = build_payloads(CALLS, "c3", params, stream=False)
    assert len(payloads) == len(CALLS)
    for call, body in zip(CALLS, payloads):
        assert body == {"model": "c3", "messages": call["prompt_messages"], "temperature": call["temperature"],
                        "max_tokens": 64, "top_p": 0.9}
    streamed = build_payloads(CALLS[:1], "c3", {"max_tokens": None, "timeout_s": 1}, stream=True)[0]
    assert streamed["stream"] is True and streamed["stream_options"] == {"include_usage": True}
    assert "max_tokens" not in streamed


class RecordingChat:
    """POST /v1/chat/completions: records the body on the wire, answers a minimal completion."""

    def __init__(self):
        self.bodies = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                fake.bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                   "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                "finish_reason": "stop"}],
                                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


def test_the_replayed_body_is_what_the_patched_agent_sends(monkeypatch):
    """Against the agent's real client (bench/agent/hooks.py), on the wire: the same fields, plus only the
    client's defaults."""
    pytest.importorskip("langchain_openai")
    from langchain_core.messages import HumanMessage

    from bench.agent import hooks
    from bench.contracts.config import load_config

    server = RecordingChat()
    config = copy.deepcopy(load_config(paths.ROOT / "configs" / "smoke-local.yaml"))
    config["roles"]["production_llm"]["endpoint"]["base_url"] = server.base_url
    monkeypatch.setattr(hooks, "_models", {})
    hooks.configure(config)
    call = CALLS[0]
    try:
        hooks.chat_model("production_llm", call["temperature"]).invoke(
            [HumanMessage(content=call["prompt_messages"][0]["content"])])
    finally:
        server.server.shutdown()
    sent = server.bodies[0]
    spec = engine_spec(config, "production_llm")
    replayed = build_payloads([call], spec["model"], spec["params"], stream=False)[0]
    assert {k: sent[k] for k in replayed} == replayed
    assert set(sent) - set(replayed) <= {"n", "stream"} and not sent.get("stream")


def test_aiperf_is_given_the_server_root_and_the_pinned_invocation():
    assert server_root("http://h:8000/v1/") == "http://h:8000"
    with pytest.raises(LoadtestError, match="does not end in /v1"):
        server_root("http://h:8000/api")
    cmd = aiperf_command("aiperf", "http://h:8000", "c3", 8, 500, TINY, 600, False, None)
    flags = {flag: (value if not value.startswith("--") else True)
             for flag, value in zip(cmd[2:], cmd[3:] + ["--"]) if flag.startswith("--")}
    assert cmd[:2] == ["aiperf", "profile"] and flags["--custom-dataset-type"] == "raw_payload"
    assert flags["--input-file"] == "payloads.jsonl" and flags["--output-artifact-dir"] == "."
    assert (flags["--concurrency"], flags["--request-count"]) == ("8", "500") and "--warmup-request-count" not in cmd
    assert flags["--dataset-sampling-strategy"] == "sequential"  # each line once, in order
    assert (flags["--tokenizer"], flags["--tokenizer-revision"]) == (TINY["repo"], TINY["revision"])
    assert "--use-server-token-count" in cmd and "--streaming" not in cmd and "--api-key" not in cmd
    keyed = aiperf_command("aiperf", "http://h:8000", "c3", 8, 500, TINY, 600, True, "k")
    assert "--streaming" in keyed and keyed[keyed.index("--api-key") + 1] == "k"


class FakeServer:
    """GET /v1/models, listing `models` once `ready_after` requests have been made; POST records the
    warm-up calls."""

    def __init__(self, models, ready_after=0):
        self.models, self.ready_after, self.seen, self.posted = models, ready_after, [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _answer(self, payload):
                body = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                fake.seen.append((self.path, self.headers.get("Authorization")))
                listed = fake.models if len(fake.seen) > fake.ready_after else []
                self._answer({"data": [{"id": m, "root": f"/adapters/{m}", "parent": "base"} for m in listed]})

            def do_POST(self):
                fake.posted.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self._answer({"choices": []})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


def test_wait_ready_waits_for_the_model_to_be_listed():
    server = FakeServer(["c3"], ready_after=2)
    try:
        waited, card = wait_ready(server.base_url, "k", "c3", timeout_s=5, poll_s=0.01)
        assert waited >= 0 and card == {"id": "c3", "root": "/adapters/c3", "parent": "base"}
        assert len(server.seen) == 3 and server.seen[0] == ("/v1/models", "Bearer k")
        with pytest.raises(LoadtestError, match="c9 not served"):
            wait_ready(server.base_url, None, "c9", timeout_s=0.05, poll_s=0.01)
    finally:
        server.close()


def _source_run(run_id="agent-B0-train-src", split="train", calls=CALLS):
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "split": split, "status": "done"}))
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return run_id


def test_each_level_replays_its_own_calls_after_the_warm_up():
    payloads = [{"i": i} for i in range(10)]
    first, second = level_slice(payloads, 2, 0, 3), level_slice(payloads, 2, 1, 3)
    assert [p["i"] for p in first["payloads"]] == [2, 3, 4] and [p["i"] for p in second["payloads"]] == [5, 6, 7]
    assert (first["offset"], first["repeats"], second["repeats"]) == (2, False, False)
    third = level_slice(payloads, 2, 2, 3)  # 8, 9, then the source is exhausted
    assert [p["i"] for p in third["payloads"]] == [8, 9, 0] and third["repeats"] is True


SOURCE = [dict(CALLS[i % 2], call_id=f"00000000-0000-4000-8000-{i:012d}", invocation_key=f"k:{i}") for i in range(12)]


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A synthetic repo whose tiny SLM candidate is 'served' by a fake that lists it and its adapter."""
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    server = FakeServer([TINY_NAME, "c0"])
    config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
    config["loadtest"].update({"concurrency": [1, 4], "request_count": 5, "warmup_request_count": 2})
    yield config_path, save(config, config_path), server
    server.close()


def _fake_aiperf(returncode=0, calls=None):
    def run(run_dir, cmd):
        (calls if calls is not None else []).append((run_dir, cmd))
        assert (run_dir / "payloads.jsonl").is_file()
        if returncode == 0:
            (run_dir / "profile_export_aiperf.json").write_text(json.dumps({"request_count": {"avg": 6}}))
        return returncode
    return run


def test_one_run_per_concurrency_level_each_with_its_manifest(served, monkeypatch):
    config_path, config, server = served
    calls = []
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf(calls=calls))
    source = _source_run(calls=SOURCE)
    run_dirs = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", source, "local")
    assert [json.loads((d / "manifest.json").read_text())["concurrency"] for d in run_dirs] == [1, 4]
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["model"] == "c0" and manifest["engine"] == f"slm:{TINY_NAME}+lora:c0"
    assert manifest["gpu"] == config["serving"]["gpu"] and manifest["prefix_cache"] is True
    assert manifest["source_run_id"] == source and manifest["source_split"] == "train"
    assert (manifest["source_calls"], manifest["n_questions"]) == (12, 1)
    assert manifest["tokenizer"] == TINY and manifest["where"] == "local" and manifest["aiperf_returncode"] == 0
    assert manifest["run_id"].startswith(f"loadtest-slm_{TINY_NAME}_lora_c0-c1-")
    assert manifest["served_model"] == {"id": "c0", "root": "/adapters/c0", "parent": "base"}  # what the server said
    levels = [[json.loads(line) for line in (d / "payloads.jsonl").read_text().splitlines()] for d in run_dirs]
    assert {p["model"] for p in levels[0]} == {"c0"} and levels[0][0]["max_tokens"] == 64
    # the warm-up calls (sent before each level), then level 1 and level 2 each on their own calls
    expected = build_payloads(SOURCE, "c0", config["roles"]["slm_candidates"][-1]["params"], False)
    assert levels == [expected[2:7], expected[7:12]]
    assert server.posted == expected[:2] * 2
    second = json.loads((run_dirs[1] / "manifest.json").read_text())
    assert (manifest["payload_offset"], second["payload_offset"], second["payloads_repeat"]) == (2, 7, False)
    _, cmd = calls[1]
    assert cmd[cmd.index("--concurrency") + 1] == "4" and cmd[cmd.index("--model") + 1] == "c0"


def test_a_failed_level_is_recorded_and_stops_the_sweep(served, monkeypatch):
    config_path, _, _ = served
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf(returncode=1))
    with pytest.raises(LoadtestError, match="AIPerf failed at concurrency 1"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run(), "local")
    manifests = [json.loads(p.read_text()) for p in paths.RUNS.glob("loadtest-*/manifest.json")]
    assert [(m["concurrency"], m["status"]) for m in manifests] == [(1, "failed")]


def test_refusals(served, monkeypatch):
    config_path, config, server = served
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf())
    with pytest.raises(LoadtestError, match="has no manifest.json and calls.jsonl"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", "agent-missing", "local")
    bad = copy.deepcopy(CALLS)
    bad[1]["retry_of"] = "not-a-call"
    with pytest.raises(LoadtestError, match="not valid C1"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run("agent-bad", calls=bad), "local")
    with pytest.raises(LoadtestError, match="is not an SLM: give its tokenizer"):
        loadtest.loadtest(str(config_path), "production_llm", _source_run("agent-ok"), "local")
    config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = None
    with pytest.raises(LoadtestError, match="no endpoint.base_url"):
        loadtest.loadtest(str(save(config, config_path) and config_path), f"slm:{TINY_NAME}", "agent-ok", "local")
    assert not list(paths.RUNS.glob("loadtest-*"))


def test_a_test_split_source_goes_through_the_barrier(tmp_path, monkeypatch):
    from bench.barrier import TestSplitLocked

    _, config_path, config = make_repo(tmp_path, monkeypatch)
    with pytest.raises(TestSplitLocked):
        loadtest.loadtest(str(config_path), "production_llm", _source_run("agent-B0-test-x", split="test"), "local",
                          tokenizer=TINY)


def test_cli_needs_a_pinned_tokenizer_revision(capsys):
    assert cli.main(["loadtest", "--engine", "production_llm", "--source", "x", "--on", "local",
                     "--tokenizer", TINY["repo"], "--tokenizer-revision", "main"]) == 2
    assert "40-hex" in capsys.readouterr().err


def test_on_modal_the_client_runs_beside_the_server_and_its_artifacts_come_back(served, monkeypatch):
    from test_train_fixtures import fake_modal_app

    config_path, config, server = served
    config["roles"]["slm_candidates"][-1]["endpoint"]["api_key_env"] = "UNSET_LOCALLY"  # the key is Modal's secret
    config = save(config, config_path)
    artifacts = {"profile_export_aiperf.json": b'{"request_count": {"avg": 6}}', "logs/aiperf.log": b"ok"}
    calls = fake_modal_app(monkeypatch, "loadtest", run_aiperf=lambda raw, args: {
        "returncode": 0, "ready_after_s": 42.0, "files": artifacts,
        "served_model": {"id": "c0", "root": "/adapters/abc", "parent": TINY_NAME, "object": "model"}})
    source = _source_run(calls=SOURCE)
    run_dirs = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", source, "modal", concurrency=[8])
    (raw, args), = calls["run_aiperf"]
    assert raw == (run_dirs[0] / "payloads.jsonl").read_bytes()
    endpoint = config["roles"]["slm_candidates"][-1]["endpoint"]
    assert (args["base_url"], args["url"], args["model"], args["concurrency"]) == (
        endpoint["base_url"], endpoint["base_url"][:-3], "c0", 8)
    assert len(args["warmup"]) == 2 and calls["app.run"] == [{}]
    assert (run_dirs[0] / "logs" / "aiperf.log").read_bytes() == b"ok"
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert (manifest["where"], manifest["status"], manifest["ready_after_s"]) == ("modal", "done", 42.0)
    assert manifest["served_model"] == {"id": "c0", "root": "/adapters/abc", "parent": TINY_NAME}
