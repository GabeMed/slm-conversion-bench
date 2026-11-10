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
from bench.loadtest import (LoadtestError, aiperf_command, aiperf_config, build_payloads, cold_cache, composition,
                           draw_level, env_headers, largest_remainder, level_counts, reset_prefix_cache, scrub_secrets, select_traffic,
                           server_root, shuffle_pool, wait_ready)
from synthetic import make_repo
from test_train_fixtures import TINY, TINY_NAME, make_s5_repo, save

CALLS = json.loads((Path(__file__).parent / "fixtures_calls.json").read_text())


def test_each_call_becomes_the_body_the_agents_client_sends_to_the_engines_model():
    params = {"max_tokens": 64, "timeout_s": 600, "top_p": 0.9}
    # the temperature is the call site's in the configuration, not what C1 recorded: a provider that
    # forces its own makes the two differ (the fixture's lines all record 0.0)
    sites = {"agent_ss": {"temperature": 0.3}, "select_tables": {"temperature": 0.7}}
    payloads = build_payloads(CALLS, "c3", params, False, sites)
    assert len(payloads) == len(CALLS)
    for call, body in zip(CALLS, payloads):
        assert body == {"model": "c3", "messages": call["prompt_messages"],
                        "temperature": sites[call["call_site"]]["temperature"], "max_tokens": 64, "top_p": 0.9}
    assert {body["temperature"] for body in payloads} == {0.3, 0.7} and {c["temperature"] for c in CALLS} == {0.0}
    streamed = build_payloads(CALLS[:1], "c3", {"max_tokens": None, "timeout_s": 1}, True, sites)[0]
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
    replayed = build_payloads([call], spec["model"], spec["params"], False, config["call_sites"])[0]
    assert {k: sent[k] for k in replayed} == replayed
    assert set(sent) - set(replayed) <= {"n", "stream"} and not sent.get("stream")


def test_aiperf_is_given_the_server_root_and_credentials_only_as_references():
    import sys

    assert server_root("http://h:8000/v1/") == "http://h:8000"
    with pytest.raises(LoadtestError, match="does not end in /v1"):
        server_root("http://h:8000/api")
    cmd = aiperf_command(sys.executable, TINY)
    assert cmd[1:] == ["profile", "--config", "aiperf.yaml", "--tokenizer", TINY["repo"], "--tokenizer-revision",
                       TINY["revision"], "--output-artifact-dir", ".", "--ui-type", "none"]
    config = aiperf_config("http://h:8000", "c3", 8, 500, 600, False, "SLM_API_KEY",
                           {"Modal-Key": "SLM_MODAL_KEY", "Modal-Secret": "SLM_MODAL_SECRET"})
    endpoint = config["benchmark"]["endpoint"]
    assert endpoint["apiKey"] == "${SLM_API_KEY}"  # AIPerf resolves it from its environment
    assert endpoint["headers"] == {"Modal-Key": "${SLM_MODAL_KEY}", "Modal-Secret": "${SLM_MODAL_SECRET}"}
    assert (endpoint["url"], endpoint["type"], endpoint["useServerTokenCount"]) == ("http://h:8000", "chat", True)
    assert "streaming" not in endpoint
    assert config["benchmark"]["dataset"] == {"type": "file", "name": "payloads", "path": "payloads.jsonl",
                                              "format": "raw_payload", "sampling": "sequential"}
    assert config["benchmark"]["phases"] == {"type": "concurrency", "name": "profiling", "concurrency": 8, "requests": 500}
    bare = aiperf_config("http://h:8000", "c3", 1, 5, 60, True, None, None)["benchmark"]["endpoint"]
    assert bare["streaming"] is True and "apiKey" not in bare and "headers" not in bare
    with pytest.raises(LoadtestError, match="AIPerf is not installed"):
        aiperf_command(None, TINY)


def test_headers_come_from_the_variables_headers_env_names():
    assert env_headers({"Modal-Key": "MK", "Modal-Secret": "MS"}, {"MK": "k", "MS": "s"}) == {"Modal-Key": "k", "Modal-Secret": "s"}
    assert env_headers(None, {}) == {}
    with pytest.raises(LoadtestError, match=r"\['MS'\]"):
        env_headers({"Modal-Key": "MK", "Modal-Secret": "MS"}, {"MK": "k"})


def test_credential_values_are_scrubbed_from_every_artifact(tmp_path):
    (tmp_path / "logs").mkdir()
    (tmp_path / "profile_export_aiperf.json").write_text('{"headers": {"Modal-Key": "mk-123", "x": "ms-456"}}')
    (tmp_path / "logs" / "aiperf.log").write_text("sent mk-123")
    (tmp_path / "clean.txt").write_text("nothing")
    assert scrub_secrets(tmp_path, ["mk-123", "ms-456", ""]) == ["logs/aiperf.log", "profile_export_aiperf.json"]
    assert json.loads((tmp_path / "profile_export_aiperf.json").read_text()) == {
        "headers": {"Modal-Key": "<redacted>", "x": "<redacted>"}}


class FakeServer:
    """GET /v1/models, listing `models` once `ready_after` requests have been made; POST records the
    warm-up calls."""

    def __init__(self, models, ready_after=0):
        self.models, self.ready_after, self.seen, self.posted, self.headers = models, ready_after, [], [], []
        self.resets, self.reset_answers = [], []  # POST /reset_prefix_cache: each an answer body, or an HTTP status
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
                fake.headers.append(dict(self.headers))
                listed = fake.models if len(fake.seen) > fake.ready_after else []
                self._answer({"data": [{"id": m, "root": f"/adapters/{m}", "parent": "base"} for m in listed]})

            def do_POST(self):
                if self.path == "/reset_prefix_cache":
                    fake.resets.append(self.headers.get("Authorization"))
                    answer = fake.reset_answers.pop(0) if fake.reset_answers else {"success": True}
                    if isinstance(answer, int):
                        self.send_response(answer)
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    return self._answer(answer)
                fake.posted.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                fake.headers.append(dict(self.headers))
                self._answer({"choices": []})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


def test_wait_ready_waits_for_the_model_to_be_listed():
    server = FakeServer(["c3"], ready_after=2)
    try:
        waited, card = wait_ready(server.base_url, {"Authorization": "Bearer k"}, "c3", timeout_s=5, poll_s=0.01)
        assert waited >= 0 and card == {"id": "c3", "root": "/adapters/c3", "parent": "base"}
        assert len(server.seen) == 3 and server.seen[0] == ("/v1/models", "Bearer k")
        with pytest.raises(LoadtestError, match="c9 not served"):
            wait_ready(server.base_url, {}, "c9", timeout_s=0.05, poll_s=0.01)
    finally:
        server.close()


def _source_run(run_id="agent-B4-calib-src", split="calib", calls=CALLS, arm="B4"):
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "arm": arm, "split": split, "status": "done"}))
    (run_dir / "calls.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return run_id


def test_largest_remainder_splits_a_count_in_proportion():
    # 10 over 6/3/1: quotas 6, 3, 1 exactly
    assert largest_remainder({"c0": 60, "c1": 30, "c2": 10}, 10) == {"c0": 6, "c1": 3, "c2": 1}
    # 5 over 13128/614/200/2197 (B0's routed sizes): quotas 4.07, 0.19, 0.06, 0.68 -> 4, 0, 0, 1
    assert largest_remainder({"c0": 13128, "c1": 614, "c2": 200, "c3": 2197}, 5) == {"c0": 4, "c1": 0, "c2": 0, "c3": 1}
    # a tie of remainders goes to the smaller name
    assert largest_remainder({"b": 1, "a": 1}, 1) == {"a": 1, "b": 0}
    assert sum(largest_remainder({"x": 7, "y": 11, "z": 13}, 17).values()) == 17


def test_each_level_draws_its_own_calls_stratified_by_cluster_none_twice():
    strata = ["c0"] * 60 + ["c1"] * 30 + ["c2"] * 10
    first, again, second = draw_level(strata, 20, 7, 0), draw_level(strata, 20, 7, 0), draw_level(strata, 20, 7, 1)
    assert first == again and first["positions"] != second["positions"]  # seeded per level
    for drawn in (first, second):
        assert len(set(drawn["positions"])) == 20 and drawn["short_pool"] is False
        assert drawn["strata"] == {"c0": 12, "c1": 6, "c2": 2}
        assert sorted(strata[i] for i in drawn["positions"]) == ["c0"] * 12 + ["c1"] * 6 + ["c2"] * 2
    # levels draw independently: a small pool's calls come back in a later level
    assert set(first["positions"]) & set(second["positions"])
    # a pool smaller than the level: all of it, once, flagged
    whole = draw_level(strata[:7], 20, 7, 0)
    assert sorted(whole["positions"]) == list(range(7)) and whole["short_pool"] is True


def test_each_level_takes_its_own_count_where_the_configuration_gives_one():
    settings = {"request_count": 500, "request_count_by_level": {1: 200, "2": 200}}
    assert level_counts([1, 2, 4, 128], settings) == [200, 200, 500, 500]
    assert level_counts([4], {"request_count": 500}) == [500]


def test_the_cache_starts_cold_always_or_never():
    assert (cold_cache("always"), cold_cache("never")) == (True, False)
    with pytest.raises(LoadtestError, match="cold_prefix_cache must be one of"):
        cold_cache("reduced")


B4 = {"run_id": "agent-B4-calib-x", "arm": "B4", "split": "calib"}
B3 = {"run_id": "agent-B3-calib-x", "arm": "B3", "split": "calib"}


def test_each_slm_engine_is_load_tested_on_the_calls_its_own_arm_sent_it():
    a0, a1 = "slm:q+lora:a0", "slm:q+lora:a1"
    calls = [{"cluster": "c0", "engine": a0, "i": 0}, {"cluster": "c1", "engine": a1, "i": 1},
             {"cluster": "c0", "engine": a0, "i": 2}]
    # an adapter: the calls B4 sent it, one cluster's, and that cluster is the sweep's traffic
    adapter = select_traffic(calls, a0, B4)
    assert [c["i"] for c in adapter["calls"]] == [0, 2] and (adapter["traffic"], adapter["kind"]) == ("c0", "adapter")
    # as deployed: all of B4's calls, each to its own adapter, the traffic all of it
    deployed = select_traffic(calls, "slm:q", B4, as_deployed=True)
    assert deployed["calls"] == calls and (deployed["traffic"], deployed["kind"]) == ("all", "as_deployed")
    # the base: every call of B3
    base = [{"cluster": None, "engine": "slm:q", "i": 0}]
    assert select_traffic(base, "slm:q", B3) == {"calls": base, "traffic": "all", "kind": "base"}
    # a non-SLM engine (the local smoke server) replays every call it is given
    assert select_traffic(calls, "production_llm", B4)["kind"] == "replay"


def test_an_slm_engine_is_never_load_tested_on_another_arms_traffic():
    a0 = "slm:q+lora:a0"
    calls = [{"cluster": "c0", "engine": a0}]
    b0 = {"run_id": "agent-B0-calib-x", "arm": "B0", "split": "calib"}
    with pytest.raises(LoadtestError, match="on B4's own end-to-end execution on calib, not on runs/agent-B0-calib-x"):
        select_traffic(calls, a0, b0)
    with pytest.raises(LoadtestError, match=r"on B4's own end-to-end execution on calib.*\(B4 on train\)"):
        select_traffic(calls, a0, {**B4, "split": "train"})
    with pytest.raises(LoadtestError, match="on B3's own end-to-end execution on calib"):
        select_traffic([{"cluster": None, "engine": "slm:q"}], "slm:q", B4)
    with pytest.raises(LoadtestError, match="sent no call to slm:q\\+lora:a9"):
        select_traffic(calls, "slm:q+lora:a9", B4)
    with pytest.raises(LoadtestError, match=r"the calls of clusters \['c0', 'c1'\]: an adapter serves one cluster"):
        select_traffic(calls + [{"cluster": "c1", "engine": a0}], a0, B4)
    with pytest.raises(LoadtestError, match=r"sent calls to \['production_llm', 'slm:q'\], not all to slm:q"):
        select_traffic([{"engine": "slm:q"}, {"engine": "production_llm"}], "slm:q", B3)
    with pytest.raises(LoadtestError, match="give the base"):
        select_traffic(calls, a0, B4, as_deployed=True)
    with pytest.raises(LoadtestError, match=r"sent calls to \['production_llm'\], not adapters of slm:q"):
        select_traffic(calls + [{"cluster": "c1", "engine": "production_llm"}], "slm:q", B4, as_deployed=True)


def test_the_composition_counts_clusters_call_sites_and_engines():
    calls = [{"cluster": "c1", "call_site": "revise", "engine": "e1"}, {"cluster": None, "call_site": "agent_ir", "engine": "e0"},
             {"cluster": "c1", "call_site": "select_tables", "engine": "e1"}]
    assert composition(calls) == {"clusters": {"c1": 2, "none": 1},
                                  "call_sites": {"agent_ir": 1, "revise": 1, "select_tables": 1},
                                  "engines": {"e0": 1, "e1": 2}}


def test_as_deployed_each_line_names_the_adapter_that_served_the_call():
    sites = {"agent_ss": {"temperature": 0.0}, "select_tables": {"temperature": 0.0}}
    calls = [dict(CALLS[0], engine="slm:q+lora:a0"), dict(CALLS[1], engine="slm:q+lora:a1")]
    bodies = build_payloads(calls, "q", {"max_tokens": 8}, False, sites, {"slm:q+lora:a0": "a0", "slm:q+lora:a1": "a1"})
    assert [b["model"] for b in bodies] == ["a0", "a1"]


def test_the_prefix_cache_reset_must_say_success_and_is_retried_while_blocks_are_held(monkeypatch):
    monkeypatch.setattr(loadtest.time, "sleep", lambda s: None)
    server = FakeServer(["c0"])
    try:
        server.reset_answers = [{"success": False}, {"success": True}]
        assert reset_prefix_cache(server.base_url, {"Authorization": "Bearer k"}, 5) == {"attempts": 2}
        assert server.resets == ["Bearer k", "Bearer k"]  # at the server root, not under /v1
        server.reset_answers = [{"success": False}] * 3
        with pytest.raises(LoadtestError, match="did not empty the prefix cache in 3 attempts"):
            reset_prefix_cache(server.base_url, {}, 5, attempts=3)
        server.reset_answers = [404]
        with pytest.raises(LoadtestError, match="VLLM_SERVER_DEV_MODE=1; serve with it, or restart vLLM between levels"):
            reset_prefix_cache(server.base_url, {}, 5)
    finally:
        server.close()


def test_the_shuffle_is_the_seeds():
    payloads = [{"i": i} for i in range(50)]
    once, again = shuffle_pool(payloads, 20260935), shuffle_pool(payloads, 20260935)
    assert once == again and once != payloads and sorted(p["i"] for p in once) == list(range(50))
    assert shuffle_pool(payloads, 20260936) != once  # another seed, another order


def test_every_level_holds_most_questions_and_every_database():
    """A source as C1 writes it: each question's calls together, the questions of a database together. Each
    level draws at random from the whole pool, so it holds most questions and every database."""
    strata = ["c0" if c < 8 else "c1" for q in range(30) for c in range(10)]
    pool = [{"q": q, "db": f"db{q // 8}"} for q in range(30) for c in range(10)]
    for index in range(6):
        chosen = [pool[i] for i in draw_level(strata, 40, 20260935, index)["positions"]]
        assert len({p["q"] for p in chosen}) >= 15 and len({p["db"] for p in chosen}) == 4


# as B4 recorded them on calib: the router sent these calls, of cluster c0, to its adapter
SOURCE = [dict(CALLS[i % 2], call_id=f"00000000-0000-4000-8000-{i:012d}", invocation_key=f"k:{i}", cluster="c0",
               engine=f"slm:{TINY_NAME}+lora:c0", model_role="slm") for i in range(12)]
# as B3 recorded them on calib: every call to the base
BASE = [dict(c, cluster=None, engine=f"slm:{TINY_NAME}") for c in SOURCE]


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A synthetic repo whose tiny SLM candidate is 'served' by a fake that lists it and its adapter."""
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    server = FakeServer([TINY_NAME, "c0"])
    config["roles"]["slm_candidates"][-1]["endpoint"].update({
        "base_url": server.base_url, "headers_env": {"Modal-Key": "T_MODAL_KEY", "Modal-Secret": "T_MODAL_SECRET"}})
    monkeypatch.setenv("T_MODAL_KEY", "mk-live")
    monkeypatch.setenv("T_MODAL_SECRET", "ms-live")
    monkeypatch.setattr(loadtest, "local_aiperf", lambda: __import__("sys").executable)  # a file that exists
    config["loadtest"].update({"concurrency": [1, 4], "request_count": 5, "warmup_request_count": 2,
                               "request_count_by_level": {}, "cold_prefix_cache": "never"})
    config["seeds"]["loadtest_shuffle"] = 7
    yield config_path, save(config, config_path), server
    server.close()


def _fake_aiperf(returncode=0, calls=None):
    def run(run_dir, cmd):
        import yaml

        (calls if calls is not None else []).append((run_dir, cmd, yaml.safe_load((run_dir / "aiperf.yaml").read_text())))
        assert (run_dir / "payloads.jsonl").is_file()
        (run_dir / "aiperf.log").write_text("headers Modal-Key: mk-live")  # as AIPerf leaks Modal-Key
        if returncode == 0:
            (run_dir / "profile_export_aiperf.json").write_text(json.dumps(
                {"request_count": {"avg": 6}, "overall_usage_prompt_cache_read_pct": {"unit": "%", "avg": 37.5}}))
        return returncode
    return run


def test_one_run_per_concurrency_level_each_with_its_manifest(served, monkeypatch):
    config_path, config, server = served
    calls = []
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf(calls=calls))
    other = [dict(c, call_id=f"00000000-0000-4000-8001-{i:012d}", invocation_key=f"o:{i}", cluster="c1",
                  engine=f"slm:{TINY_NAME}+lora:c1") for i, c in enumerate(SOURCE[:5])]
    source = _source_run(calls=SOURCE[:6] + other + SOURCE[6:])  # another cluster's calls, never replayed here
    run_dirs = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", source, "local")
    assert [json.loads((d / "manifest.json").read_text())["concurrency"] for d in run_dirs] == [1, 4]
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["status"] == "done" and manifest["model"] == "c0" and manifest["engine"] == f"slm:{TINY_NAME}+lora:c0"
    assert manifest["gpu"] == config["serving"]["gpu"] and manifest["prefix_cache"] is True
    assert manifest["source_run_id"] == source and manifest["source_split"] == "calib"
    assert (manifest["source_calls"], manifest["source_calls_all"], manifest["n_questions"]) == (12, 17, 1)
    assert (manifest["traffic"], manifest["traffic_kind"], manifest["cluster"]) == ("c0", "adapter", "c0")
    assert manifest["models"] is None and manifest["short_pool"] is False and manifest["strata"] == {"c0": 5}
    assert manifest["stratified_by"] == "cluster"
    assert (manifest["request_count"], manifest["request_count_configured"]) == (5, 5)
    assert manifest["cold_prefix_cache"] is False and manifest["prefix_cache_reset"] is None and not server.resets
    assert manifest["composition"]["source"]["clusters"] == {"c0": 12}
    assert sum(manifest["composition"]["level"]["call_sites"].values()) == 5
    assert manifest["tokenizer"] == TINY and manifest["where"] == "local" and manifest["aiperf_returncode"] == 0
    assert manifest["run_id"].startswith(f"loadtest-slm_{TINY_NAME}_lora_c0-c1-")
    assert manifest["served_model"] == {"id": "c0", "root": "/adapters/c0", "parent": "base"}  # what the server said
    assert manifest["observed"] == {"gpus": None, "vllm_command": None, "prompt_cache_read_pct": 37.5}
    levels = [[json.loads(line) for line in (d / "payloads.jsonl").read_text().splitlines()] for d in run_dirs]
    assert {p["model"] for p in levels[0]} == {"c0"} and levels[0][0]["max_tokens"] == 64
    # the warm-up calls (the seed's first two, sent before each level), then each level its own seeded draw
    expected = build_payloads(SOURCE, "c0", config["roles"]["slm_candidates"][-1]["params"], False, config["call_sites"])
    strata = ["c0"] * 12
    assert levels == [[expected[i] for i in draw_level(strata, 5, 7, k)["positions"]] for k in (0, 1)]
    assert server.posted == shuffle_pool(expected, 7)[:2] * 2
    second = json.loads((run_dirs[1] / "manifest.json").read_text())
    assert (manifest["payload_shuffle_seed"], manifest["level_seed"], second["level_seed"]) == (7, "7:0", "7:1")
    _, cmd, aiperf_yaml = calls[1]
    phases, endpoint = aiperf_yaml["benchmark"]["phases"], aiperf_yaml["benchmark"]["endpoint"]
    assert (phases["concurrency"], aiperf_yaml["benchmark"]["model"]) == (4, "c0")
    assert endpoint["headers"] == {"Modal-Key": "${T_MODAL_KEY}", "Modal-Secret": "${T_MODAL_SECRET}"}
    assert not {"mk-live", "ms-live"} & set(cmd)  # never on AIPerf's command line
    # the model list and the warm-up carry the proxy-auth headers; the leaked value is scrubbed afterwards
    assert {(h["Modal-Key"], h["Modal-Secret"]) for h in server.headers} == {("mk-live", "ms-live")}
    assert manifest["secrets_redacted_in"] == ["aiperf.log"] and "mk-live" not in (run_dirs[0] / "aiperf.log").read_text()
    assert manifest["source_arm"] == "B4" and manifest["levels"] == [1, 4] and manifest["level_index"] == 0
    assert second["sweep_id"] == manifest["sweep_id"] and second["level_index"] == 1


def test_a_failed_level_is_recorded_and_stops_the_sweep(served, monkeypatch):
    config_path, _, _ = served
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf(returncode=1))
    with pytest.raises(LoadtestError, match="AIPerf failed at concurrency 1"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run(calls=BASE, arm="B3"), "local")
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
    with pytest.raises(LoadtestError, match="on B3's own end-to-end execution on calib, not on runs/agent-B4-calib-src"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run(calls=SOURCE), "local")
    seedless = {**config, "seeds": {k: v for k, v in config["seeds"].items() if k != "loadtest_shuffle"}}
    with pytest.raises(LoadtestError, match="seeds.loadtest_shuffle is not set"):
        loadtest.loadtest(str(save(seedless, config_path) and config_path), f"slm:{TINY_NAME}+lora:c0",
                          "agent-B4-calib-src", "local")
    save(config, config_path)
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
        "returncode": 0, "ready_after_s": 42.0, "files": artifacts, "secrets_redacted_in": ["profile_export_aiperf.json"],
        "served_model": {"id": "c0", "root": "/adapters/abc", "parent": TINY_NAME, "object": "model"},
        "server_state": {"gpus": ["NVIDIA H200"], "vllm_command": ["vllm", "serve"], "revision": "r", "adapters": {}}})
    source = _source_run(calls=SOURCE)
    monkeypatch.delenv("BENCH_CONFIG", raising=False)
    run_dirs = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", source, "modal", concurrency=[8])
    import os

    assert os.environ["BENCH_CONFIG"] == str(config_path.resolve())  # the Modal app reads the --config given
    (raw, args), = calls["run_aiperf"]
    assert raw == (run_dirs[0] / "payloads.jsonl").read_bytes()
    endpoint = config["roles"]["slm_candidates"][-1]["endpoint"]
    assert (args["base_url"], args["url"], args["model"], args["concurrency"]) == (
        endpoint["base_url"], endpoint["base_url"][:-3], "c0", 8)
    assert len(args["warmup"]) == 2 and calls["app.run"] == [{}] and args["candidate"] == TINY_NAME
    assert (args["models"], args["reset_prefix_cache"], args["request_count"]) == (None, False, 5)
    assert args["headers_env"] == endpoint["headers_env"]  # the variable names, never their values
    assert "mk-live" not in json.dumps(args) and "ms-live" not in json.dumps(args)
    assert (run_dirs[0] / "logs" / "aiperf.log").read_bytes() == b"ok"
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert (manifest["where"], manifest["status"], manifest["ready_after_s"]) == ("modal", "done", 42.0)
    assert manifest["served_model"] == {"id": "c0", "root": "/adapters/abc", "parent": TINY_NAME}
    assert manifest["gpu"] == config["serving"]["gpu"]  # configured...
    assert manifest["observed"]["gpus"] == ["NVIDIA H200"]  # ...and what the server says it got


def test_a_failure_is_recorded_without_local_paths(served, monkeypatch):
    config_path, _, _ = served

    def crash(run_dir, cmd):
        raise OSError(f"cannot open {paths.ROOT}/somewhere")

    monkeypatch.setattr(loadtest, "run_aiperf", crash)
    with pytest.raises(OSError):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run("agent-B3-calib-src", calls=BASE, arm="B3"), "local", concurrency=[1])
    manifest = json.loads(next(paths.RUNS.glob("loadtest-*/manifest.json")).read_text())
    assert manifest["status"] == "failed" and "<repo>/somewhere" in manifest["stopped_by"]
    assert str(paths.ROOT) not in manifest["stopped_by"]


def test_without_aiperf_the_load_test_says_so_before_anything(served, monkeypatch):
    config_path, _, server = served
    monkeypatch.setattr(loadtest, "local_aiperf", lambda: None)
    with pytest.raises(LoadtestError, match="AIPerf is not installed"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run("agent-B3-calib-src", calls=BASE, arm="B3"), "local", concurrency=[1, 4])
    assert not list(paths.RUNS.glob("loadtest-*")) and not server.seen and not server.posted


@pytest.mark.parametrize("code", [401, 403])
def test_refused_credentials_fail_at_once_instead_of_waiting_for_a_cold_start(code):
    import time as clock

    server = FakeServer([])
    handler = server.server.RequestHandlerClass

    def refuse(self):
        server.seen.append((self.path, None))
        self.send_response(code)
        self.send_header("Content-Length", "0")
        self.end_headers()

    handler.do_GET = refuse
    try:
        t0 = clock.monotonic()
        with pytest.raises(LoadtestError, match=f"refused the credentials \\(HTTP {code}\\)"):
            wait_ready(server.base_url, {"Modal-Key": "wrong"}, "c0", timeout_s=60, poll_s=0.01)
        assert clock.monotonic() - t0 < 5 and len(server.seen) == 1
    finally:
        server.close()


def test_a_missing_proxy_auth_variable_stops_the_load_test_before_any_request(served, monkeypatch):
    config_path, _, server = served
    monkeypatch.delenv("T_MODAL_SECRET")
    with pytest.raises(LoadtestError, match="T_MODAL_SECRET"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run("agent-B3-calib-src", calls=BASE, arm="B3"), "local", concurrency=[1])
    assert not server.seen and not server.posted



def test_an_interrupted_aiperf_still_leaves_no_credential(served, monkeypatch):
    config_path, _, _ = served

    def interrupted(run_dir, cmd):
        (run_dir / "profile_export_aiperf.json").write_text('{"Modal-Key": "mk-live"}')
        raise KeyboardInterrupt

    monkeypatch.setattr(loadtest, "run_aiperf", interrupted)
    with pytest.raises(KeyboardInterrupt):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run("agent-B3-calib-src", calls=BASE, arm="B3"), "local", concurrency=[1])
    run_dir = next(paths.RUNS.glob("loadtest-*"))
    assert "mk-live" not in (run_dir / "profile_export_aiperf.json").read_text()


def test_a_warm_up_error_is_a_load_test_error_not_a_traceback(served, monkeypatch, capsys):
    from bench import cli

    config_path, _, server = served
    handler = server.server.RequestHandlerClass

    def refuse(self):
        self.send_response(503)
        self.send_header("Content-Length", "0")
        self.end_headers()

    handler.do_POST = refuse
    _source_run("agent-B3-calib-src", calls=BASE, arm="B3")
    assert cli.main(["loadtest", "--config", str(config_path), "--engine", f"slm:{TINY_NAME}", "--source",
                     "agent-B3-calib-src", "--on", "local", "--concurrency", "1"]) == 2
    assert "warm-up refused" in capsys.readouterr().err and "HTTP 503" in json.loads(
        next(paths.RUNS.glob("loadtest-*/manifest.json")).read_text())["stopped_by"]


def test_as_deployed_b4_is_served_by_one_server_each_call_to_its_own_adapter(served, monkeypatch):
    config_path, config, server = served
    server.models.append("c1")
    calls = []
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf(calls=calls))
    mixed = [dict(c, cluster="c1", engine=f"slm:{TINY_NAME}+lora:c1") if i % 3 == 0 else c for i, c in enumerate(SOURCE)]
    run_dirs = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run(calls=mixed), "local", as_deployed=True)
    manifest = json.loads((run_dirs[0] / "manifest.json").read_text())
    assert manifest["engine"] == f"slm:{TINY_NAME}+as-deployed" and manifest["run_id"].startswith(
        f"loadtest-slm_{TINY_NAME}_as-deployed-c1-")
    assert (manifest["traffic"], manifest["traffic_kind"], manifest["cluster"]) == ("all", "as_deployed", None)
    assert manifest["models"] == ["c0", "c1"] and manifest["composition"]["source"]["clusters"] == {"c0": 8, "c1": 4}
    levels = [[json.loads(line) for line in (d / "payloads.jsonl").read_text().splitlines()] for d in run_dirs]
    replayed = [mixed[i] for k in (0, 1) for i in draw_level([c["cluster"] for c in mixed], 5, 7, k)["positions"]]
    assert [p["model"] for level in levels for p in level] == [c["cluster"] for c in replayed]  # each to its adapter
    # stratified: 5 of 8 c0 and 4 c1 -> quotas 3.33 and 1.67: 3 and 2
    assert manifest["strata"] == {"c0": 3, "c1": 2}
    assert {p["model"] for p in server.posted} <= {"c0", "c1"}  # the warm-up too
    assert calls[0][2]["benchmark"]["model"] == TINY_NAME  # AIPerf's model; each raw line carries its own
    assert ("/v1/models", None) in server.seen


def test_every_level_starts_from_an_empty_prefix_cache_and_a_small_pool_is_reused(served, monkeypatch):
    config_path, config, server = served
    config["loadtest"].update({"cold_prefix_cache": "always", "request_count": 10})
    save(config, config_path)
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf())
    run_dirs = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", _source_run(calls=SOURCE), "local")
    manifests = [json.loads((d / "manifest.json").read_text()) for d in run_dirs]
    # 12 calls: each level draws 10 of them, the second again from all 12
    assert [(m["concurrency"], m["request_count"], m["short_pool"]) for m in manifests] == [(1, 10, False), (4, 10, False)]
    assert all(m["cold_prefix_cache"] for m in manifests)
    assert [m["prefix_cache_reset"] for m in manifests] == [{"attempts": 1}] * 2 and len(server.resets) == 2
    # a pool smaller than a level: replayed whole, once, and flagged
    config["loadtest"]["request_count"] = 50
    save(config, config_path)
    short = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", "agent-B4-calib-src", "local")
    assert [json.loads((d / "manifest.json").read_text())["short_pool"] for d in short] == [True, True]
    assert [json.loads((d / "manifest.json").read_text())["request_count"] for d in short] == [12, 12]
    config["loadtest"]["cold_prefix_cache"] = "never"
    save(config, config_path)
    loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", "agent-B4-calib-src", "local")
    assert len(server.resets) == 4  # never: no reset


def test_a_server_without_the_reset_route_stops_the_cold_sweep(served, monkeypatch):
    config_path, config, server = served
    config["loadtest"]["cold_prefix_cache"] = "always"
    save(config, config_path)
    server.reset_answers = [404]
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf())
    with pytest.raises(LoadtestError, match="VLLM_SERVER_DEV_MODE"):
        loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}+lora:c0", _source_run(calls=SOURCE), "local")
    manifest = json.loads(next(paths.RUNS.glob("loadtest-*/manifest.json")).read_text())
    assert manifest["status"] == "failed" and "VLLM_SERVER_DEV_MODE" in manifest["stopped_by"]


def test_the_base_is_stratified_by_call_site_since_b3_does_not_route(served, monkeypatch):
    config_path, config, server = served
    config["loadtest"].update({"concurrency": [1], "request_count": 6})
    save(config, config_path)
    monkeypatch.setattr(loadtest, "run_aiperf", _fake_aiperf())
    # 12 calls of B3: 9 agent_ss and 3 select_tables (no cluster); a level of 6 holds them in proportion, 4.5 and 1.5
    sites = ["agent_ss"] * 9 + ["select_tables"] * 3
    calls = [dict(c, call_site=site, **({"parsed_output": {"tool": "x"}} if site == "agent_ss" else {}))
             for c, site in zip(BASE, sites)]
    run_dir, = loadtest.loadtest(str(config_path), f"slm:{TINY_NAME}", _source_run("agent-B3-calib-src", calls=calls,
                                                                                  arm="B3"), "local")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["stratified_by"] == "call_site" and manifest["traffic_kind"] == "base"
    assert manifest["strata"] == {"agent_ss": 5, "select_tables": 1}  # largest remainder, ties to the smaller name
    assert manifest["composition"]["level"]["call_sites"] == {"agent_ss": 5, "select_tables": 1}
