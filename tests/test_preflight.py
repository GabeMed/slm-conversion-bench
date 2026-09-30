"""`bench preflight`: every SPEC 7.1 check with its action, and P-4 proven against fake vLLM endpoints (one
that passes, one where the adapter changes nothing, one that diverges from HF-PEFT). The real P-4 runs on
the morning of Day 1."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from bench import cli, paths
from bench.preflight import (FAIL, PASS, PENDING, PreflightError, check_agent_runs, check_call_sites, check_data,
                             check_lora_parity, check_pilot_spend, check_schedule, check_teacher_terms, compare,
                             lora_parity, parity_verdict, served_generate)
from test_train_fixtures import TINY, TINY_NAME, fake_adapter, make_s5_repo, save

STOP = [2]


def gen(tokens, alternatives=(99,)):
    """A generation whose top-k at each position is the chosen token and the alternatives."""
    return {"tokens": list(tokens), "top": [[t, *alternatives] for t in tokens]}


# ---------------------------------------------------------------- the parity rule

def test_identical_generations_match_exactly():
    assert compare(gen([5, 6, 7]), gen([5, 6, 7]), STOP) == {"exact": True, "close": True, "first_difference": None}


def test_a_numerical_near_tie_at_the_first_difference_still_matches():
    served = {"tokens": [5, 8, 9], "top": [[5], [8, 6], [9]]}
    reference = {"tokens": [5, 6, 7], "top": [[5], [6, 8], [7]]}
    assert compare(served, reference, STOP) == {"exact": False, "close": True, "first_difference": 1}


def test_a_token_outside_the_other_sides_top_k_is_a_divergence():
    served = {"tokens": [5, 8], "top": [[5], [8, 6]]}
    reference = {"tokens": [5, 6], "top": [[5], [6, 3]]}  # 8 is not among the reference's candidates
    assert compare(served, reference, STOP)["close"] is False


def test_stop_tokens_end_a_generation_and_an_early_stop_is_judged_as_a_near_tie():
    assert compare(gen([5, 6, 2, 0, 0]), gen([5, 6, 2]), STOP)["exact"] is True
    stops_early = {"tokens": [5, 2], "top": [[5], [2, 6]]}
    continues = {"tokens": [5, 6, 7], "top": [[5], [6, 2], [7]]}
    assert compare(stops_early, continues, STOP) == {"exact": False, "close": True, "first_difference": 1}
    without_top = {"tokens": [5], "top": [[5]]}  # the server gave no top-k where it stopped: not shown close
    assert compare(without_top, continues, STOP)["close"] is False


def _four(n, served_adapter, ref_adapter=None):
    base = [gen([10 + i, 20, 30]) for i in range(n)]
    adapter = [gen([40 + i, 50, 60]) for i in range(n)]
    return base, served_adapter(base, adapter), base, ref_adapter(base, adapter) if ref_adapter else adapter


def test_verdict_pass_when_the_adapter_changes_the_output_and_matches_peft():
    verdict = parity_verdict(*_four(3, lambda b, a: a), STOP)
    assert verdict["status"] == PASS and verdict["adapter_changes_output"] == 3 and verdict["adapter_matches_peft"] == 3


def test_verdict_when_vllm_ignores_the_adapter():
    verdict = parity_verdict(*_four(3, lambda b, a: b), STOP)
    assert verdict["status"] == FAIL and "ignores the adapter" in verdict["diagnosis"]
    assert "reserve" in verdict["action"]


def test_verdict_when_the_served_adapter_diverges_from_peft():
    verdict = parity_verdict(*_four(3, lambda b, a: [gen([40 + i, 51, 61]) for i in range(3)]), STOP)
    assert verdict["status"] == FAIL and "diverges from HF-PEFT" in verdict["diagnosis"]
    assert "reserve" in verdict["action"] and verdict["base_matches_hf"] == 3


def test_verdict_when_the_serving_itself_disagrees_does_not_blame_the_lora():
    base = [gen([10, 20]), gen([11, 21])]
    other = [gen([70, 80]), gen([71, 81])]
    adapter = [gen([40, 50]), gen([41, 51])]
    verdict = parity_verdict(other, [gen([90, 91]), gen([92, 93])], base, adapter, STOP)
    assert verdict["status"] == FAIL and "even without the adapter" in verdict["diagnosis"]
    assert "not decided" in verdict["action"]


def test_verdict_when_the_adapter_changes_nothing_even_in_peft_is_uninformative():
    base = [gen([10, 20])]
    verdict = parity_verdict(base, base, base, base, STOP)
    assert verdict["status"] == FAIL and "uninformative" in verdict["diagnosis"] and "not decided" in verdict["action"]


def test_the_verdict_needs_four_generations_per_prompt():
    with pytest.raises(PreflightError):
        parity_verdict([gen([1])], [], [gen([1])], [gen([1])], STOP)


# ---------------------------------------------------------------- P-4 against fake vLLM endpoints

class FakeVLLM:
    """An OpenAI-compatible chat endpoint answering greedily from a script {model: [tokens per prompt]},
    with top-k logprobs as vLLM's `return_tokens_as_token_ids` gives them."""

    def __init__(self, script, prompts):
        self.script, self.prompts, self.requests = script, prompts, []
        self.token_name = lambda t: f"token_id:{t}"  # vLLM with return_tokens_as_token_ids
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
                i = fake.prompts.index(body["messages"])
                generation = fake.script[body["model"]][i]
                name = fake.token_name
                content = [{"token": name(t), "logprob": -0.1,
                            "top_logprobs": [{"token": name(x), "logprob": -1.0} for x in top]}
                           for t, top in zip(generation["tokens"], generation["top"])]
                answer = json.dumps({"choices": [{"message": {"role": "assistant", "content": "x"},
                                                  "logprobs": {"content": content}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def parity_repo(tmp_path, monkeypatch):
    _, config_path, config = make_s5_repo(tmp_path, monkeypatch)
    fake_adapter("c0", config)
    prompts = [json.loads(line)["prompt"] for line in (paths.ROOT / "train/datasets/c0.jsonl").read_text().splitlines()]
    return config_path, config, prompts


def _reference(prompts):
    calls = []

    def reference(base, got_prompts, template_kwargs, max_new_tokens, top_k):
        calls.append((base, got_prompts, template_kwargs, max_new_tokens, top_k))
        return {"stop_ids": STOP, "base": [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                "adapter": [gen([40 + i, 50, 60]) for i in range(len(prompts))]}
    return reference, calls


SERVED = {
    "passes": lambda i: gen([40 + i, 50, 60]),
    "adapter changes nothing": lambda i: gen([10 + i, 20, 30]),
    "diverges from PEFT": lambda i: gen([40 + i, 51, 61]),
}


@pytest.mark.parametrize("fake, status, diagnosis", [
    ("passes", PASS, "reproduces HF-PEFT"),
    ("adapter changes nothing", FAIL, "vLLM ignores the adapter"),
    ("diverges from PEFT", FAIL, "diverges from HF-PEFT"),
])
def test_p4_against_a_fake_vllm(parity_repo, monkeypatch, fake, status, diagnosis):
    config_path, config, prompts = parity_repo
    server = FakeVLLM({TINY_NAME: [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                       "c0": [SERVED[fake](i) for i in range(len(prompts))]}, prompts)
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"].update({"base_url": server.base_url, "api_key_env": "TEST_SLM_KEY"})
        monkeypatch.setenv("TEST_SLM_KEY", "k-123")
        reference, calls = _reference(prompts)
        verdict = lora_parity(save(config, config_path), "c0", "local", reference=reference)
    finally:
        server.close()
    assert verdict["status"] == status and diagnosis in verdict["diagnosis"]
    assert verdict["served"] == {"base": TINY_NAME, "adapter": "c0"} and verdict["n_prompts"] == len(prompts)
    # the reference renders the same prompts with the serving kwargs, from the pinned base
    settings = config["preflight"]["lora_parity"]
    assert calls == [(TINY, prompts, {"enable_thinking": False}, settings["max_new_tokens"], settings["top_logprobs"])]
    # every served request is greedy, with top-k as token ids, and carries the configured key
    assert {r["auth"] for r in server.requests} == {"Bearer k-123"}
    assert {r["path"] for r in server.requests} == {"/v1/chat/completions"}
    for r in server.requests:
        assert r["body"]["temperature"] == 0.0 and r["body"]["logprobs"] is True
        assert r["body"]["return_tokens_as_token_ids"] is True
        assert r["body"]["top_logprobs"] == settings["top_logprobs"] and r["body"]["max_tokens"] == settings["max_new_tokens"]
    assert len(server.requests) == 2 * len(prompts)


def test_served_generate_refuses_a_server_that_does_not_return_token_ids(parity_repo):
    _, _, prompts = parity_repo

    server = FakeVLLM({"m": [gen([1])] * len(prompts)}, prompts)
    try:
        assert served_generate(server.base_url, "EMPTY", "m", prompts[0], 4, 2, 10) == gen([1])
        server.token_name = lambda t: f"piece{t}"  # a server that returns text pieces
        with pytest.raises(PreflightError, match="not a token id"):
            served_generate(server.base_url, "EMPTY", "m", prompts[0], 4, 2, 10)
    finally:
        server.close()


def test_p4_is_pending_until_asked_and_fails_with_its_reason_when_nothing_is_served(parity_repo):
    _, config, _ = parity_repo
    assert check_lora_parity(config, None, "modal")["status"] == PENDING
    failed = check_lora_parity(config, "c0", "local")  # no base_url for the candidate
    assert failed["status"] == FAIL and "base_url is not set" in failed["evidence"]["error"]
    assert "not decided" in failed["action"]


# ---------------------------------------------------------------- SPEC 7.1

def _agent_run(run_id, split, status="done", sites=("agent_ir", "extract_keywords")):
    run_dir = paths.RUNS / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(json.dumps({"run_id": run_id, "split": split, "status": status,
                                                       "call_sites_seen": list(sites)}))


def test_teacher_terms(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_teacher_terms(config)["status"] == PASS
    config["roles"]["production_llm"]["terms"] = None
    failed = check_teacher_terms(config)
    assert failed["status"] == FAIL and failed["evidence"]["missing"] == ["weights_license", "provider_terms", "checked_on"]
    assert "reserve" in failed["action"]


def test_agent_runs_and_the_call_site_record(tmp_path, monkeypatch):
    make_s5_repo(tmp_path, monkeypatch)
    assert check_agent_runs()["status"] == FAIL
    _agent_run("agent-B0-train-1", "train")
    _agent_run("agent-B0-train-2", "train", status="failed", sites=("revise",))
    assert check_agent_runs()["status"] == PASS
    pending = check_call_sites()
    assert pending["status"] == PENDING and pending["evidence"]["without_done_runs"] == ["calib"]
    _agent_run("agent-B0-calib-1", "calib", sites=("agent_ss",))
    done = check_call_sites()
    assert done["status"] == PASS
    assert done["evidence"]["call_sites"] == {"train": ["agent_ir", "extract_keywords"], "calib": ["agent_ss"]}


def test_data_check_passes_on_pinned_data_and_fails_on_overlapping_splits(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_data(config)["status"] == PASS
    splits = json.loads(paths.SPLITS.read_text())
    paths.SPLITS.write_text(json.dumps({**splits, "test": ["9", "1"]}))
    failed = check_data(config)
    assert failed["status"] == FAIL and failed["evidence"]["test"]["overlap_with_train_or_calib"] == ["1"]
    paths.SPLITS.unlink()
    assert "error" in check_data(config)["evidence"]


def test_pilot_spend_is_pending_on_what_other_fronts_provide(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_pilot_spend(config)["status"] == PENDING


def test_schedule_projects_the_measured_training_time_over_every_dataset(tmp_path, monkeypatch):
    _, _, config = make_s5_repo(tmp_path, monkeypatch)
    assert check_schedule(config)["status"] == PENDING
    local = fake_adapter("c0", config).parent / "manifest.json"
    manifest = json.loads(local.read_text())
    local.write_text(json.dumps({**manifest, "where": "local", "stats": {"train_seconds": 1, "examples_seen": 1}}))
    assert check_schedule(config)["status"] == PENDING  # a CPU run does not project GPU time
    local.write_text(json.dumps({**manifest, "where": "modal", "stats": {"train_seconds": 100, "examples_seen": 50}}))
    waiting = check_schedule(config)
    epochs = config["train"]["sft"]["num_train_epochs"]
    assert waiting["evidence"]["dataset_rows"] == 8 and waiting["evidence"]["seconds_per_example"] == 2
    assert waiting["evidence"]["projected_train_hours"] == round(2 * 8 * epochs / 3600, 2)
    assert waiting["status"] == PENDING and "load test" in waiting["evidence"]["needs"]
    (paths.RUNS / "loadtest-x").mkdir(parents=True)
    (paths.RUNS / "loadtest-x" / "manifest.json").write_text(json.dumps({"run_id": "loadtest-x", "concurrency": 4, "status": "done"}))
    assert check_schedule(config)["status"] == PASS
    config["preflight"]["schedule"]["train_hours_max"] = 0.001
    assert check_schedule(config)["status"] == FAIL


def test_bench_preflight_writes_the_report_and_exits_1_unless_everything_passes(tmp_path, monkeypatch, capsys):
    _, config_path, _ = make_s5_repo(tmp_path, monkeypatch)
    assert cli.main(["preflight", "--config", str(config_path)]) == 1
    report_path = next(paths.RUNS.glob("preflight-*/report.json"))
    report = json.loads(report_path.read_text())
    assert [c["id"] for c in report["checks"]] == ["agent_end_to_end", "call_sites_registered", "data_ids_and_gold",
                                                   "teacher_terms", "pilot_spend", "schedule", "lora_parity"]
    assert report["all_pass"] is False and all(c["action"] for c in report["checks"])
    assert str(paths.ROOT) not in report_path.read_text()
    assert "lora_parity" in capsys.readouterr().out


def test_p4_on_modal_asks_the_gpu_reference_for_the_adapter_by_its_sha256(parity_repo, monkeypatch):
    from test_train_fixtures import fake_modal_app

    config_path, config, prompts = parity_repo
    server = FakeVLLM({TINY_NAME: [gen([10 + i, 20, 30]) for i in range(len(prompts))],
                       "c0": [gen([40 + i, 50, 60]) for i in range(len(prompts))]}, prompts)
    reference, _ = _reference(prompts)
    calls = fake_modal_app(monkeypatch, "train", peft_reference=lambda sha, *args: reference(*args))
    try:
        config["roles"]["slm_candidates"][-1]["endpoint"]["base_url"] = server.base_url
        verdict = lora_parity(save(config, config_path), "c0", "modal")
    finally:
        server.close()
    manifest = json.loads((paths.ROOT / "train/adapters/c0/manifest.json").read_text())
    (sha, base, got_prompts, kwargs, _, _), = calls["peft_reference"]
    assert sha == manifest["adapter_sha256"] and base == TINY and got_prompts == prompts
    assert verdict["status"] == PASS and verdict["reference_on"] == "modal"
